"""Tests for EdgeStructuredSexual net_beta formula and transmission dynamics."""

import warnings
import numpy as np
import sciris as sc

import starsim as ss
import stisim as sti
from enroute import EdgeStructuredSexual, StructuredCondomUse

from calc_tolerances import calc_tolerance


class NoopCU(StructuredCondomUse):
    """Inert CondomUse that equalizes the RNG object graph for cross-implementation comparison."""

    def init_pre(self, sim):
        ss.Intervention.init_pre(self, sim)
        return

    def step(self):
        return

    def update_uses(self, network):
        return

    def relative_risk(self, network, disease_beta, disease, uids=None, direction=None):
        return 1


def setup_sim_pars(**kwargs):
    pars = dict(
        n_agents=100,
        start=2000,
        dur=10,
        dt=1 / 12,
        verbose=0,
        diseases=[sti.Chlamydia()],
    )
    pars.update(kwargs)
    return pars


def make_diseases():
    """Shared disease setup for cross-implementation tests."""
    ng = sti.Gonorrhea(beta_m2f=0.038, rel_beta_f2m=0.5, init_prev=0.05)
    ct = sti.Chlamydia(beta_m2f=0.05, rel_beta_f2m=0.5, init_prev=0.05)
    tv = sti.Trichomoniasis(beta_m2f=0.02, rel_beta_f2m=0.5, init_prev=0.05)
    return [ng, ct, tv]


# Enforced condom effectiveness per disease
eff_condom = {"ng": 0.9, "ct": 0.46, "tv": 0.0}


def _run_pair(diseases, eff_condom, condom_data, n_acts, n_agents, tol, warn_tol=None):
    base = dict(n_agents=n_agents, start=2000, dur=10, dt=1 / 12, rand_seed=1, verbose=0)
    act_pars = dict(acts=ss.constant(v=n_acts))

    diseases_ref = [sc.dcp(d) for d in diseases]
    for d in diseases_ref:
        d.pars.eff_condom = eff_condom[d.name]
    sim_ref = ss.Sim(**base, diseases=diseases_ref, networks=sti.StructuredSexual(condom_data=condom_data, **act_pars), interventions=[NoopCU()])
    sim_ref.init()
    sim_ref.run_one_step()

    diseases_edge = [sc.dcp(d) for d in diseases]
    for d in diseases_edge:
        d.pars.eff_condom = eff_condom[d.name]
    sim_edge = ss.Sim(**base, diseases=diseases_edge, networks=EdgeStructuredSexual(**act_pars), interventions=[StructuredCondomUse(condom_data=condom_data)])
    sim_edge.init()
    sim_edge.run_one_step()

    net_ref = list(sim_ref.networks.values())[0]
    net_edge = sim_edge.networks.structuredsexual
    nk = ss.standardize_netkey(list(sim_ref.networks.keys())[0])
    n_edges = len(net_edge.edges.acts)

    all_rows = []
    for name in [d.name for d in diseases]:
        d_ref = sim_ref.diseases[name]
        d_edge = sim_edge.diseases[name]
        betamap_ref = d_ref.validate_beta()
        betamap_edge = d_edge.validate_beta()

        for direction, label in [(0, "m2f"), (1, "f2m")]:
            db_ref = betamap_ref[nk][direction]
            db_edge = betamap_edge[nk][direction]
            db_ref = db_ref.to_prob(sim_ref.t.dt) if isinstance(db_ref, ss.Rate) else float(db_ref)
            db_edge = db_edge.to_prob(sim_edge.t.dt) if isinstance(db_edge, ss.Rate) else float(db_edge)

            beta_ref = net_ref.net_beta(disease_beta=db_ref, disease=d_ref)
            beta_edge = net_edge.net_beta(disease_beta=db_edge, disease=d_edge)

            abs_diff = np.abs(beta_edge - beta_ref)
            nonzero = beta_ref > 0
            rel_diff = np.zeros_like(beta_ref)
            rel_diff[nonzero] = abs_diff[nonzero] / beta_ref[nonzero]

            # tol=None means: derive the tolerance from the exact per-act binomial
            # variance at the *realized* edge count (robust to upstream matcher changes
            # that alter how many edges form per step). Otherwise use the passed tol.
            if tol is None:
                sigma = calc_tolerance(db_edge, eff_condom[name], condom_data, n_acts, n_edges=n_edges)["sigma"]
                row_tol = 3 * sigma
                row_warn = 2 * sigma
            else:
                row_tol = tol
                row_warn = warn_tol

            all_rows.append(
                {
                    "name": name,
                    "eff": eff_condom[name],
                    "condom": condom_data,
                    "acts": n_acts,
                    "n_edges": n_edges,
                    "dir": label,
                    "tol": row_tol,
                    "warn_tol": row_warn,
                    "mean_ref": beta_ref.mean(),
                    "mean_edge": beta_edge.mean(),
                    "mean_abs": abs_diff.mean(),
                    "max_abs": abs_diff.max(),
                    "mean_rel": rel_diff.mean(),
                    "max_rel": rel_diff.max(),
                }
            )

    return all_rows


def _classify_row(r):
    """Classify a row as pass/WARN/FAIL and issue a warning if near tolerance."""
    rd = r["rel_diff"]
    wt = r["warn_tol"]
    exceeds_tol = rd >= r["tol"]
    exceeds_warn = wt is not None and rd >= wt

    if exceeds_tol:
        r["pass"] = False
        r["status"] = "FAIL"
    elif exceeds_warn:
        r["pass"] = True
        r["status"] = "WARN"
        warnings.warn(
            f"Near tolerance: {r['name']} {r['dir']} rel_diff={rd:.2%} exceeds warn_tol={wt:.0%} (tol={r['tol']:.0%})",
            stacklevel=3,
        )
    else:
        r["pass"] = True
        r["status"] = "pass"
    return


def _print_table(title, all_rows):
    """Print a formatted comparison table with pass/warn/FAIL status column."""
    for r in all_rows:
        r["rel_diff"] = abs(r["mean_edge"] - r["mean_ref"]) / r["mean_ref"] if r["mean_ref"] > 0 else 0
        _classify_row(r)

    # Tolerance column: "warn / tol" when warn_tol is set, otherwise just "tol"
    tol_strs = [f"{r['warn_tol']:.0%} / {r['tol']:.0%}" if r["warn_tol"] is not None else f"{r['tol']:.0%}" for r in all_rows]

    df = sc.dataframe(
        name=[r["name"] for r in all_rows],
        eff=[f"{r['eff']:.2f}" for r in all_rows],
        cond=[f"{r['condom']:.2f}" for r in all_rows],
        acts=[r["acts"] for r in all_rows],
        n=[r["n_edges"] for r in all_rows],
        dir=[r["dir"] for r in all_rows],
        mean_ref=[f"{r['mean_ref']:.4f}" for r in all_rows],
        mean_edge=[f"{r['mean_edge']:.4f}" for r in all_rows],
        rel_diff=[f"{r['rel_diff']:.2%}" for r in all_rows],
        tol=tol_strs,
        status=[r["status"] for r in all_rows],
    )

    df.disp()


@sc.timer()
def test_net_beta_no_interventions():
    # Without condoms, EdgeStructuredSexual should exactly match StructuredSexual
    sc.heading("Testing net_beta equivalence vs StructuredSexual (no condoms)")
    eff = {"ng": 0, "ct": 0, "tv": 0}
    tol = 1e-6  # exact match expected
    rows = _run_pair(make_diseases(), eff, 0, 100, 100, tol)
    _print_table("No condoms (exact)", rows)
    for r in rows:
        assert r["rel_diff"] < r["tol"], f"Zero-condom net_beta should exactly match StructuredSexual ({r['name']} {r['dir']}: rel_diff={r['rel_diff']:.2%})"
    return rows


@sc.timer()
def test_net_beta_with_condoms():
    # The edge method discretizes condom use into per-act Bernoulli draws, so per-edge
    # values scatter around the reference's blended-rate answer. The means should agree.
    sc.heading("Testing net_beta equivalence vs StructuredSexual (with condoms)")
    rows = _run_pair(make_diseases(), eff_condom, 0.5, 100, 100, 0.01)
    _print_table("With condoms (ng/ct/tv)", rows)
    for r in rows:
        assert r["pass"], f"Mean net_beta should agree in expectation ({r['name']} {r['dir']}): rel_diff={r['rel_diff']:.2%} exceeds tol={r['tol']:.0%}"
    return rows


@sc.timer()
def test_net_beta_range():
    sc.heading("Testing net_beta equivalence vs StructuredSexual (parameter sweep)")

    # Tolerances are derived per-row from the exact per-act binomial variance at the
    # realized edge count (tol=None below). This keeps the test self-calibrating: it
    # tracks however many edges the current stisim matcher forms per step, instead of
    # hardcoding bounds tied to a particular network's steady-state edge count.
    scenarios = [
        # (beta_m2f, eff_condom, condom_data, n_acts, n_agents)
        (0.005, 1.0, 0.8, 10, 100),  # Worst case: all drivers aligned
        (0.019, 0.9, 0.5, 10, 100),  # Typical hard case, no extreme
        (0.0083, 0.8, 0.8, 100, 100),  # n_acts=100 concentrates the binomial
        (0.05, 1.0, 0.5, 10, 200),  # Population size averaging
        (0.1, 0.9, 0.8, 10, 200),  # High db saturation compresses
        (0.038, 0.46, 0.2, 10, 100),  # Low eff * cond → near-deterministic
    ]

    all_rows = []
    for beta_m2f, eff, condom_data, n_acts, n_agents in scenarios:
        ng = sti.Gonorrhea(beta_m2f=beta_m2f, rel_beta_f2m=0.5, init_prev=0.05)
        all_rows.extend(_run_pair([ng], {"ng": eff}, condom_data, n_acts, n_agents, tol=None))

    _print_table("Parameter sweep (ng)", all_rows)

    for r in all_rows:
        assert r["pass"], (
            f"Mean net_beta should agree in expectation ({r['name']} {r['dir']}, acts={r['acts']}, eff={r['eff']}, condom={r['condom']}): rel_diff={r['rel_diff']:.2%} exceeds tol={r['tol']:.0%}"
        )

    return all_rows


@sc.timer()
def test_condomuse_reduces_transmission():
    sc.heading("Testing condom use reduces transmission")
    ng = sti.Gonorrhea(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ng.pars.eff_condom = 1.0

    pars_no_condom = setup_sim_pars(rand_seed=7, diseases=[ng], networks=EdgeStructuredSexual())
    sim_no_condom = ss.Sim(**pars_no_condom)
    sim_no_condom.run(verbose=False)

    ng2 = sti.Gonorrhea(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ng2.pars.eff_condom = 1.0

    pars_full_condom = setup_sim_pars(rand_seed=7, diseases=[ng2], networks=EdgeStructuredSexual(), interventions=[StructuredCondomUse(condom_data=1.0)])
    sim_full_condom = ss.Sim(**pars_full_condom)
    sim_full_condom.run(verbose=False)

    inc_no_condom = sim_no_condom.summary.get("ng_incidence", 0)
    inc_full_condom = sim_full_condom.summary.get("ng_incidence", 0)

    assert inc_full_condom < inc_no_condom, f"Full condom coverage (eff=1.0) should reduce incidence: {inc_full_condom:.4f} vs {inc_no_condom:.4f}"
    return sim_full_condom


if __name__ == "__main__":
    T = sc.timer()

    test_net_beta_no_interventions()
    test_net_beta_with_condoms()
    test_net_beta_range()
    test_condomuse_reduces_transmission()

    T.toc()
