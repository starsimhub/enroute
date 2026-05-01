"""
Demo: StructuredCondomUse reducing STI transmission across multiple diseases.

Run:
    python enroute_examples/demo_condom_use.py
"""

import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import stisim as sti
from enroute import EdgeStructuredSexual, StructuredCondomUse

sc.options(interactive=False)


def make_sim(condom_data=None, n_agents=1000, seed=42, condom_cls=None):
    """Build a multi-disease sim with optional condom use."""
    if condom_cls is None:
        condom_cls = StructuredCondomUse

    # Diseases — elevated betas for a visible demo signal
    ng = sti.Gonorrhea(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ct = sti.Chlamydia(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ct.pars.eff_condom = 0.7  # default is 0; override for demo purposes

    # Network and intervention
    net = EdgeStructuredSexual()
    interventions = []
    if condom_data is not None:
        interventions.append(condom_cls(condom_data=condom_data))

    # Demographics
    pregnancy = sti.Pregnancy(fertility_rate=10)
    death = ss.Deaths(death_rate=10)

    return ss.Sim(
        n_agents=n_agents,
        rand_seed=seed,
        start=2000,
        dur=15,
        dt=1 / 12,
        diseases=[ng, ct],
        networks=net,
        demographics=[pregnancy, death],
        interventions=interventions,
    )


def run_scenarios(condom_values=None, n=1):
    """Run condom-use scenarios in parallel and compare.

    Args:
        condom_values (list): condom-use probabilities to test; None means [0.0, 0.5, 1.0]
        n            (int):  number of stochastic runs to average per scenario
    """
    if condom_values is None:
        condom_values = [0.0, 0.5, 1.0]

    # Build scenario dict with formatted labels
    scenarios = {}
    for val in condom_values:
        pct = val * 100
        pct = int(round(pct)) if pct == round(pct) else round(pct, 1)
        label = f"{pct}% condoms"
        scenarios[label] = val if val > 0 else None

    sc.heading("Running condom-use scenarios")

    # Build all sims across scenarios and seeds
    all_sims = []
    sim_map = {}  # label -> list of sim indices
    for label, condom_data in scenarios.items():
        idxs = []
        for seed in range(n):
            idxs.append(len(all_sims))
            all_sims.append(make_sim(condom_data=condom_data, seed=seed))
        sim_map[label] = idxs

    # Run everything in parallel
    msim = ss.parallel(*all_sims, verbose=0)

    # Collect per-scenario results
    results = {}
    sims = {}  # label -> representative sim (or reduced MultiSim)
    for label, idxs in sim_map.items():
        scenario_sims = [msim.sims[i] for i in idxs]
        if n == 1:
            sims[label] = scenario_sims[0]
            results[label] = scenario_sims[0].summary
        else:
            scenario_msim = ss.MultiSim(sims=scenario_sims)
            scenario_msim.median()
            sims[label] = scenario_msim.sims[0]
            results[label] = scenario_msim.sims[0].summary

    return sims, results


def print_summary(results):
    """Print disease outcomes across scenarios."""
    sc.heading("Results")

    keys = ["ng_incidence", "ng_prevalence", "ct_incidence", "ct_prevalence"]
    labels = list(results.keys())

    # Header
    col_w = 16
    header = f"{'Metric':<20}" + "".join(f"{lab:>{col_w}}" for lab in labels)
    print(header)
    print("-" * len(header))

    # Rows
    for key in keys:
        row = f"{key:<20}"
        for lab in labels:
            val = results[lab].get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    return


def plot_scenarios(sims):
    """Plot prevalence over time for each scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    cmap = plt.cm.coolwarm_r
    labels = list(sims.keys())
    colors = {lab: cmap(i / max(len(labels) - 1, 1)) for i, lab in enumerate(labels)}

    for disease_idx, disease_key in enumerate(["ng", "ct"]):
        ax = axes[disease_idx]
        for label, sim in sims.items():
            disease = sim.diseases[disease_key]
            tvec = sim.t.yearvec
            prev = disease.results.prevalence
            ax.plot(tvec, prev, label=label, color=colors[label], linewidth=2)
        ax.set_title(disease_key.upper() + " prevalence")
        ax.set_xlabel("Year")
        ax.set_ylabel("Prevalence")
        ax.legend()

    fig.suptitle("Impact of condom use on STI transmission", fontsize=14)
    fig.tight_layout()
    plt.savefig(sc.path(sc.thisdir(__file__)) / "demo_condom_use.png", dpi=150)

    return fig


if __name__ == "__main__":
    timer = sc.timer()

    sims, results = run_scenarios(condom_values=[0.0, 0.5, 1.0], n=3)
    print_summary(results)
    plot_scenarios(sims)

    timer.toc()
