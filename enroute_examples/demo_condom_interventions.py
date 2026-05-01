"""
Demo: StructuredCondomUse and BehavioralCondomUse — prevalence, condom dynamics, decay.

Section A: StructuredCondomUse prevalence across condom-use scenarios.
Section B: BehavioralCondomUse with Beta preferences, exponential decay, and diagnosis boost.
Section C: Side-by-side comparison of both intervention classes.

Run:
    python enroute_examples/demo_condom_interventions.py
"""

import sciris as sc
import starsim as ss
import stisim as sti
import pylab as pl

from enroute import EdgeStructuredSexual, StructuredCondomUse, CondomTracker
from enroute_examples.behavioral_condom_use import BehavioralCondomUse
import enroute.plotting as plt_lib

sc.options(interactive=False)

n_agents = 500
n_seeds = 3
do_plot = True


# %% Section A: StructuredCondomUse

STRUCTURED_SCENARIOS = {
    "No condoms": None,
    "Low (30%)": 0.3,
    "Medium (60%)": 0.6,
    "Full (100%)": 1.0,
}


def make_structured_sim(condom_data, seed):
    """Build a sim with StructuredCondomUse + CondomTracker.

    Args:
        condom_data (float/None): condom probability; None means no condom intervention
        seed        (int):        random seed
    """
    ng = sti.Gonorrhea(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)  # eff_condom=0.9 by default in stisim
    ct = sti.Chlamydia(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ct.pars.eff_condom = 0.7  # stisim default is 0; override for demo

    net = EdgeStructuredSexual()
    interventions = []
    analyzers = []
    if condom_data is not None:
        interventions.append(StructuredCondomUse(condom_data=condom_data))
        analyzers.append(CondomTracker())

    pregnancy = sti.Pregnancy(fertility_rate=10)
    death = ss.Deaths(death_rate=10)

    return ss.Sim(
        n_agents=n_agents, rand_seed=seed, start=2000, dur=10, dt=1 / 12,
        diseases=[ng, ct], networks=net,
        demographics=[pregnancy, death],
        interventions=interventions, analyzers=analyzers,
    )


def run_structured():
    """Run StructuredCondomUse scenarios, return grouped sims and trackers."""
    sc.heading("Section A: StructuredCondomUse scenarios")

    all_sims = []
    sim_map = {}  # label -> list of indices
    for label, cd in STRUCTURED_SCENARIOS.items():
        idxs = []
        for seed in range(n_seeds):
            idxs.append(len(all_sims))
            all_sims.append(make_structured_sim(cd, seed))
        sim_map[label] = idxs

    msim = ss.parallel(*all_sims, verbose=0)

    scenarios_sims = {}
    scenarios_trackers = {}
    for label, idxs in sim_map.items():
        sims = [msim.sims[i] for i in idxs]
        scenarios_sims[label] = sims
        trackers = []
        for s in sims:
            if hasattr(s.analyzers, "condomtracker"):
                trackers.append(s.analyzers.condomtracker)
        scenarios_trackers[label] = trackers

    return scenarios_sims, scenarios_trackers


# %% Section B: BehavioralCondomUse

BEHAVIORAL_BETA_PARAMS = {
    (0, "f"): (0.3, 0.7), (0, "m"): (0.3, 0.7),  # low-risk: mean ~0.30
    (1, "f"): (0.5, 0.5), (1, "m"): (0.5, 0.5),  # med-risk: mean  0.50
    (2, "f"): (0.7, 0.3), (2, "m"): (0.7, 0.3),  # high-risk: mean ~0.70
}

BEHAVIORAL_NEGOTIATION = {"stable": 0.3, "casual": 0.4, "onetime": 0.5, "sw": 0.7}

BEHAVIORAL_DECAY = {
    "stable": {"half_life": ss.dur(18, "months"), "floor": 0.1},
}

BEHAVIORAL_BOOST = {
    "disease": "ct",
    "prob": {"stable": 0.95, "casual": 0.90, "onetime": 0.85, "sw": 0.95},
    "duration": ss.dur(6, "months"),
}


def make_behavioral_sim(seed):
    """Build a sim with BehavioralCondomUse + CondomTracker.

    Args:
        seed (int): random seed
    """
    ct = sti.Chlamydia(beta_m2f=0.15, rel_beta_f2m=0.5, init_prev=0.05)
    ct.pars.eff_condom = 0.7

    net = EdgeStructuredSexual()
    condoms = BehavioralCondomUse(
        beta_params=BEHAVIORAL_BETA_PARAMS,
        negotiation_weights=BEHAVIORAL_NEGOTIATION,
        decay_pars=BEHAVIORAL_DECAY,
        diagnosis_boost=BEHAVIORAL_BOOST,
    )
    tracker = CondomTracker()

    pregnancy = sti.Pregnancy(fertility_rate=10)
    death = ss.Deaths(death_rate=10)

    return ss.Sim(
        n_agents=n_agents, rand_seed=seed, start=2000, dur=10, dt=1 / 12,
        diseases=[ct], networks=net,
        demographics=[pregnancy, death],
        interventions=[condoms], analyzers=[tracker],
    )


def run_behavioral():
    """Run BehavioralCondomUse scenario, return sims and trackers."""
    sc.heading("Section B: BehavioralCondomUse with decay + boost")

    all_sims = [make_behavioral_sim(seed) for seed in range(n_seeds)]
    msim = ss.parallel(*all_sims, verbose=0)
    sims = msim.sims
    trackers = [s.analyzers.condomtracker for s in sims if hasattr(s.analyzers, "condomtracker")]
    return sims, trackers


# %% Section C: comparison

def run_comparison():
    """Run three arms: no condoms, StructuredCondomUse, BehavioralCondomUse."""
    sc.heading("Section C: StructuredCondomUse vs BehavioralCondomUse")

    all_sims = []
    sim_map = {}

    for label, make_fn in [
        ("No condoms", lambda s: make_structured_sim(None, s)),
        ("Structured (60%)", lambda s: make_structured_sim(0.6, s)),
        ("Behavioral (prefs)", make_behavioral_sim),
    ]:
        idxs = []
        for seed in range(n_seeds):
            idxs.append(len(all_sims))
            all_sims.append(make_fn(seed))
        sim_map[label] = idxs

    msim = ss.parallel(*all_sims, verbose=0)

    scenarios = {}
    for label, idxs in sim_map.items():
        scenarios[label] = [msim.sims[i] for i in idxs]
    return scenarios


# %% Main

if __name__ == "__main__":
    T = sc.timer()

    sims_A, trackers_A = run_structured()
    sims_B, trackers_B = run_behavioral()
    sims_C = run_comparison()

    if do_plot:
        sc.heading("Plotting Section A: StructuredCondomUse")

        fig = plt_lib.plot_prevalence_comparison(sims_A, disease_key="ct", title="CT prevalence — StructuredCondomUse scenarios")
        plt_lib.savefig(fig, "demo_structured_prevalence_ct")

        fig = plt_lib.plot_prevalence_comparison(sims_A, disease_key="ng", title="NG prevalence — StructuredCondomUse scenarios")
        plt_lib.savefig(fig, "demo_structured_prevalence_ng")

        tracked = trackers_A.get("Medium (60%)", [])
        if tracked:
            fig = plt_lib.plot_condom_prob_by_type(tracked, title="Condom prob by type — 60% scenario")
            plt_lib.savefig(fig, "demo_structured_condom_by_type")

            fig = plt_lib.plot_acts_protection(tracked, title="Act protection — 60% scenario")
            plt_lib.savefig(fig, "demo_structured_acts_protection")

        fig = plt_lib.plot_structured_summary(sims_A, trackers_A, disease_key="ct")
        plt_lib.savefig(fig, "demo_structured_summary")

        sc.heading("Plotting Section B: BehavioralCondomUse")

        fig = plt_lib.plot_beta_preference_dist(BEHAVIORAL_BETA_PARAMS)
        plt_lib.savefig(fig, "demo_behavioral_beta_prefs")

        fig = plt_lib.plot_condom_prob_by_type(trackers_B, title="Condom prob by type — BehavioralCondomUse")
        plt_lib.savefig(fig, "demo_behavioral_condom_by_type")

        fig = plt_lib.plot_decay_curve(trackers_B, etype="stable")
        if fig is not None:
            plt_lib.savefig(fig, "demo_behavioral_decay_stable")

        fig = plt_lib.plot_acts_protection(trackers_B, title="Act protection — BehavioralCondomUse")
        plt_lib.savefig(fig, "demo_behavioral_acts_protection")

        fig = plt_lib.plot_network_graph(sims_B[0], title="Network — BehavioralCondomUse")
        if fig is not None:
            plt_lib.savefig(fig, "demo_behavioral_network")

        fig = plt_lib.plot_behavioral_summary(trackers_B, BEHAVIORAL_BETA_PARAMS)
        plt_lib.savefig(fig, "demo_behavioral_summary")

        sc.heading("Plotting Section C: Comparison")

        fig = plt_lib.plot_prevalence_comparison(sims_C, disease_key="ct", title="CT prevalence — Structured vs Behavioral")
        plt_lib.savefig(fig, "demo_comparison_prevalence")

        pl.show()

    T.toc()
