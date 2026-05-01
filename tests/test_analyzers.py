"""Tests for CondomTracker analyzer: timestep counts, data integrity, and parallel compatibility."""

import numpy as np
import sciris as sc

import starsim as ss
import stisim as sti
from enroute import EdgeStructuredSexual, StructuredCondomUse, CondomTracker
from enroute_examples.behavioral_condom_use import BehavioralCondomUse

n_agents = 200
do_plot = False
sc.options(interactive=False)


def make_sim(condom_data=0.5, condom_cls=None, condom_kwargs=None, seed=42, dur=5):
    """Build a sim with EdgeStructuredSexual + condom intervention + CondomTracker.

    Args:
        condom_data  (float/None): condom probability
        condom_cls   (class):      intervention class; None means StructuredCondomUse
        condom_kwargs (dict):      extra kwargs for the intervention constructor
        seed         (int):        random seed
        dur          (int):        sim duration in years
    """
    if condom_cls is None:
        condom_cls = StructuredCondomUse
    if condom_kwargs is None:
        condom_kwargs = {}

    net = EdgeStructuredSexual()
    ct = sti.Chlamydia()
    ct.pars.eff_condom = 0.7
    condoms = condom_cls(condom_data=condom_data, **condom_kwargs)
    tracker = CondomTracker()

    return ss.Sim(
        n_agents=n_agents, start=2000, dur=dur, dt=1 / 12, rand_seed=seed,
        diseases=[ct], networks=net,
        interventions=[condoms], analyzers=[tracker],
    )


@sc.timer()
def test_tracker_timestep_count():
    """Tracker records one entry per simulation timestep."""
    sim = make_sim()
    sim.run()
    tracker = sim.analyzers.condomtracker
    expected = len(sim.t.yearvec)
    actual = len(tracker.data.ti)
    assert actual == expected, f"Expected {expected} timestep entries, got {actual}"
    return tracker


@sc.timer()
def test_tracker_mean_matches_known_prob():
    """With condom_data=0.5, mean condom probability should be ~0.5."""
    sim = make_sim(condom_data=0.5)
    sim.run()
    tracker = sim.analyzers.condomtracker
    overall_mean = np.mean(tracker.data.mean_condom)
    assert np.isclose(overall_mean, 0.5, rtol=0.05), f"Expected mean condom ~0.5, got {overall_mean:.4f}"
    return tracker


@sc.timer()
def test_uses_bounded_by_acts():
    """Protected acts never exceed total acts at any timestep."""
    sim = make_sim(condom_data=0.8)
    sim.run()
    tracker = sim.analyzers.condomtracker
    uses = np.array(tracker.data.total_uses)
    acts = np.array(tracker.data.total_acts)
    assert np.all(uses <= acts), "total_uses exceeds total_acts on at least one timestep"
    return tracker


@sc.timer()
def test_by_type_sums_to_total():
    """Sum of per-type acts equals total_acts at every timestep."""
    sim = make_sim(condom_data=0.5)
    sim.run()
    tracker = sim.analyzers.condomtracker
    total = np.array(tracker.data.total_acts)
    by_type_sum = sum(np.array(tracker.data.acts_by_type[et]) for et in tracker.data.acts_by_type)
    assert np.allclose(total, by_type_sum), "Sum of per-type acts should equal total_acts at every timestep"
    return tracker


@sc.timer()
def test_rel_ages_present_with_behavioral():
    """BehavioralCondomUse with decay produces non-empty relationship-age data."""
    sim = make_sim(
        condom_data=None,
        condom_cls=BehavioralCondomUse,
        condom_kwargs=dict(decay_pars={"stable": {"half_life": ss.dur(18, "months"), "floor": 0.1}}),
    )
    sim.run()
    tracker = sim.analyzers.condomtracker
    final_ages = tracker.data.rel_ages_by_type.stable[-1]
    assert len(final_ages) > 0, "Expected non-empty relationship-age data for stable edges with decay"
    return tracker


@sc.timer()
def test_rel_ages_empty_without_start_ti():
    """Plain StructuredCondomUse (no start_ti) produces empty relationship-age data."""
    sim = make_sim(condom_data=0.5, condom_cls=StructuredCondomUse)
    sim.run()
    tracker = sim.analyzers.condomtracker
    final_ages = tracker.data.rel_ages_by_type.stable[-1]
    assert len(final_ages) == 0, "Expected empty relationship-age data without start_ti"
    return tracker


@sc.timer()
def test_tracker_survives_parallel():
    """CondomTracker works correctly across seeds run via ss.parallel."""
    sim0 = make_sim(condom_data=0.5, seed=0)
    sim1 = make_sim(condom_data=0.5, seed=1)
    msim = ss.parallel(sim0, sim1, verbose=0)

    tr0 = msim.sims[0].analyzers.condomtracker
    tr1 = msim.sims[1].analyzers.condomtracker

    assert len(tr0.data.ti) == len(tr1.data.ti), "Both seeds should have the same number of timesteps"
    assert len(tr0.data.ti) > 0, "Tracker data should be non-empty after parallel run"

    # different seeds should produce different stochastic realizations in act counts
    u0 = np.array(tr0.data.total_uses)
    u1 = np.array(tr1.data.total_uses)
    assert u0.sum() > 0, "Seed 0 should have some protected acts"
    assert u1.sum() > 0, "Seed 1 should have some protected acts"
    assert not np.array_equal(u0, u1), "Different seeds should produce different protected-act counts"
    return msim


if __name__ == "__main__":
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    test_tracker_timestep_count()
    test_tracker_mean_matches_known_prob()
    test_uses_bounded_by_acts()
    test_by_type_sums_to_total()
    test_rel_ages_present_with_behavioral()
    test_rel_ages_empty_without_start_ti()
    test_tracker_survives_parallel()

    T.toc()
