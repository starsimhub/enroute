"""
Tests for DoxyPEPEnrollment and DoxyPEP interventions.

Covers: enrollment state, eligibility, dose dispensing, refills, edge-level
usage computation, uptake dict-by-edge-type, relative risk lookup, and
condom stacking.
"""

import numpy as np
import sciris as sc

import starsim as ss
import stisim as sti
from enroute import EdgeStructuredSexual, StructuredCondomUse
from enroute.doxypep import DoxyPEPEnrollment, DoxyPEP

n_agents = 500
do_plot = False
sc.options(interactive=False)


def make_sim(
    enroll_prob=0.1,
    prescribed_doses=30,
    refill_threshold=5,
    refill_prob=0.8,
    uptake=0.6,
    rel_risk=None,
    eligibility=None,
    condom_data=None,
    diseases=True,  # passed to DoxyPEP
    seed=1,
    dur=3,
    dt=1 / 12,
    n=None,
):
    """Create a minimal sim with EdgeStructuredSexual + Chlamydia + DoxyPEP stack."""
    if n is None:
        n = n_agents

    enrollment = DoxyPEPEnrollment(
        name="dpep_enrollment",
        eligibility=eligibility,
        enroll_prob=enroll_prob,
        prescribed_doses=prescribed_doses,
        refill_threshold=refill_threshold,
        refill_prob=refill_prob,
    )
    doxypep = DoxyPEP(
        name="doxypep",
        uptake=uptake,
        rel_risk=rel_risk,
        diseases=diseases,
    )

    # Condom goes first so DoxyPEP can check for prior edge-level interventions
    intvs = []
    if condom_data is not None:
        intvs.append(StructuredCondomUse(condom_data=condom_data))
    intvs.extend([enrollment, doxypep])

    return ss.Sim(
        rand_seed=seed,
        dt=dt,
        start=2000,
        dur=dur,
        n_agents=n,
        diseases=sti.Chlamydia(),
        networks=EdgeStructuredSexual(),
        interventions=intvs,
        verbose=0,
    )


@sc.timer()
def test_enrollment_initializes():
    """All 6 states exist and no agents enrolled before the first step."""
    sc.heading("Testing enrollment initializes")
    sim = make_sim()
    sim.init()

    enr = sim.interventions.dpep_enrollment
    for state in ("enrolled", "eligible", "ever_enrolled", "doses", "ti_enrolled", "ti_unenrolled"):
        assert hasattr(enr, state), f"Expected state '{state}' on DoxyPEPEnrollment"

    n_enrolled = np.count_nonzero(enr.enrolled.raw)
    assert n_enrolled == 0, f"Expected 0 agents enrolled before first step, got {n_enrolled}"
    return sim


@sc.timer()
def test_no_enrollment_at_zero_prob():
    """With enroll_prob=0, no one should ever enroll."""
    sc.heading("Testing zero enroll_prob → no enrollments")
    sim = make_sim(enroll_prob=0, dur=5)
    sim.run()

    enr = sim.interventions.dpep_enrollment
    n_enrolled = np.count_nonzero(enr.enrolled.raw)
    assert n_enrolled == 0, f"Expected 0 enrolled with enroll_prob=0, got {n_enrolled}"

    all_zero = np.all(sim.results.dpep_enrollment.n_enrolled.values == 0)
    assert all_zero, "Expected n_enrolled results to be all-zero with enroll_prob=0"
    return sim


@sc.timer()
def test_enrollment_increases_with_prob():
    """Scientific: higher enroll_prob → more enrolled agents (same seed)."""
    sc.heading("Testing higher enroll_prob → more enrolled")
    sim_lo = make_sim(enroll_prob=0.05, dur=5, seed=42)
    sim_hi = make_sim(enroll_prob=0.95, dur=5, seed=42)
    sim_lo.run()
    sim_hi.run()

    n_lo = sim_lo.results.dpep_enrollment.n_enrolled.values[-1]
    n_hi = sim_hi.results.dpep_enrollment.n_enrolled.values[-1]
    assert n_hi > n_lo, f"Expected more enrolled with high prob ({n_hi}) than low ({n_lo})"

    # High prob should enroll >50% of population
    min_frac = 0.5  # majority threshold
    frac_hi = n_hi / n_agents
    assert frac_hi > min_frac, f"Expected >50% enrolled at high enroll_prob, got {frac_hi:.1%}"
    return sim_hi


@sc.timer()
def test_enrolled_agents_get_doses():
    """Enrolling with prob=1 and no consumption → all enrolled agents have >= prescribed_doses."""
    sc.heading("Testing enrolled agents get doses")
    prescribed = 30
    sim = make_sim(enroll_prob=1.0, prescribed_doses=prescribed, uptake=0, refill_prob=0, dur=1)
    sim.run()

    enr = sim.interventions.dpep_enrollment
    enrolled_mask = enr.enrolled.raw
    doses = enr.doses.raw[enrolled_mask]

    # Everyone who ever enrolled should have at least prescribed_doses
    assert len(doses) > 0, "Expected some agents to be enrolled"
    assert np.all(doses >= prescribed), f"Expected all enrolled agents to have >= {prescribed} doses, min={doses.min()}"
    return sim


@sc.timer()
def test_unenrollment_on_lost_eligibility():
    """Agents unenrolled after losing eligibility keep their remaining doses."""
    sc.heading("Testing unenrollment on lost eligibility")

    # Eligibility: everyone on first call, nobody afterwards
    call_count = [0]

    def toggling_eligibility(sim):
        call_count[0] += 1
        if call_count[0] <= 1:
            return sim.people.alive.uids  # everyone eligible on first step
        return ss.uids([])  # nobody eligible afterwards

    sim = make_sim(
        enroll_prob=1.0,
        prescribed_doses=30,
        uptake=0,  # no consumption → doses stay positive
        refill_prob=0,
        eligibility=toggling_eligibility,
        dur=2,
    )
    sim.run()

    enr = sim.interventions.dpep_enrollment
    n_enrolled_now = np.count_nonzero(enr.enrolled.raw)
    n_ever = np.count_nonzero(enr.ever_enrolled.raw)
    assert n_ever > 0, "Expected some agents to have ever been enrolled"
    assert n_enrolled_now == 0, f"Expected 0 currently enrolled after eligibility removed, got {n_enrolled_now}"

    # Unenrolled agents should retain their doses (no consumption)
    doses_unenrolled = enr.doses.raw[enr.ever_enrolled.raw & ~enr.enrolled.raw]
    assert np.all(doses_unenrolled > 0), "Expected unenrolled agents to keep their doses"

    total_unenrollments = sim.results.dpep_enrollment.new_unenrollments.values.sum()
    assert total_unenrollments > 0, "Expected new_unenrollments result > 0"
    return sim


@sc.timer()
def test_refill_restores_doses():
    """Refill triggers when doses < refill_threshold and restores doses above prescribed_doses."""
    sc.heading("Testing refill restores doses")
    # refill_threshold > prescribed_doses → triggers immediately on first step
    prescribed = 2
    sim = make_sim(
        enroll_prob=1.0,
        prescribed_doses=prescribed,
        refill_threshold=5,  # above prescribed → refills immediately
        refill_prob=1.0,
        uptake=0,  # no consumption so we can track dispensing clearly
        dur=1,
    )
    sim.run()

    enr = sim.interventions.dpep_enrollment
    doses = enr.doses.raw[enr.enrolled.raw]
    assert len(doses) > 0, "Expected some agents enrolled"

    # After enroll (+2) and immediate refill (+2) → doses should be > prescribed (2)
    assert np.any(doses > prescribed), f"Expected some agents to have doses > {prescribed} after refill, max={doses.max()}"
    return sim


@sc.timer()
def test_doxypep_initializes():
    """DoxyPEP auto-discovers enrollment and registers its edge column on init."""
    sc.heading("Testing DoxyPEP initializes")
    sim = make_sim()
    sim.init()

    dpep = sim.interventions.doxypep
    assert isinstance(dpep._enrollment, DoxyPEPEnrollment), "Expected _enrollment to be DoxyPEPEnrollment"

    net = sim.networks.structuredsexual
    assert "doxypep_uses" in net.meta, "Expected 'doxypep_uses' in network meta after init"
    return sim


@sc.timer()
def test_no_uses_when_nobody_enrolled():
    """With enroll_prob=0, nobody has doses, so all doxypep_uses should be zero."""
    sc.heading("Testing no uses when nobody enrolled")
    sim = make_sim(enroll_prob=0, uptake=1.0, dur=1)
    sim.run()

    net = sim.networks.structuredsexual
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    uses = net.edges.doxypep_uses
    assert np.all(uses == 0), f"Expected all doxypep_uses==0 with enroll_prob=0, sum={uses.sum()}"
    return sim


@sc.timer()
def test_uses_positive_when_enrolled():
    """With high enroll_prob, many doses, and high uptake, doxypep_uses should be > 0."""
    sc.heading("Testing uses positive when enrolled")
    sim = make_sim(enroll_prob=0.99, prescribed_doses=1000, uptake=0.9, dur=3)
    sim.run()

    net = sim.networks.structuredsexual
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    uses = net.edges.doxypep_uses
    total = int(uses.sum())
    assert total > 0, f"Expected positive doxypep_uses with high enrollment and uptake, got {total}"
    return sim


@sc.timer()
def test_uptake_zero_means_no_uses():
    """uptake=0 should produce zero doxypep_uses regardless of enrollment."""
    sc.heading("Testing uptake=0 → no uses")
    sim = make_sim(enroll_prob=1.0, prescribed_doses=1000, uptake=0.0, dur=1)
    sim.run()

    net = sim.networks.structuredsexual
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    uses = net.edges.doxypep_uses
    assert np.all(uses == 0), f"Expected doxypep_uses==0 with uptake=0, sum={uses.sum()}"
    return sim


@sc.timer()
def test_uptake_one_means_max_uses():
    """uptake=1 and no condoms: on edges where both p1 and p2 are enrolled, uses must equal acts."""
    sc.heading("Testing uptake=1 → uses == acts on fully-enrolled edges")
    sim = make_sim(enroll_prob=1.0, prescribed_doses=10_000, uptake=1.0, dur=1)
    sim.init()
    sim.run_one_step()

    net = sim.networks.structuredsexual
    enr = sim.interventions.dpep_enrollment

    acts = net.edges.acts
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    uses = net.edges.doxypep_uses

    p1_arr = np.asarray(net.edges.p1)
    p2_arr = np.asarray(net.edges.p2)
    enrolled = enr.enrolled.raw

    # Find edges where both endpoints are enrolled and there are acts
    both_enrolled = enrolled[p1_arr] & enrolled[p2_arr]
    has_acts = acts > 0
    mask = both_enrolled & has_acts

    assert mask.sum() > 0, "Expected at least one edge with both endpoints enrolled and acts > 0"
    # For those edges, uses should equal acts (all protected with uptake=1)
    assert np.all(uses[mask] == acts[mask]), f"Expected doxypep_uses == acts on fully-enrolled edges with uptake=1; mismatches: {(uses[mask] != acts[mask]).sum()}"
    return sim


@sc.timer()
def test_doses_decrease_over_time():
    """With refill_prob=0 and positive uptake, total doses remaining should fall over time."""
    sc.heading("Testing doses decrease over time")
    sim = make_sim(enroll_prob=1.0, prescribed_doses=500, refill_prob=0, uptake=0.8, dur=5)
    sim.init()

    enr = sim.interventions.dpep_enrollment
    doses_per_step = []
    for _ in range(sim.t.npts):
        sim.run_one_step()
        doses_per_step.append(int(enr.doses.raw.sum()))

    peak = max(doses_per_step)
    final = doses_per_step[-1]
    assert peak > 0, "Expected peak total doses to be positive"
    assert final < peak, f"Expected final total doses ({final}) < peak total doses ({peak})"
    return doses_per_step


@sc.timer()
def test_zero_doses_stops_uses():
    """Once doses run out (prescribed=1, refill=0, uptake=1), uses should eventually hit 0."""
    sc.heading("Testing zero doses stops uses")
    sim = make_sim(enroll_prob=1.0, prescribed_doses=1, refill_prob=0, uptake=1.0, dur=10)
    sim.init()

    uses_per_step = []
    n_steps = sim.t.npts
    for _ in range(n_steps):
        sim.run_one_step()
        net = sim.networks.structuredsexual
        assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
        uses = net.edges.doxypep_uses
        uses_per_step.append(int(uses.sum()))

    nonzero_steps = [u for u in uses_per_step if u > 0]
    last_steps = uses_per_step[-10:]

    assert len(nonzero_steps) > 0, "Expected at least one step with positive uses"
    assert any(u == 0 for u in last_steps), f"Expected zero uses in late steps after doses exhausted; last steps: {last_steps}"
    return uses_per_step


@sc.timer()
def test_relative_risk_known_diseases():
    """relative_risk() returns the correct values from the default dict (exact lookup)."""
    sc.heading("Testing relative_risk for known diseases")
    sim = make_sim(dur=1)
    sim.init()
    sim.run_one_step()

    dpep = sim.interventions.doxypep
    net = sim.networks.structuredsexual

    # Chlamydia is present in sim — use actual disease object for 'ct'
    ct_disease = sim.diseases.ct

    # Mock objects for diseases not in this sim
    class MockDisease:
        def __init__(self, name):
            self.name = name

    ng_disease = MockDisease("ng")
    syph_disease = MockDisease("syph")

    rr_ct = dpep.relative_risk(net, None, ct_disease)
    rr_ng = dpep.relative_risk(net, None, ng_disease)
    rr_syph = dpep.relative_risk(net, None, syph_disease)

    # Exact lookups — rtol=1e-9
    assert np.isclose(rr_ct, 0.12, rtol=1e-9), f"Expected ct rr=0.12, got {rr_ct}"
    assert np.isclose(rr_ng, 0.45, rtol=1e-9), f"Expected ng rr=0.45, got {rr_ng}"
    assert np.isclose(rr_syph, 0.13, rtol=1e-9), f"Expected syph rr=0.13, got {rr_syph}"
    return dpep


@sc.timer()
def test_relative_risk_unknown_disease():
    """relative_risk() returns exactly 1.0 for an unknown disease name."""
    sc.heading("Testing relative_risk for unknown disease → 1.0")
    sim = make_sim(dur=1)
    sim.init()
    sim.run_one_step()

    dpep = sim.interventions.doxypep
    net = sim.networks.structuredsexual

    class MockDisease:
        name = "unknown_xyz"

    rr = dpep.relative_risk(net, None, MockDisease())
    assert np.isclose(rr, 1.0, rtol=1e-9), f"Expected rr=1.0 for unknown disease, got {rr}"
    return rr


@sc.timer()
def test_uptake_dict_by_edge_type():
    """Dict uptake: stable=0 should yield zero uses on stable edges; casual=1 → positive uses on casual edges."""
    sc.heading("Testing uptake dict by edge type")
    uptake = {"stable": 0.0, "casual": 1.0, "onetime": 0.0, "sw": 0.0}
    sim = make_sim(enroll_prob=1.0, prescribed_doses=10_000, uptake=uptake, dur=1)
    sim.init()
    sim.run_one_step()

    net = sim.networks.structuredsexual
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    uses = net.edges.doxypep_uses
    acts = net.edges.acts
    edge_type = net.edges.edge_type

    stable_val = net.edge_types["stable"]
    casual_val = net.edge_types["casual"]

    stable_uses = uses[edge_type == stable_val]
    assert np.all(stable_uses == 0), f"Expected doxypep_uses==0 on all stable edges with uptake=0, sum={stable_uses.sum()}"

    casual_mask = (edge_type == casual_val) & (acts > 0)
    assert casual_mask.sum() > 0, "Expected at least one casual edge with acts > 0 after one step"
    casual_uses = uses[casual_mask]
    assert casual_uses.sum() > 0, "Expected positive doxypep_uses on some casual edges with uptake=1 and acts>0"
    return sim


@sc.timer()
def test_condom_stacking():
    """DoxyPEP stacks on top of condom coverage: doxypep_uses + condom_uses <= acts."""
    sc.heading("Testing condom + doxypep stacking")
    sim = make_sim(
        condom_data=0.5,
        enroll_prob=0.8,
        uptake=0.9,
        prescribed_doses=10_000,
        dur=1,
    )
    sim.init()
    sim.run_one_step()

    net = sim.networks.structuredsexual
    acts = net.edges.acts
    assert "doxypep_uses" in net.meta, "doxypep_uses column must be registered"
    doxy_uses = net.edges.doxypep_uses
    assert "condomuse_uses" in net.meta, "condomuse_uses column must be registered"
    cond_uses = net.edges.condomuse_uses

    assert np.all(doxy_uses + cond_uses <= acts), f"Expected doxypep_uses + condom_uses <= acts on every edge; violations: {((doxy_uses + cond_uses) > acts).sum()}"
    assert doxy_uses.sum() > 0, "Expected positive doxypep uses"
    assert cond_uses.sum() > 0, "Expected positive condom uses"
    return sim


@sc.timer()
def test_female_only_eligibility():
    """With only one sex enrolled, edges still get doxypep protection via the enrolled partner."""
    sim = make_sim(
        enroll_prob=1.0,
        prescribed_doses=10_000,
        uptake=1.0,
        eligibility=lambda sim: sim.people.female.uids,
        refill_prob=0,
        dur=3,
    )
    sim.run()

    enr = sim.interventions.dpep_enrollment
    net = sim.networks.structuredsexual

    # Only females should be enrolled
    enrolled_uids = enr.enrolled.uids
    assert len(enrolled_uids) > 0, "Expected some enrolled agents"
    assert sim.people.female[enrolled_uids].all(), "Only females should be enrolled"

    # Non-eligible agents must never have received doses
    non_eligible = (~sim.people.female).uids
    assert np.all(enr.doses[non_eligible] == 0), "Non-eligible agents should have zero doses"

    # Key regression: enrolled agents must have consumed doses.
    # With refill_prob=0, each got exactly prescribed_doses at enrollment.
    # Any consumption proves edge-level uses were positive.
    total_initial = len(enrolled_uids) * 10_000
    total_remaining = float(enr.doses[enrolled_uids].sum())
    assert total_remaining < total_initial, (
        f"Expected dose consumption from enrolled agents; "
        f"initial={total_initial}, remaining={total_remaining}. "
        f"Zero consumption indicates the bilateral-enrollment bug."
    )
    return sim


if __name__ == "__main__":
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    test_enrollment_initializes()
    test_no_enrollment_at_zero_prob()
    test_enrollment_increases_with_prob()
    test_enrolled_agents_get_doses()
    test_unenrollment_on_lost_eligibility()
    test_refill_restores_doses()
    test_doxypep_initializes()
    test_no_uses_when_nobody_enrolled()
    test_uses_positive_when_enrolled()
    test_uptake_zero_means_no_uses()
    test_uptake_one_means_max_uses()
    test_doses_decrease_over_time()
    test_zero_doses_stops_uses()
    test_relative_risk_known_diseases()
    test_relative_risk_unknown_disease()
    test_uptake_dict_by_edge_type()
    test_condom_stacking()
    test_female_only_eligibility()

    T.toc()
