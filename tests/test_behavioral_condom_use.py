"""Tests for BehavioralCondomUse: preference, negotiation, exponential decay, and boosts."""

import numpy as np
import sciris as sc

import starsim as ss
import stisim as sti
import matplotlib.pyplot as plt
from enroute import EdgeStructuredSexual, StructuredCondomUse
from enroute_examples.behavioral_condom_use import BehavioralCondomUse

n_agents = 500
do_plot = False
sc.options(interactive=False)


def make_sim(condom_data=None, beta_params=None, negotiation_weights=None, decay_pars=None, diagnosis_boost=None, treatment_boost=None, diseases=None, seed=42, dur=5, **kwargs):
    """Build a sim with EdgeStructuredSexual + BehavioralCondomUse."""
    if diseases is None:
        diseases = [sti.Chlamydia()]
    net = EdgeStructuredSexual()
    condoms = BehavioralCondomUse(
        condom_data=condom_data,
        beta_params=beta_params,
        negotiation_weights=negotiation_weights,
        decay_pars=decay_pars,
        diagnosis_boost=diagnosis_boost,
        treatment_boost=treatment_boost,
    )
    extra_interventions = kwargs.pop("extra_interventions", [])
    interventions = [condoms] + extra_interventions
    pars = dict(
        n_agents=n_agents,
        start=2000,
        dur=dur,
        dt=1 / 12,
        rand_seed=seed,
        diseases=diseases,
        networks=net,
        interventions=interventions,
    )
    pars.update(kwargs)
    return ss.Sim(**pars)


def diagnose_agent(sim, agent_uid, disease_name, ti=0):
    """Manually mark an agent as diagnosed at a given timestep."""
    disease = sim.diseases[disease_name]
    disease.diagnosed = np.zeros(len(sim.people), dtype=bool)
    disease.ti_diagnosed = np.full(len(sim.people), np.nan)
    disease.diagnosed[agent_uid] = True
    disease.ti_diagnosed[agent_uid] = ti
    return


def treat_agent(sim, agent_uid, disease_name, ti=0):
    """Manually mark an agent as treated at a given timestep."""
    disease = sim.diseases[disease_name]
    if not hasattr(disease, "treated"):
        disease.treated = np.zeros(len(sim.people), dtype=bool)
        disease.ti_treated = np.full(len(sim.people), np.nan)
    disease.diagnosed = np.zeros(len(sim.people), dtype=bool)
    disease.ti_diagnosed = np.full(len(sim.people), np.nan)
    disease.treated[agent_uid] = True
    disease.ti_treated[agent_uid] = ti
    return


def get_agent_edges(net, uid):
    """Return a boolean mask of edges touching a given agent."""
    return (np.asarray(net.edges.p1) == uid) | (np.asarray(net.edges.p2) == uid)


def get_intv(sim):
    """Retrieve the BehavioralCondomUse intervention from a sim."""
    return sim.interventions["condomuse"]


@sc.timer()
def test_prefs_drawn_from_beta():
    """Per-agent preferences should have mean ≈ a/(a+b) and be bounded in [0, 1]."""
    sc.heading("test_prefs_drawn_from_beta")

    beta_params = {
        (0, "f"): (0.3, 0.7),
        (0, "m"): (0.3, 0.7),  # mean ~0.30
        (1, "f"): (0.5, 0.5),
        (1, "m"): (0.5, 0.5),  # mean 0.50
        (2, "f"): (0.7, 0.3),
        (2, "m"): (0.7, 0.3),  # mean ~0.70
    }

    sim = make_sim(beta_params=beta_params, dur=1, seed=7)
    sim.init()
    sim.run_one_step()

    intv = get_intv(sim)
    net = sim.networks.structuredsexual
    pref = intv._condom_pref

    assert pref is not None, "Expected _condom_pref to be allocated after init_post"
    assert np.all(pref >= 0) and np.all(pref <= 1), "All preferences should be in [0, 1]"

    rtol = 0.35  # bimodal U-shaped beta has high variance; ~80 agents per subgroup
    for (rg, sex_label), (a, b) in beta_params.items():
        sex_mask = sim.people.female if sex_label == "f" else sim.people.male
        in_group = (np.asarray(net.risk_group) == rg) & np.asarray(sex_mask)
        group_prefs = pref[in_group]
        min_group_size = 10
        if len(group_prefs) < min_group_size:
            continue
        expected_mean = a / (a + b)
        actual_mean = np.mean(group_prefs)
        assert np.isclose(actual_mean, expected_mean, rtol=rtol), f"Beta({a},{b}) mean for rg={rg},sex={sex_label}: expected ~{expected_mean:.2f}, got {actual_mean:.2f}"

    return sim


@sc.timer()
def test_negotiation_weights():
    """With extreme prefs (male=0, female=1), edge probs should reflect negotiation weights."""
    sc.heading("test_negotiation_weights")

    sim = make_sim(dur=1, seed=42)
    sim.init()

    intv = get_intv(sim)
    net = sim.networks.structuredsexual
    ppl = sim.people

    intv._condom_pref[np.asarray(ppl.female)] = 1.0
    intv._condom_pref[np.asarray(ppl.male)] = 0.0

    sim.run_one_step()

    icp = np.asarray(net.edges["initial_condom_prob"])
    etypes = np.asarray(net.edges.edge_type)
    weights = intv._negotiation_weights

    for etype_name, etype_code in net.edge_types.items():
        mask = etypes == etype_code
        if not mask.any():
            continue
        expected = weights.get(etype_name, 0.5)
        vals = icp[mask]
        assert np.allclose(vals, expected, atol=1e-10), f"{etype_name} edges: expected initial_condom_prob={expected}, got {vals[:5]}"

    return sim


@sc.timer()
def test_initial_prob_persists():
    """initial_condom_prob should not change for existing edges across steps."""
    sc.heading("test_initial_prob_persists")

    sim = make_sim(dur=2, seed=10)
    sim.init()

    for _ in range(6):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    icp_before = np.asarray(net.edges["initial_condom_prob"]).copy()
    edge_ids_before = (np.asarray(net.edges.p1).copy(), np.asarray(net.edges.p2).copy())

    sim.run_one_step()

    icp_after = np.asarray(net.edges["initial_condom_prob"])
    p1_after = np.asarray(net.edges.p1)
    p2_after = np.asarray(net.edges.p2)

    for i in range(len(edge_ids_before[0])):
        old_p1, old_p2 = edge_ids_before[0][i], edge_ids_before[1][i]
        matches = (p1_after == old_p1) & (p2_after == old_p2)
        for j in np.where(matches)[0]:
            assert np.isclose(icp_after[j], icp_before[i], atol=1e-10), f"initial_condom_prob changed for edge ({old_p1},{old_p2}): was {icp_before[i]:.4f}, now {icp_after[j]:.4f}"

    return sim


@sc.timer()
def test_gender_lookup_robust():
    """Negotiation should use actual sex lookup, not p1/p2 position assumption."""
    sc.heading("test_gender_lookup_robust")

    sim = make_sim(dur=1, seed=42)
    sim.init()

    intv = get_intv(sim)
    net = sim.networks.structuredsexual
    ppl = sim.people

    intv._condom_pref[np.asarray(ppl.female)] = 0.8
    intv._condom_pref[np.asarray(ppl.male)] = 0.2

    sim.run_one_step()

    icp = np.asarray(net.edges["initial_condom_prob"])
    assert np.all(icp >= 0.2 - 0.01) and np.all(icp <= 0.8 + 0.01), f"negotiated values should be between male (0.2) and female (0.8) prefs, got range [{icp.min():.3f}, {icp.max():.3f}]"

    return sim


@sc.timer()
def test_decay_formula_exact():
    """edges.condoms should match the exponential decay formula exactly.

    The formula is applied at the end of each step, so the test verifies the
    relationship between start_ti, icp, and condoms as they exist post-step.
    We need to account for the fact that edges stamped this step have age=0
    (no decay yet), and the condoms value reflects the formula at age = ti - start_ti - 1
    because _set_condom_probabilities runs *during* the step (before ti advances).
    """
    sc.heading("test_decay_formula_exact")

    base = 0.7
    floor = 0.1
    half_life = 6

    sim = make_sim(condom_data=base, decay_pars={"stable": {"half_life": half_life, "floor": floor}}, dur=3, seed=10)
    sim.init()

    for _ in range(half_life + 6):
        sim.run_one_step()

    net = sim.networks.structuredsexual

    stable = np.asarray(net.edges.edge_type) == net.edge_types["stable"]
    ages = net.ti - np.asarray(net.edges["start_ti"])
    icp = np.asarray(net.edges["initial_condom_prob"])
    condoms = np.asarray(net.edges.condoms)

    lam = np.log(2) / half_life
    old_stable = stable & (ages > 1)
    if not old_stable.any():
        return sim

    # verify relative decay: younger edges should have higher condom prob than older ones
    age_vals = ages[old_stable]
    cond_vals = condoms[old_stable]
    for a in np.unique(age_vals):
        older = age_vals > a
        if not older.any():
            continue
        assert np.all(cond_vals[age_vals == a] >= cond_vals[older].min() - 1e-6), f"edges at age {a} should have higher condom prob than older edges"

    # verify the decay formula matches within each age cohort using icp and ages
    # condoms were last written during this step, so age used by the code was (ti - start_ti)
    # but after the step, ti has advanced by 1, so the test sees ages one higher
    code_ages = ages[old_stable] - 1  # the age the code used when computing decay
    expected = floor + (icp[old_stable] - floor) * np.exp(-lam * code_ages.astype(float))
    assert np.allclose(condoms[old_stable], expected, atol=1e-5), f"decay formula mismatch: max diff={np.max(np.abs(condoms[old_stable] - expected)):.6f}"

    return sim


@sc.timer()
def test_decay_floor_approached():
    """After many steps, edges.condoms should approach floor but not go below it (when initial > floor)."""
    sc.heading("test_decay_floor_approached")

    base = 0.8
    floor = 0.15
    sim = make_sim(condom_data=base, decay_pars={"stable": {"half_life": 3, "floor": floor}}, dur=5, seed=5)
    sim.init()

    for _ in range(50):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    stable = np.asarray(net.edges.edge_type) == net.edge_types["stable"]
    ages = net.ti - np.asarray(net.edges["start_ti"])
    age_cutoff = 20  # well past the half-life
    old = stable & (ages > age_cutoff)

    if old.any():
        probs = np.asarray(net.edges.condoms)[old]
        assert np.all(probs >= floor - 1e-6), f"condom probs went below floor {floor}: min={probs.min():.4f}"
        assert np.all(probs < floor + 0.05), f"old edges should be near floor {floor}, got max={probs.max():.4f}"

    return sim


@sc.timer()
def test_decay_scoped_to_configured_types():
    """Decay only applies to configured edge types — others keep their base probability."""
    sc.heading("test_decay_scoped_to_configured_types")

    base = 0.7
    floor = 0.01
    sim = make_sim(condom_data=base, decay_pars={"stable": {"half_life": 3, "floor": floor}}, dur=3, seed=5)
    sim.init()

    for _ in range(20):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    casual = np.asarray(net.edges.edge_type) == net.edge_types["casual"]
    ages = net.ti - np.asarray(net.edges["start_ti"])
    age_cutoff = 6  # same age that would trigger decay on stable
    old_casual = casual & (ages > age_cutoff)
    assert old_casual.any(), "expected some old casual edges to test scoping against"

    probs = np.asarray(net.edges.condoms)[old_casual]
    assert np.all(probs > floor + 0.01), f"casual edges should not be decayed (floor={floor}), got {probs[:5]}"

    return sim


@sc.timer()
def test_decay_increases_transmission():
    """Decay (floor=0.0) removes condom protection, so transmission should go up."""
    sc.heading("test_decay_increases_transmission")

    base = 0.8
    sim_nodecay = make_sim(condom_data=base, decay_pars=None, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_nodecay.run(verbose=False)

    sim_decay = make_sim(condom_data=base, decay_pars={"stable": {"half_life": 3, "floor": 0.0}}, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_decay.run(verbose=False)

    inc_nodecay = sim_nodecay.summary.get("ct_incidence", 0)
    inc_decay = sim_decay.summary.get("ct_incidence", 0)

    assert inc_decay >= inc_nodecay, f"decay should increase transmission: decay={inc_decay:.4f} vs no-decay={inc_nodecay:.4f}"

    return sim_decay


@sc.timer()
def test_half_life_semantics():
    """After exactly one half-life, condom prob should be the midpoint between initial and floor."""
    sc.heading("test_half_life_semantics")

    base = 0.8
    floor = 0.2
    half_life = 6

    sim = make_sim(condom_data=base, decay_pars={"stable": {"half_life": half_life, "floor": floor}}, dur=3, seed=10)
    sim.init()

    for _ in range(half_life + 3):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    stable = np.asarray(net.edges.edge_type) == net.edge_types["stable"]
    ages = net.ti - np.asarray(net.edges["start_ti"])
    icp = np.asarray(net.edges["initial_condom_prob"])

    # ti advances after step, so the code used age=(ages-1); at the half-life the code age is half_life
    at_half_life = stable & (ages == half_life + 1)
    if not at_half_life.any():
        return sim

    expected = floor + (icp[at_half_life] - floor) * 0.5
    actual = np.asarray(net.edges.condoms)[at_half_life]
    assert np.allclose(actual, expected, atol=1e-6), f"half-life midpoint: expected {expected[:3]}, got {actual[:3]}"

    return sim


@sc.timer()
def test_boost_overrides_decay():
    """On a diagnosed agent's old decayed edge, boost should override the decay floor."""
    sc.heading("test_boost_overrides_decay")

    base = 0.6
    floor = 0.05
    boost_prob = 0.95
    half_life = 3

    sim = make_sim(
        condom_data=base,
        decay_pars={"stable": {"half_life": half_life, "floor": floor}},
        diagnosis_boost={"disease": "ct", "prob": {"stable": boost_prob, "casual": boost_prob}, "duration": 1000},
        dur=3,
        seed=42,
    )
    sim.init()
    diagnose_agent(sim, 0, "ct", ti=0)

    for _ in range(18):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    agent0_stable = get_agent_edges(net, 0) & (np.asarray(net.edges.edge_type) == net.edge_types["stable"])

    if agent0_stable.any():
        probs = np.asarray(net.edges.condoms)[agent0_stable]
        assert np.allclose(probs, boost_prob, atol=1e-10), f"boost ({boost_prob}) should override decay floor ({floor}), got {probs[:5]}"

    return sim


@sc.timer()
def test_boost_expires():
    """After the boost duration elapses, condom probability should revert to decayed value."""
    sc.heading("test_boost_expires")

    base = 0.3
    boost_prob = 0.95
    boost_dur = 3

    sim = make_sim(condom_data=base, diagnosis_boost={"disease": "ct", "prob": {"stable": boost_prob, "casual": boost_prob}, "duration": boost_dur}, dur=2, seed=42)
    sim.init()
    diagnose_agent(sim, 0, "ct", ti=1)

    for _ in range(8):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    assert net.ti > 1 + boost_dur, f"need ti > {1 + boost_dur} for boost to expire, got ti={net.ti}"

    agent0 = get_agent_edges(net, 0)
    assert agent0.any(), "agent 0 should have at least one edge after 8 steps"

    probs = np.asarray(net.edges.condoms)[agent0]
    assert np.all(probs < boost_prob - 0.01), f"after boost expires, condom prob should revert from {boost_prob} to ~{base}, got {probs[:5]}"

    return sim


@sc.timer()
def test_boost_scoped_to_diagnosed():
    """Boost only applies to edges of the diagnosed agent — other agents are unaffected."""
    sc.heading("test_boost_scoped_to_diagnosed")

    base = 0.3
    boost_prob = 0.99

    sim = make_sim(
        condom_data=base,
        diagnosis_boost={"disease": "ct", "prob": {"stable": boost_prob, "casual": boost_prob, "onetime": boost_prob, "sw": boost_prob}, "duration": 1000},
        dur=2,
        seed=42,
    )
    sim.init()
    diagnose_agent(sim, 0, "ct", ti=0)

    for _ in range(6):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    other = ~get_agent_edges(net, 0)
    assert other.any(), "expected edges not touching agent 0"

    other_probs = np.asarray(net.edges.condoms)[other]
    assert np.all(other_probs < boost_prob - 0.01), f"non-diagnosed agents should have base prob ~{base}, got {other_probs[:5]}"

    return sim


@sc.timer()
def test_treatment_triggers_diagnosis_boost():
    """When only diagnosis_boost is configured, treatment events should still trigger it."""
    sc.heading("test_treatment_triggers_diagnosis_boost")

    base = 0.3
    boost_prob = 0.95

    sim = make_sim(
        condom_data=base,
        diagnosis_boost={"disease": "ct", "prob": {"stable": boost_prob, "casual": boost_prob}, "duration": 1000},
        dur=2,
        seed=42,
    )
    sim.init()
    treat_agent(sim, 0, "ct", ti=0)

    for _ in range(6):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    agent0 = get_agent_edges(net, 0)
    assert agent0.any(), "agent 0 should have at least one edge after 6 steps"

    probs = np.asarray(net.edges.condoms)[agent0]
    assert np.any(probs >= boost_prob - 0.01), f"treatment should trigger diagnosis_boost, got {probs[:5]}"

    return sim


@sc.timer()
def test_simple_mode_scalar():
    """condom_data=0.5, no beta_params → edges.condoms ≈ 0.5 everywhere."""
    sc.heading("test_simple_mode_scalar")

    sim = make_sim(condom_data=0.5, dur=1, seed=42)
    sim.init()
    sim.run_one_step()

    net = sim.networks.structuredsexual
    probs = np.asarray(net.edges.condoms)
    if len(probs) > 0:
        assert np.allclose(probs, 0.5, atol=1e-10), f"scalar mode: expected condoms=0.5, got range [{probs.min():.3f}, {probs.max():.3f}]"

    return sim


@sc.timer()
def test_simple_mode_dict_of_edge_types():
    """condom_data={'stable': 0.4, 'casual': 0.6} → correct per-type values."""
    sc.heading("test_simple_mode_dict_of_edge_types")

    cd = {"stable": 0.4, "casual": 0.6, "onetime": 0.5, "sw": 0.8}
    sim = make_sim(condom_data=cd, dur=1, seed=42)
    sim.init()

    for _ in range(3):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    for etype_name, expected in cd.items():
        mask = np.asarray(net.edges.edge_type) == net.edge_types[etype_name]
        if not mask.any():
            continue
        probs = np.asarray(net.edges.condoms)[mask]
        assert np.allclose(probs, expected, atol=1e-10), f"{etype_name}: expected {expected}, got {probs[:5]}"

    return sim


@sc.timer()
def test_simple_mode_with_decay():
    """Scalar + decay_pars → decay applies from scalar baseline."""
    sc.heading("test_simple_mode_with_decay")

    base = 0.7
    floor = 0.1
    sim = make_sim(condom_data=base, decay_pars={"stable": {"half_life": 3, "floor": floor}}, dur=3, seed=10)
    sim.init()

    for _ in range(20):
        sim.run_one_step()

    net = sim.networks.structuredsexual
    stable = np.asarray(net.edges.edge_type) == net.edge_types["stable"]
    ages = net.ti - np.asarray(net.edges["start_ti"])
    age_cutoff = 10
    old = stable & (ages > age_cutoff)

    if old.any():
        probs = np.asarray(net.edges.condoms)[old]
        assert np.all(probs < base - 0.01), f"old stable edges should have decayed below {base}, got {probs[:5]}"
        assert np.all(probs >= floor - 1e-6), f"condom probs should not go below floor {floor}, got min={probs.min():.4f}"

    return sim


@sc.timer()
def test_noop_matches_structured():
    """BehavioralCondomUse(condom_data=0.5) with no decay/boost ≈ StructuredCondomUse(condom_data=0.5)."""
    sc.heading("test_noop_matches_structured")

    sim_base = ss.Sim(
        n_agents=n_agents,
        start=2000,
        dur=5,
        dt=1 / 12,
        rand_seed=42,
        diseases=[sti.Chlamydia()],
        networks=EdgeStructuredSexual(),
        interventions=[StructuredCondomUse(condom_data=0.5)],
    )
    sim_base.run(verbose=False)

    sim_bcu = ss.Sim(
        n_agents=n_agents,
        start=2000,
        dur=5,
        dt=1 / 12,
        rand_seed=42,
        diseases=[sti.Chlamydia()],
        networks=EdgeStructuredSexual(),
        interventions=[BehavioralCondomUse(condom_data=0.5)],
    )
    sim_bcu.run(verbose=False)

    rtol = 0.01
    for key in ["ct_prevalence", "ct_incidence"]:
        v_base = sim_base.summary.get(key, 0)
        v_bcu = sim_bcu.summary.get(key, 0)
        assert np.isclose(v_base, v_bcu, rtol=rtol), f"no-op BehavioralCondomUse should match StructuredCondomUse for {key}: base={v_base:.4f}, bcu={v_bcu:.4f}"

    return sim_bcu


@sc.timer()
def test_high_prefs_reduce_transmission():
    """High condom preference (a=5, b=0.5) should produce lower incidence than low preference (a=0.5, b=5)."""
    sc.heading("test_high_prefs_reduce_transmission")

    high_pref = {(rg, sex): (5.0, 0.5) for rg in range(3) for sex in ("f", "m")}
    low_pref = {(rg, sex): (0.5, 5.0) for rg in range(3) for sex in ("f", "m")}

    sim_hi = make_sim(beta_params=high_pref, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_hi.run(verbose=False)

    sim_lo = make_sim(beta_params=low_pref, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_lo.run(verbose=False)

    inc_hi = sim_hi.summary.get("ct_incidence", 0)
    inc_lo = sim_lo.summary.get("ct_incidence", 0)

    assert inc_hi <= inc_lo, f"high condom preference should reduce transmission: hi={inc_hi:.4f} vs lo={inc_lo:.4f}"

    return sim_hi


@sc.timer()
def test_low_female_weight_reduces_use():
    """Low female_weight with female-favoring prefs should produce lower effective condom use."""
    sc.heading("test_low_female_weight_reduces_use")

    beta_params = {(rg, "f"): (5.0, 0.5) for rg in range(3)}  # females prefer condoms, males don't
    beta_params.update({(rg, "m"): (0.5, 5.0) for rg in range(3)})

    low_female = {"stable": 0.1, "casual": 0.1, "onetime": 0.1, "sw": 0.1}
    high_female = {"stable": 0.9, "casual": 0.9, "onetime": 0.9, "sw": 0.9}

    sim_low = make_sim(beta_params=beta_params, negotiation_weights=low_female, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_low.run(verbose=False)

    sim_high = make_sim(beta_params=beta_params, negotiation_weights=high_female, diseases=[sti.Chlamydia(init_prev=0.05)], dur=8, seed=1)
    sim_high.run(verbose=False)

    inc_low = sim_low.summary.get("ct_incidence", 0)
    inc_high = sim_high.summary.get("ct_incidence", 0)

    assert inc_low >= inc_high, f"low female weight should increase transmission: low={inc_low:.4f} vs high={inc_high:.4f}"

    return sim_low


if __name__ == "__main__":
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    test_prefs_drawn_from_beta()
    test_negotiation_weights()
    test_initial_prob_persists()
    test_gender_lookup_robust()
    test_decay_formula_exact()
    test_decay_floor_approached()
    test_decay_scoped_to_configured_types()
    test_decay_increases_transmission()
    test_half_life_semantics()
    test_boost_overrides_decay()
    test_boost_expires()
    test_boost_scoped_to_diagnosed()
    test_treatment_triggers_diagnosis_boost()
    test_simple_mode_scalar()
    test_simple_mode_dict_of_edge_types()
    test_simple_mode_with_decay()
    test_noop_matches_structured()
    test_high_prefs_reduce_transmission()
    test_low_female_weight_reduces_use()

    T.toc()

    if do_plot:
        plt.show()
