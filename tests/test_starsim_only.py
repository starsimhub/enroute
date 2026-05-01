"""Tests for EdgeNetwork and CondomUse using starsim-only (ss.SIR)."""

import numpy as np
import sciris as sc

import starsim as ss
from enroute import EdgeNetwork, CondomUse

n_agents = 200
sc.options(interactive=False)


class SimpleMFEdgeNet(EdgeNetwork):
    """Minimal MF network for testing without stisim."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            duration=ss.lognorm_ex(mean=ss.years(15), std=ss.years(1)),
            debut=ss.normal(loc=16),
            acts=ss.poisson(lam=ss.freqperyear(80)),
            participation=ss.bernoulli(p=0.9),
        )
        self.dist = ss.choice(name="SimpleMFEdgeNet", replace=False)

    def append(self, edges=None, **kwargs):
        """Fill dynamically-registered meta columns with zeros."""
        edges = sc.mergedicts(edges, kwargs)
        for key in self.meta_keys():
            if key not in edges:
                some_key = next(iter(edges))
                n = len(edges[some_key])
                edges[key] = np.zeros(n, dtype=self.meta[key])
        super().append(edges)

    def init_post(self):
        self.set_network_states()
        super().init_post()

    def set_network_states(self, upper_age=None):
        people = self.sim.people
        if upper_age is None:
            uids = people.auids
        else:
            uids = (people.age < upper_age).uids
        self.debut[uids] = self.pars.debut.rvs(uids)
        self.participant[uids] = self.pars.participation.rvs(uids)

    def add_pairs(self):
        people = self.sim.people
        available_m = self.available(people, "male")
        available_f = self.available(people, "female")
        if len(available_m) <= len(available_f):
            self.dist.set(a=available_f)
            p1 = available_m
            p2 = self.dist.rvs(n=len(p1))
        else:
            self.dist.set(a=available_m)
            p2 = available_f
            p1 = self.dist.rvs(n=len(p2))
        self.dist.jump()
        beta = np.ones_like(p1)
        dur_vals = self.pars.duration.rvs(p1)
        act_vals = self.pars.acts.rvs(p1).astype(int)
        self.append(p1=p1, p2=p2, beta=beta, dur=dur_vals, acts=act_vals)
        return len(p1)

    def step(self):
        self.end_pairs()
        self.set_network_states(upper_age=float(self.t.dt))
        self.add_pairs()


def make_sim(interventions=None, seed=42, dur=2, init_prev=0.1, beta=0.1):
    """Build a minimal starsim-only sim."""
    net = SimpleMFEdgeNet()
    disease = ss.SIR(beta=beta, dur_inf=ss.normal(loc=10))
    disease.pars.init_prev = ss.bernoulli(p=init_prev)
    pars = dict(
        n_agents=n_agents,
        rand_seed=seed,
        start=2000,
        dur=dur,
        dt=1 / 12,
        verbose=0,
        diseases=[disease],
        networks=[net],
    )
    if interventions is not None:
        pars["interventions"] = interventions
    return ss.Sim(**pars)


def make_sim_with_condoms(eff=0.8, condom_data=0.5, seed=42, **kwargs):
    """Build a sim with a CondomUse intervention attached to a SimpleMFEdgeNet."""
    net = SimpleMFEdgeNet()
    cu = CondomUse(eff=eff, condom_data=condom_data, network=net)
    disease = ss.SIR(beta=0.1, dur_inf=ss.normal(loc=10))
    disease.pars.init_prev = ss.bernoulli(p=0.1)
    pars = dict(
        n_agents=n_agents,
        rand_seed=seed,
        start=2000,
        dur=kwargs.pop("dur", 2),
        dt=1 / 12,
        verbose=0,
        diseases=[disease],
        networks=[net],
        interventions=[cu],
    )
    pars.update(kwargs)
    return ss.Sim(**pars)


@sc.timer()
def test_net_beta_no_interventions():
    """Without interventions, EdgeNetwork.net_beta should match ss.SexualNetwork.net_beta."""
    sc.heading("Testing net_beta matches SexualNetwork (no interventions)")
    sim = make_sim()
    sim.init()
    sim.run_one_step()

    net = sim.networks.simplemfedgenet
    disease = sim.diseases.sir
    disease_beta = disease.pars.beta
    db = disease_beta.to_prob(sim.t.dt) if isinstance(disease_beta, ss.Rate) else disease_beta

    expected = ss.SexualNetwork.net_beta(net, disease_beta=db, disease=disease)
    result = net.net_beta(disease_beta=db, disease=disease)

    rtol = 1e-6  # float32 edge arrays limit precision
    np.testing.assert_allclose(result, expected, rtol=rtol, err_msg="EdgeNetwork.net_beta should match SexualNetwork.net_beta without interventions")
    return sim


@sc.timer()
def test_condom_data_zero():
    """With condom_data=0, all Bernoulli draws should produce zero uses."""
    sc.heading("Testing condom_data=0 produces zero uses")
    sim = make_sim_with_condoms(eff=0.8, condom_data=0)
    sim.init()
    sim.run_one_step()

    net = sim.networks.simplemfedgenet
    uses = net.edges.condomuse_uses
    assert np.all(uses == 0), f"Expected all condomuse_uses==0 with condom_data=0, but max={uses.max()}"
    return sim


@sc.timer()
def test_condom_data_one():
    """With condom_data=1.0, every act should be a condom use."""
    sc.heading("Testing condom_data=1.0 produces uses==acts")
    sim = make_sim_with_condoms(eff=0.8, condom_data=1.0)
    sim.init()
    sim.run_one_step()

    net = sim.networks.simplemfedgenet
    uses = net.edges.condomuse_uses
    acts = net.edges.acts
    has_acts = acts > 0
    assert np.all(uses[has_acts] == acts[has_acts]), f"Expected condomuse_uses==acts on edges with acts>0 when condom_data=1.0"
    return sim


@sc.timer()
def test_condom_net_beta_analytical():
    """EdgeNetwork.net_beta with condoms should match the analytical expected value."""
    sc.heading("Testing condom net_beta vs analytical expectation")
    # from calc_tolerances
    db, eff, condom_data, n_acts, n_agents, rtol = (0.0083, 0.8, 0.5, 100, 200, 0.02)
    net = SimpleMFEdgeNet()
    net.pars.acts = ss.constant(v=n_acts)
    cu = CondomUse(eff=eff, condom_data=condom_data, network=net)
    disease = ss.SIR(beta=db, dur_inf=ss.normal(loc=10))
    disease.pars.init_prev = ss.bernoulli(p=0.1)
    sim = ss.Sim(
        n_agents=n_agents,
        rand_seed=42,
        start=2000,
        dur=2,
        dt=1 / 12,
        verbose=0,
        diseases=[disease],
        networks=[net],
        interventions=[cu],
    )
    sim.init()
    sim.run_one_step()

    net = sim.networks.simplemfedgenet
    disease = sim.diseases.sir
    disease_beta = disease.pars.beta
    db = disease_beta.to_prob(sim.t.dt) if isinstance(disease_beta, ss.Rate) else disease_beta

    acts = net.edges.acts
    edge_beta = net.edges.beta
    result = net.net_beta(disease_beta=db, disease=disease)

    # Analytical: each act has prob condom_data of being protected, protected acts transmit at db*(1-eff)
    db_eff = db * (1 - condom_data * eff)
    expected = edge_beta * (1 - (1 - db_eff) ** acts)

    mean_expected = expected.mean()
    mean_result = result.mean()
    rel_diff = abs(mean_result - mean_expected) / mean_expected

    assert rel_diff < rtol, f"Mean net_beta rel_diff={rel_diff:.4%} exceeds {rtol:.0%} (expected={mean_expected:.6f}, result={mean_result:.6f})"
    return sim


@sc.timer()
def test_reduces_transmission():
    """Full condom coverage with eff=1.0 should reduce cumulative infections."""
    sc.heading("Testing full condom coverage reduces transmission")
    sim_no = make_sim(seed=7, dur=5, init_prev=0.1, beta=0.15)
    sim_no.run(verbose=False)

    net = SimpleMFEdgeNet()
    cu = CondomUse(eff=1.0, condom_data=1.0, network=net)
    disease = ss.SIR(beta=0.15, dur_inf=ss.normal(loc=10))
    disease.pars.init_prev = ss.bernoulli(p=0.1)
    sim_full = ss.Sim(
        n_agents=n_agents,
        rand_seed=7,
        start=2000,
        dur=5,
        dt=1 / 12,
        verbose=0,
        diseases=[disease],
        networks=[net],
        interventions=[cu],
    )
    sim_full.run(verbose=False)

    cum_no = sim_no.summary.get("sir_cum_infections", 0)
    cum_full = sim_full.summary.get("sir_cum_infections", 0)
    assert cum_full < cum_no, f"Full condom coverage (eff=1.0) should reduce infections: no-condom={cum_no:.1f}, with-condom={cum_full:.1f}"
    return sim_full


if __name__ == "__main__":
    T = sc.timer()

    test_net_beta_no_interventions()
    test_condom_data_zero()
    test_condom_data_one()
    test_condom_net_beta_analytical()
    test_reduces_transmission()

    T.toc()
