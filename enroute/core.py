"""
Edge/Act interface for sexual networks with per-act intervention support.
Starsim-only. See sti.py for stisim classes.
"""

import starsim as ss
import sciris as sc
import numpy as np

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int


class EdgeNetwork(ss.SexualNetwork):
    """ss.SexualNetwork with intervention-aware net_beta (stacking formula)."""

    def init_pre(self, sim):
        super().init_pre(sim)
        self._edge_intvs = []
        return

    def net_beta(self, disease_beta=None, uids=None, disease=None):
        if uids is None:
            uids = Ellipsis
        acts = self.edges.acts[uids]

        edge_intvs = self._edge_intvs

        if not edge_intvs:
            return self.edges.beta[uids] * (1 - (1 - disease_beta) ** acts)

        n_edges = len(acts)
        unprotected_frac = np.ones(n_edges, dtype=ss_float_)
        p_trans = np.ones(n_edges, dtype=ss_float_)

        for iv in edge_intvs:
            uses = self.edges[iv.edge_col][uids]
            rate = np.divide(uses, acts, out=np.zeros_like(uses, dtype=float), where=acts > 0)
            rate = np.clip(rate, 0, 1)  # guard against uses > acts from buggy subclasses
            rr = iv.relative_risk(self, disease_beta, disease, uids)
            unprotected_frac *= 1 - rate
            p_trans *= (1 - disease_beta * rr) ** (acts * rate)

        p_trans *= (1 - disease_beta) ** (acts * unprotected_frac)
        return self.edges.beta[uids] * (1 - p_trans)


class EdgeIntervention(ss.Intervention):
    """Base for per-act edge-level interventions.

    Subclass contract:
        update_uses(network): Set self.edge_uses to integer protected-act counts per edge.
                              Must satisfy 0 <= edge_uses[i] <= network.edges.acts[i].
        relative_risk(network, disease_beta, disease, uids):
                              Return a scalar or per-edge array of relative risk.
                              1.0 = no protection; 0.0 = full block.

    Multiple EdgeInterventions stack multiplicatively in EdgeNetwork.net_beta().
    """

    def __init__(self, network=None, diseases=True, **kwargs):
        super().__init__(**kwargs)
        self.diseases = diseases
        self._network = network
        self.uses_dist = ss.bernoulli(p=0)  # Auto-discovered by ss.Dists for seeded RNG
        return

    @property
    def edge_col(self):
        return f"{self.name}_uses"

    @property
    def edge_uses(self):
        return self._network.edges[self.edge_col]

    @edge_uses.setter
    def edge_uses(self, value):
        self._network.edges[self.edge_col] = value

    def init_pre(self, sim):
        super().init_pre(sim)

        if self.diseases is True:
            self.diseases = list(sim.diseases.keys())
        else:
            diseases = sc.promotetolist(self.diseases)
            self.diseases = [d if isinstance(d, str) else d.name for d in diseases]

        if self._network is None:
            edge_nets = [n for n in sim.networks.values() if isinstance(n, EdgeNetwork)]
            if len(edge_nets) == 0:
                errormsg = f"{self.__class__.__name__} requires at least one EdgeNetwork in the simulation."
                raise ValueError(errormsg)
            if len(edge_nets) > 1:
                errormsg = f"{self.__class__.__name__} found {len(edge_nets)} EdgeNetworks but no network was specified. Pass network=<name_or_instance> to disambiguate."
                raise ValueError(errormsg)
            self._network = edge_nets[0]
        elif isinstance(self._network, str):
            self._network = sim.networks[self._network]

        self._network._edge_intvs.append(self)
        net = self._network
        if self.edge_col not in net.meta:
            net.meta[self.edge_col] = ss_int_
            net.edges[self.edge_col] = np.empty((0,), dtype=ss_int_)
        return

    def step(self):
        self.update_uses(self._network)
        return

    def update_uses(self, network):
        """Set self.edge_uses to integer protected act counts per edge."""
        raise NotImplementedError(f"{type(self).__name__} must implement update_uses()")

    def relative_risk(self, network, disease_beta, disease, uids=None):
        """Return scalar or per-edge relative risk. 1.0 = no protection; 0.0 = full block."""
        raise NotImplementedError(f"{type(self).__name__} must implement relative_risk()")


class CondomUse(EdgeIntervention):
    """Condom use via per-act Bernoulli draws. Stores protected act counts in edges.<name>_uses."""

    def __init__(self, eff=None, condom_data=None, **kwargs):
        super().__init__(**kwargs)
        if eff is None:
            eff = {
            "ng": 0.9,   # doi:10.1001/archpedi.159.6.536
            "ct": 0.46,  # doi:10.1001/archpedi.159.6.536
            "hiv": 0.9,
            "bv": 0,
            "syph": 0.14,
        }
        self.eff = eff
        self.condom_data = condom_data
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        net = self._network
        if "condoms" not in net.meta:
            net.meta.condoms = ss_float_
            net.edges.condoms = np.empty((0,), dtype=ss_float_)
        return

    def _set_condom_probabilities(self, net):
        """Write condom probability per edge into edges.condoms."""
        if sc.isnumber(self.condom_data):
            net.edges.condoms[:] = self.condom_data
        return

    def update_uses(self, network):
        self._set_condom_probabilities(network)
        acts = network.edges.acts
        prob = network.edges.condoms
        total_acts = int(acts.sum())
        if total_acts == 0:
            self.edge_uses = np.zeros(len(acts), dtype=ss_int_)
            return
        # Only process edges with acts > 0 to avoid reduceat misattribution
        has_acts = acts > 0
        sub_acts = acts[has_acts]
        sub_prob = prob[has_acts]
        per_act_prob = np.repeat(sub_prob, sub_acts)
        self.uses_dist.set(p=per_act_prob)
        draws = self.uses_dist.rvs(int(sub_acts.sum()))
        offsets = np.concatenate([[0], np.cumsum(sub_acts[:-1])])
        uses = np.zeros(len(acts), dtype=ss_int_)
        uses[has_acts] = np.add.reduceat(draws.astype(ss_int_), offsets)
        self.edge_uses = uses
        return

    def relative_risk(self, network, disease_beta, disease, uids=None):
        if isinstance(self.eff, dict):
            disease_name = disease.name if hasattr(disease, "name") else str(disease)
            if disease_name not in self.eff:
                errormsg = f"CondomUse.eff has no entry for disease '{disease_name}'; known keys: {list(self.eff.keys())}. Assuming eff=0 (no protection)."
                ss.warn(errormsg)
            eff = self.eff.get(disease_name, 0)
        else:
            eff = self.eff
        return 1 - eff
