"""BehavioralCondomUse: per-agent preference, gender-weighted negotiation, exponential decay, and diagnosis boosts."""

import starsim as ss
import numpy as np
import sciris as sc

from enroute.sti import StructuredCondomUse

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int


DEFAULT_BETA_PARAMS = {
    (0, "f"): (0.3, 0.7),
    (0, "m"): (0.3, 0.7),  # low-risk: mean ~0.30
    (1, "f"): (0.5, 0.5),
    (1, "m"): (0.5, 0.5),  # med-risk: mean  0.50
    (2, "f"): (0.7, 0.3),
    (2, "m"): (0.7, 0.3),  # high-risk: mean ~0.70
}

DEFAULT_NEGOTIATION_WEIGHTS = {
    "stable": 0.3,  # male-dominated decision in SSA literature
    "casual": 0.4,
    "onetime": 0.5,
    "sw": 0.7,  # FSW has more negotiating power with clients
}


class BehavioralCondomUse(StructuredCondomUse):
    """Per-agent condom preference with gender-weighted negotiation, exponential fatigue decay, and diagnosis boosts.

    Two modes:
        Preference mode (beta_params provided or condom_data is None):
            Each agent gets a condom_pref drawn from Beta(a,b). New edges negotiate
            an initial probability as a gender-weighted average of partner preferences.
            That initial value decays exponentially toward a floor over the relationship.

        Simple mode (condom_data provided, beta_params is None):
            Delegates to StructuredCondomUse for baseline probabilities. Decay and
            boost can still be applied on top.

    Args:
        condom_data     (float/dict): scalar, dict of edge-type->prob, or risk-group DataFrame
        beta_params     (dict):       (risk_group, 'f'|'m') -> (a, b) for Beta distribution
        negotiation_weights (dict):   edge_type_name -> female_weight in [0, 1]
        decay_pars      (dict):       edge_type_name -> {'half_life': ss.dur|number, 'floor': float}
        diagnosis_boost (dict):       {'disease': str, 'prob': {etype: float}, 'duration': ss.dur}
        treatment_boost (dict):       same shape as diagnosis_boost
    """

    def __init__(self, condom_data=None, beta_params=None, negotiation_weights=None, decay_pars=None, diagnosis_boost=None, treatment_boost=None, name="condomuse", **kwargs):
        use_negotiation = beta_params is not None or condom_data is None

        # parent shouldn't see condom_data in preference mode or edge-type-dict mode
        if use_negotiation:
            parent_data = None
        elif isinstance(condom_data, dict) and _is_edge_type_dict(condom_data):
            parent_data = None
        else:
            parent_data = condom_data

        super().__init__(condom_data=parent_data, name=name, **kwargs)

        self._preference_mode = use_negotiation
        self._beta_params = beta_params if beta_params is not None else dict(DEFAULT_BETA_PARAMS)
        self._negotiation_weights = negotiation_weights if negotiation_weights is not None else dict(DEFAULT_NEGOTIATION_WEIGHTS)
        self._decay_pars = decay_pars if decay_pars is not None else {}
        self._diagnosis_boost = diagnosis_boost
        self._treatment_boost = treatment_boost
        self._condom_data_by_etype = condom_data if (isinstance(condom_data, dict) and _is_edge_type_dict(condom_data)) else None
        self._condom_pref = None  # allocated in init_post
        self._prefs_drawn = False
        self.pref_dist = ss.beta_dist(a=1, b=1)  # placeholder; auto-discovered by ss.Dists
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        net = self._network

        if self._decay_pars or self._preference_mode:
            if "start_ti" not in net.meta:
                net.meta["start_ti"] = ss_int_
                net.edges["start_ti"] = np.empty((0,), dtype=ss_int_)
            if "initial_condom_prob" not in net.meta:
                net.meta["initial_condom_prob"] = ss_float_
                net.edges["initial_condom_prob"] = np.empty((0,), dtype=ss_float_)
            if "_icp_set" not in net.meta:
                net.meta["_icp_set"] = ss_int_
                net.edges["_icp_set"] = np.empty((0,), dtype=ss_int_)  # 0=unset, 1=initialized

        self._validate_pars(net)

        for etype, epars in self._decay_pars.items():
            steps = self._to_steps(epars["half_life"])
            epars["_lambda"] = np.log(2) / steps if steps > 0 else 0.0

        for cfg in [self._diagnosis_boost, self._treatment_boost]:
            if cfg is not None:
                cfg["_duration_steps"] = self._to_steps(cfg["duration"])

        return

    def init_post(self):
        super().init_post()
        if self._preference_mode:
            self._draw_condom_prefs()
        return

    def _draw_condom_prefs(self):
        """Allocate and fill per-agent condom preferences from Beta(a,b) per (risk_group, sex)."""
        net = self._network
        people = self.sim.people
        n_uids = people.n_uids  # total allocated UID slots (includes dead agents)

        self._condom_pref = np.zeros(n_uids, dtype=ss_float_)
        self._fill_prefs(net, people, np.arange(n_uids))
        self._prefs_drawn = True
        return

    def _fill_prefs(self, net, people, uids):
        """Draw Beta preferences for a set of uids. Shared by init and capacity growth."""
        for (rg, sex_label), (a, b) in self._beta_params.items():
            sex_mask = people.female if sex_label == "f" else people.male
            in_group = (net.risk_group.raw[uids] == rg) & sex_mask.raw[uids]
            target = uids[in_group]
            if len(target) == 0:
                continue
            self.pref_dist.set(a=a, b=b)
            self._condom_pref[target] = self.pref_dist.rvs(target)
        return

    def _ensure_pref_capacity(self):
        """Grow _condom_pref if demographics added agents since last check."""
        if self._condom_pref is None:
            return
        n_needed = self.sim.people.n_uids
        if n_needed <= len(self._condom_pref):
            return
        old_len = len(self._condom_pref)
        self._condom_pref = np.concatenate([self._condom_pref, np.zeros(n_needed - old_len, dtype=ss_float_)])
        new_uids = np.arange(old_len, n_needed)
        self._fill_prefs(self._network, self.sim.people, new_uids)
        return

    def step(self):
        net = self._network

        if self._preference_mode:
            self._ensure_pref_capacity()

        if self._decay_pars or self._preference_mode:
            self._stamp_new_edges(net)

        if self._preference_mode:
            self._negotiate_initial_probs(net)

        super().step()
        return

    def _stamp_new_edges(self, net):
        """Mark new edges with start_ti and set initial_condom_prob to NaN (sentinel for negotiation).

        Uses _icp_set==0 to detect uninitialized edges (zero-fill from append()).
        This avoids the false-positive where initial_condom_prob==0.0 is a legitimate
        probability (e.g. condom_data=0.0 or Beta(a<1, b) producing exact zero).
        The NaN sentinel on initial_condom_prob then flags them for negotiation/snapshotting,
        which sets _icp_set=1 to prevent re-stamping on subsequent steps.
        """
        icp_set = net.edges["_icp_set"]
        if len(icp_set) == 0:
            return

        # edges from append() arrive with _icp_set=0
        new = np.asarray(icp_set) == 0
        if not new.any():
            return

        net.edges["start_ti"][new] = net.ti
        net.edges["initial_condom_prob"][new] = np.nan
        return

    def _negotiate_initial_probs(self, net):
        """Compute gender-weighted condom probability for edges that need it (NaN sentinel)."""
        icp = np.asarray(net.edges["initial_condom_prob"])
        needs_init = np.isnan(icp)
        if not needs_init.any():
            return

        p1 = net.edges.p1[needs_init]
        p2 = net.edges.p2[needs_init]

        is_female_p1 = self.sim.people.female.raw[p1]  # don't assume p1=male; index by UID into raw
        pref_p1 = self._condom_pref[p1]
        pref_p2 = self._condom_pref[p2]
        pref_f = np.where(is_female_p1, pref_p1, pref_p2)
        pref_m = np.where(is_female_p1, pref_p2, pref_p1)

        etypes = net.edges.edge_type[needs_init]
        result = np.full(int(needs_init.sum()), fill_value=0.5, dtype=ss_float_)  # default weight if type not in dict

        for etype_name, etype_code in net.edge_types.items():
            mask = etypes == etype_code
            if not mask.any():
                continue
            w = self._negotiation_weights.get(etype_name, 0.5)
            result[mask] = w * pref_f[mask] + (1 - w) * pref_m[mask]

        net.edges["initial_condom_prob"][needs_init] = result
        net.edges["_icp_set"][needs_init] = 1  # mark as initialized; prevents re-stamping
        return

    def _set_condom_probabilities(self, net):
        """Write per-edge condom probability into edges.condoms, applying decay and boosts."""
        if self._preference_mode:
            icp = np.asarray(net.edges["initial_condom_prob"])
            net.edges.condoms[:] = icp
        elif self._condom_data_by_etype is not None:
            for etype_name, prob in self._condom_data_by_etype.items():
                if etype_name in net.edge_types:
                    mask = net.edges.edge_type == net.edge_types[etype_name]
                    net.edges.condoms[mask] = prob
        else:
            super()._set_condom_probabilities(net)

        # in simple mode, snapshot baseline into initial_condom_prob so decay has a reference
        if self._decay_pars and not self._preference_mode:
            icp = np.asarray(net.edges["initial_condom_prob"])
            new = np.isnan(icp)
            if new.any():
                icp[new] = np.asarray(net.edges.condoms)[new]
                net.edges["initial_condom_prob"][:] = icp
                net.edges["_icp_set"][new] = 1  # mark as initialized; prevents re-stamping

        if self._decay_pars:
            self._apply_decay(net)

        self._apply_boosts(net)

        if np.isnan(np.asarray(net.edges.condoms)).any():
            raise ValueError("NaN survived in edges.condoms after _set_condom_probabilities")

        return

    def _apply_decay(self, net):
        """p(t) = floor + (initial - floor) * exp(-lambda * t). Floor is an asymptote, not a strict minimum."""
        icp = np.asarray(net.edges["initial_condom_prob"])
        ages = (net.ti - np.asarray(net.edges["start_ti"])).astype(float)

        for etype_name, epars in self._decay_pars.items():
            etype_code = net.edge_types[etype_name]
            mask = np.asarray(net.edges.edge_type) == etype_code
            if not mask.any():
                continue
            floor = epars["floor"]
            lam = epars["_lambda"]
            net.edges.condoms[mask] = floor + (icp[mask] - floor) * np.exp(-lam * ages[mask])

        return

    def _apply_boosts(self, net):
        has_dx = self._diagnosis_boost is not None
        has_tx = self._treatment_boost is not None

        if not has_dx and not has_tx:
            return

        if has_dx:
            triggered = self._check_events(net, self._diagnosis_boost, dx=True, tx=not has_tx)
            self._boost_edges(net, triggered, self._diagnosis_boost["prob"])

        if has_tx:
            triggered = self._check_events(net, self._treatment_boost, dx=not has_dx, tx=True)
            self._boost_edges(net, triggered, self._treatment_boost["prob"])

        return

    @staticmethod
    def _raw(arr):
        """Return the UID-indexed raw array from an ss.Arr, or the array itself if already numpy."""
        return arr.raw if hasattr(arr, "raw") else arr

    def _check_events(self, net, cfg, dx=False, tx=False):
        dur = cfg["_duration_steps"]
        ti = net.ti
        triggered = np.zeros(self.sim.people.n_uids, dtype=bool)  # UID-indexed; p1/p2 are UID values

        if "disease" in cfg:
            disease = self.sim.diseases[cfg["disease"]]
            if dx and hasattr(disease, "diagnosed"):
                triggered |= self._raw(disease.diagnosed) & (self._raw(disease.ti_diagnosed) >= (ti - dur))
            if tx and hasattr(disease, "treated"):
                triggered |= self._raw(disease.treated) & (self._raw(disease.ti_treated) >= (ti - dur))
        elif "test" in cfg:
            intv = self.sim.interventions[cfg["test"]]
            if dx:
                triggered |= self._raw(intv.diagnosed) & (self._raw(intv.ti_positive) >= (ti - dur))

        return triggered

    def _boost_edges(self, net, triggered, prob_dict):
        p1_boosted = triggered[net.edges.p1]
        p2_boosted = triggered[net.edges.p2]
        any_boosted = p1_boosted | p2_boosted

        if not any_boosted.any():
            return

        for etype, prob in prob_dict.items():
            mask = net.edges.edge_type == net.edge_types[etype]
            apply = any_boosted & mask
            net.edges.condoms[apply] = np.fmax(net.edges.condoms[apply], prob)  # fmax propagates non-NaN

        return

    def _validate_pars(self, net):
        for etype in self._decay_pars.keys():
            if etype not in net.edge_types:
                errormsg = f"BehavioralCondomUse: decay_pars key '{etype}' not in edge_types {list(net.edge_types.keys())}"
                raise ValueError(errormsg)
            if "half_life" not in self._decay_pars[etype]:
                raise ValueError(f"BehavioralCondomUse: decay_pars['{etype}'] missing 'half_life'")
            if "floor" not in self._decay_pars[etype]:
                raise ValueError(f"BehavioralCondomUse: decay_pars['{etype}'] missing 'floor'")

        if self._preference_mode:
            for etype in self._negotiation_weights.keys():
                if etype not in net.edge_types:
                    errormsg = f"BehavioralCondomUse: negotiation_weights key '{etype}' not in edge_types {list(net.edge_types.keys())}"
                    raise ValueError(errormsg)
            for rg in range(net.pars.n_risk_groups):
                for sex in ("f", "m"):
                    if (rg, sex) not in self._beta_params:
                        raise ValueError(f"BehavioralCondomUse: beta_params missing key ({rg}, '{sex}')")

        if self._condom_data_by_etype is not None:
            for etype in self._condom_data_by_etype.keys():
                if etype not in net.edge_types:
                    errormsg = f"BehavioralCondomUse: condom_data key '{etype}' not in edge_types {list(net.edge_types.keys())}"
                    raise ValueError(errormsg)

        return

    def _to_steps(self, val):
        if isinstance(val, ss.dur):
            return int(round(val.years / self.t.dt_year))
        elif sc.isnumber(val):
            return int(val)
        else:
            raise TypeError(f"BehavioralCondomUse: expected ss.dur or number, got {type(val)}")


def _is_edge_type_dict(d):
    """Check if a dict has string keys (edge-type names) rather than tuple keys (risk-group pairs)."""
    if not isinstance(d, dict) or len(d) == 0:
        return False
    first_key = next(iter(d))
    return isinstance(first_key, str)
