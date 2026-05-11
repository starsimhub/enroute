"""stisim-compatible edge interface: EdgeStructuredSexual, StructuredCondomUse."""

import starsim as ss
import sciris as sc
import numpy as np
import stisim as sti
import pandas as pd
from collections import defaultdict
from bisect import bisect_left
from stisim.networks import NoPartnersFound, PriorPartners

from .core import EdgeNetwork, CondomUse

ss_float_ = ss.dtypes.float


class EdgeNetworkPars(sti.NetworkPars):
    def __init__(self, **kwargs):
        super().__init__()
        del self.condom_data
        return


class EdgeStructuredSexual(EdgeNetwork):
    """StructuredSexual reimplemented on EdgeNetwork. Risk groups, concurrency, sex work, edge types."""

    def __init__(self, pars=None, **kwargs):
        super().__init__(name="structuredsexual")

        self.meta.sw = bool
        self.meta.age_p1 = ss_float_
        self.meta.age_p2 = ss_float_
        self.meta.edge_type = ss_float_

        default_pars = EdgeNetworkPars()
        self.define_pars(**default_pars)
        self.update_pars(pars, **kwargs)

        self.edge_types = {"stable": 0, "casual": 1, "onetime": 2, "sw": 3}

        self.define_states(
            ss.BoolArr("participant", default=True),
            ss.FloatArr("debut", default=0),
            ss.FloatArr("risk_group"),
            ss.BoolArr("fsw"),
            ss.BoolArr("client"),
            ss.FloatArr("concurrency"),
            ss.FloatArr("partners", default=0),
            ss.FloatArr("partners_12", default=0),
            ss.FloatArr("lifetime_partners", default=0),
            ss.FloatArr("casual_partners", default=0),
            ss.FloatArr("stable_partners", default=0),
            ss.FloatArr("onetime_partners", default=0),
            ss.FloatArr("sw_partners", default=0),
            ss.FloatArr("lifetime_casual_partners", default=0),
            ss.FloatArr("lifetime_stable_partners", default=0),
            ss.FloatArr("lifetime_onetime_partners", default=0),
            ss.FloatArr("lifetime_sw_partners", default=0),
            ss.FloatArr("sw_intensity"),
            reset=True,
        )

        self.relationship_durs = defaultdict(list)
        return

    def append(self, edges=None, **kwargs):
        edges = sc.mergedicts(edges, kwargs)
        for key in self.meta_keys():
            if key not in edges:
                some_key = next(iter(edges))
                n = len(edges[some_key])
                edges[key] = np.zeros(n, dtype=self.meta[key])
        super().append(edges)

    def get_age_risk_pars(self, uids, par):
        loc = np.full(uids.shape, fill_value=np.nan)
        scale = np.full(uids.shape, fill_value=np.nan)
        for a_label, (age_lower, age_upper) in self.pars.f_age_group_bins.items():
            for rg in range(self.pars.n_risk_groups):
                in_risk_group = (self.sim.people.age[uids] >= age_lower) & (self.sim.people.age[uids] < age_upper) & (self.risk_group[uids] == rg)
                p0 = par[a_label][rg][0]
                p1 = par[a_label][rg][1]
                if isinstance(p0, ss.dur):
                    p0 = p0.months
                    p1 = p1.months
                loc[in_risk_group] = p0
                scale[in_risk_group] = p1
        if np.isnan(scale).any() or np.isnan(loc).any():
            errormsg = "Invalid entries for age difference preferences."
            raise ValueError(errormsg)
        return loc, scale

    def init_pre(self, sim):
        super().init_pre(sim)

        if self.pars.recall_prior:
            isprior = [isinstance(nw, PriorPartners) for nw in self.sim.networks.values()]
            if not any(isprior):
                errormsg = "PriorPartners network is required if recall_prior is True."
                raise ValueError(errormsg)
        return

    def init_post(self):
        super().init_post(add_pairs=False)
        self.set_network_states()
        return

    def set_network_states(self, upper_age=None):
        self.set_risk_groups(upper_age=upper_age)
        self.set_concurrency(upper_age=upper_age)
        self.set_sex_work(upper_age=upper_age)
        self.set_debut(upper_age=upper_age)
        return

    @property
    def over_debut(self):
        return self.sim.people.age > self.debut

    def _get_uids(self, upper_age=None, by_sex=True):
        people = self.sim.people
        if upper_age is None:
            upper_age = 1000
        within_age = people.age <= upper_age
        if by_sex:
            f_uids = (within_age & people.female).uids
            m_uids = (within_age & people.male).uids
            return f_uids, m_uids
        else:
            uids = within_age.uids
            return uids

    def set_risk_groups(self, upper_age=None):
        ppl = self.sim.people
        uids = self._get_uids(upper_age=upper_age, by_sex=False)

        p_lo = np.full(len(uids), fill_value=np.nan, dtype=ss_float_)
        p_lo[ppl.female[uids]] = self.pars.prop_f0
        p_lo[ppl.male[uids]] = self.pars.prop_m0
        self.pars.p_lo_risk.set(p=p_lo)
        lo_risk, hi_med_risk = self.pars.p_lo_risk.split(uids)

        p_hi = np.full(len(hi_med_risk), fill_value=np.nan, dtype=ss_float_)
        p_hi[ppl.female[hi_med_risk]] = self.pars.prop_f2 / (1 - self.pars.prop_f0)
        p_hi[ppl.male[hi_med_risk]] = self.pars.prop_m2 / (1 - self.pars.prop_m0)
        self.pars.p_hi_risk.set(p=p_hi)
        hi_risk, med_risk = self.pars.p_hi_risk.split(hi_med_risk)

        self.risk_group[lo_risk] = 0
        self.risk_group[med_risk] = 1
        self.risk_group[hi_risk] = 2
        return

    def set_concurrency(self, upper_age=None):
        people = self.sim.people
        if upper_age is None:
            upper_age = 1000
        in_age_lim = people.age < upper_age
        uids = in_age_lim.uids

        lam = np.full(uids.shape, fill_value=np.nan, dtype=ss_float_)
        for rg in range(self.pars.n_risk_groups):
            f_conc = self.pars[f"f{rg}_conc"]
            m_conc = self.pars[f"m{rg}_conc"]
            in_risk_group = self.risk_group == rg
            in_group = in_risk_group & in_age_lim
            f_in = (people.female & in_group)[uids]
            m_in = (people.male & in_group)[uids]
            if f_in.any():
                lam[f_in] = f_conc
            if m_in.any():
                lam[m_in] = m_conc

        self.pars.concurrency_dist.set(lam=lam)
        self.concurrency[uids] = self.pars.concurrency_dist.rvs(uids) + 1
        return

    def set_sex_work(self, upper_age=None):
        f_uids, m_uids = self._get_uids(upper_age=upper_age)
        self.fsw[f_uids] = self.pars.fsw_shares.rvs(f_uids)
        self.client[m_uids] = self.pars.client_shares.rvs(m_uids)
        return

    def set_debut(self, upper_age=None):
        uids = self._get_uids(upper_age=upper_age, by_sex=False)
        par1 = np.full(len(uids), fill_value=np.nan, dtype=ss_float_)
        par2 = np.full(len(uids), fill_value=np.nan, dtype=ss_float_)
        par1[self.sim.people.female[uids]] = self.pars.debut_pars_f[0]
        par2[self.sim.people.female[uids]] = self.pars.debut_pars_f[1]
        par1[self.sim.people.male[uids]] = self.pars.debut_pars_m[0]
        par2[self.sim.people.male[uids]] = self.pars.debut_pars_m[1]
        self.pars.debut.set(mean=par1, std=par2)
        self.debut[uids] = self.pars.debut.rvs(uids)
        return

    def match_pairs(self):
        ppl = self.sim.people

        active = self.over_debut
        underpartnered = self.partners < self.concurrency
        f_eligible = active & ppl.female & underpartnered
        m_eligible = active & ppl.male & underpartnered
        f_looking = self.pars.p_pair_form.filter(f_eligible.uids)

        if len(f_looking) == 0 or m_eligible.count() == 0:
            raise NoPartnersFound()

        loc, scale = self.get_age_risk_pars(f_looking, self.pars.age_diff_pars)
        self.pars.age_diffs.set(loc=loc, scale=scale)
        age_gaps = self.pars.age_diffs.rvs(f_looking)
        desired_ages = ppl.age[f_looking] + age_gaps
        m_ages = ppl.age[m_eligible]
        ind_m = np.argsort(m_ages, stable=True)
        ind_f = np.argsort(desired_ages, stable=True)

        if len(ind_m) == 0 or len(ind_f) == 0:
            raise NoPartnersFound()

        youngest_preferred_male_age = desired_ages[ind_f[0]]
        youngest_male_age = m_ages[ind_m[0]]

        if youngest_male_age < youngest_preferred_male_age:
            cutoff_index = bisect_left(m_ages[ind_m], youngest_preferred_male_age)
            ind_m = ind_m[cutoff_index:]
        elif youngest_preferred_male_age < youngest_male_age:
            cutoff_index = bisect_left(desired_ages[ind_f], youngest_male_age)
            ind_f = ind_f[cutoff_index:]

        if len(ind_m) == 0 or len(ind_f) == 0:
            raise NoPartnersFound()

        oldest_preferred_male_age = desired_ages[ind_f[-1]]
        oldest_male_age = m_ages[ind_m[-1]]

        if oldest_male_age > oldest_preferred_male_age:
            cutoff_index = bisect_left(m_ages[ind_m], oldest_preferred_male_age)
            ind_m = ind_m[:cutoff_index]
        elif oldest_preferred_male_age > oldest_male_age:
            cutoff_index = bisect_left(desired_ages[ind_f], oldest_male_age)
            ind_f = ind_f[:cutoff_index]

        if len(ind_m) < len(ind_f):
            ind_f_subset = np.random.choice(len(ind_f), size=len(ind_m), replace=False)
            ind_f_subset.sort()
            ind_f = ind_f[ind_f_subset]
        elif len(ind_f) < len(ind_m):
            ind_m_subset = np.random.choice(len(ind_m), size=len(ind_f), replace=False)
            ind_m_subset.sort()
            ind_m = ind_m[ind_m_subset]

        if len(ind_m) == 0 or len(ind_f) == 0:
            raise NoPartnersFound()

        p1 = m_eligible.uids[ind_m]
        p2 = f_looking[ind_f]

        return p1, p2

    def add_pairs_sw(self):
        ppl = self.sim.people

        try:
            p1, p2 = self.match_sex_workers()
        except NoPartnersFound:
            return

        match_count = len(p1)
        beta = np.ones(match_count, dtype=ss_float_)
        acts = (self.pars.acts.rvs(p2)).astype(int)
        dur = np.full(match_count, fill_value=1)
        age_p1 = ppl.age[p1]
        age_p2 = ppl.age[p2]
        edge_types = np.full(match_count, dtype=ss_float_, fill_value=self.edge_types["sw"])

        self.append(
            p1=p1,
            p2=p2,
            beta=beta,
            dur=dur,
            acts=acts,
            sw=[True] * match_count,
            age_p1=age_p1,
            age_p2=age_p2,
            edge_type=edge_types,
        )

        p1_edges, p1_counts = np.unique(p1, return_counts=True)
        p2_edges, p2_counts = np.unique(p2, return_counts=True)

        self.lifetime_sw_partners[p1_edges] += p1_counts
        self.lifetime_sw_partners[p2_edges] += p2_counts
        return

    def add_pairs_nonsw(self):
        ppl = self.sim.people

        try:
            p1, p2 = self.match_pairs()
        except NoPartnersFound:
            return

        matched_risk = self.risk_group[p1] == self.risk_group[p2]
        mismatched_risk = self.risk_group[p1] != self.risk_group[p2]

        p_match = np.full(len(p1), fill_value=np.nan, dtype=ss_float_)
        for rg in range(self.pars.n_risk_groups):
            p_match[matched_risk & (self.risk_group[p1] == rg)] = self.pars.p_matched_stable[rg]
            p_match[mismatched_risk & (self.risk_group[p2] == rg)] = self.pars.p_mismatched_casual[rg]
        self.pars.match_dist.set(p=p_match)
        matches = self.pars.match_dist.rvs(p2)

        stable = matches & matched_risk
        casual = matches & mismatched_risk
        any_match = stable | casual

        match_count = len(p1)

        beta = np.ones(match_count, dtype=ss_float_)
        acts = (self.pars.acts.rvs(p2)).astype(int)
        dur = np.full(match_count, fill_value=1)
        age_p1 = ppl.age[p1]
        age_p2 = ppl.age[p2]
        edge_types = np.full(match_count, dtype=ss_float_, fill_value=np.nan)
        edge_types[stable] = self.edge_types["stable"]
        edge_types[casual] = self.edge_types["casual"]

        dur_mean = np.full(match_count, fill_value=np.nan, dtype=ss_float_)
        dur_std = np.full(match_count, fill_value=np.nan, dtype=ss_float_)
        for which, bools in {"stable": stable, "casual": casual}.items():
            if bools.any():
                uids = p2[bools]
                thesepars = self.pars[f"{which}_dur_pars"]
                mean, std = self.get_age_risk_pars(uids, thesepars)
                dur_mean[bools] = mean
                dur_std[bools] = std
        self.pars.dur_dist.set(mean=dur_mean[any_match], std=dur_std[any_match])
        dur[any_match] = self.pars.dur_dist.rvs(p2[any_match])

        edge_types[(dur == 1)] = self.edge_types["onetime"]

        relationships = dur > 1
        for a, b, reldur, etype in zip(p1[relationships], p2[relationships], dur[relationships], edge_types[relationships]):
            pair = (min(a, b), max(a, b))
            self.relationship_durs[pair].append({"start": self.ti, "dur": reldur, "edge_type": int(etype)})

        self.append(
            p1=p1,
            p2=p2,
            beta=beta,
            dur=dur,
            acts=acts,
            sw=[False] * match_count,
            age_p1=age_p1,
            age_p2=age_p2,
            edge_type=edge_types,
        )

        if (self.sim.people.female[p1].any() or self.sim.people.male[p2].any()) and (self.name == "structuredsexual"):
            errormsg = "Same-sex pairings should not be possible in this network"
            raise ValueError(errormsg)
        if len(p1) != len(p2):
            errormsg = "Unequal lengths in edge list"
            raise ValueError(errormsg)

        for key, edge_type in self.edge_types.items():
            p1_edges = p1[edge_types == edge_type]
            p2_edges = p2[edge_types == edge_type]
            self.partners[p1_edges] += 1
            self.partners[p2_edges] += 1
            self.lifetime_partners[p1_edges] += 1
            self.lifetime_partners[p2_edges] += 1
            getattr(self, f"{key}_partners")[p1_edges] += 1
            getattr(self, f"{key}_partners")[p2_edges] += 1
            getattr(self, f"lifetime_{key}_partners")[p1_edges] += 1
            getattr(self, f"lifetime_{key}_partners")[p2_edges] += 1
        return

    def add_pairs(self):
        self.add_pairs_nonsw()
        self.add_pairs_sw()
        return

    def match_sex_workers(self):
        active = self.over_debut
        active_fsw = active & self.fsw
        active_clients = active & self.client
        self.sw_intensity[active_fsw.uids] = self.pars.sw_intensity.rvs(active_fsw.uids)

        self.pars.sw_seeking_dist.pars.p = self.pars.sw_seeking_rate.to_prob()
        m_looking = self.pars.sw_seeking_dist.filter(active_clients.uids)

        if len(m_looking) == 0 or len(active_fsw.uids) == 0:
            raise NoPartnersFound()

        if len(m_looking) > len(active_fsw.uids):
            n_repeats = (self.sw_intensity[active_fsw] * 10).astype(int) + 1
            fsw_repeats = np.repeat(active_fsw.uids, n_repeats)
            if len(fsw_repeats) < len(m_looking):
                fsw_repeats = np.repeat(fsw_repeats, 10)

            n_pairs = min(len(fsw_repeats), len(m_looking))
            if len(fsw_repeats) < len(m_looking):
                p1 = m_looking[:n_pairs]
                p2 = fsw_repeats
            else:
                unique_sw, counts_sw = np.unique(fsw_repeats, return_counts=True)
                count_repeats = np.repeat(counts_sw, counts_sw)
                weights = self.sw_intensity[fsw_repeats] / count_repeats
                choices = np.argsort(-weights)[:n_pairs]
                p2 = fsw_repeats[choices]
                p1 = m_looking
        else:
            n_pairs = len(m_looking)
            weights = self.sw_intensity[active_fsw]
            choices = np.argsort(-weights)[:n_pairs]
            p2 = active_fsw.uids[choices]
            p1 = m_looking

        return p1, p2

    def end_pairs(self):
        people = self.sim.people

        self.edges.dur = self.edges.dur - 1

        alive_bools = people.alive[ss.uids(self.edges.p1)] & people.alive[ss.uids(self.edges.p2)]
        active = (self.edges.dur > 0) & alive_bools

        if self.pars.recall_prior:
            prior_network = self.sim.networks.get("priorpartners")
            if prior_network is not None:
                ended_p1 = self.edges.p1[~active]
                ended_p2 = self.edges.p2[~active]
                durs = np.zeros_like(ended_p1, dtype=ss_float_)
                betas = np.zeros_like(ended_p1, dtype=ss_float_)
                prior_network.append(p1=ended_p1, p2=ended_p2, dur=durs, beta=betas)

        inactive_gp = ~active & (~self.edges.sw)

        p1_edges = self.edges.p1[inactive_gp]
        p2_edges = self.edges.p2[inactive_gp]
        edge_types = self.edges.edge_type[inactive_gp]
        onetimes = edge_types == self.edge_types["onetime"]
        casuals = edge_types == self.edge_types["casual"]
        stables = edge_types == self.edge_types["stable"]
        sw = edge_types == self.edge_types["sw"]

        self.partners[p1_edges] -= 1
        self.partners[p2_edges] -= 1

        self.onetime_partners[p1_edges[onetimes]] -= 1
        self.onetime_partners[p2_edges[onetimes]] -= 1
        self.casual_partners[p1_edges[casuals]] -= 1
        self.casual_partners[p2_edges[casuals]] -= 1
        self.stable_partners[p1_edges[stables]] -= 1
        self.stable_partners[p2_edges[stables]] -= 1
        self.sw_partners[p1_edges[sw]] -= 1
        self.sw_partners[p2_edges[sw]] -= 1

        if len(active) > 0:
            for k in self.meta_keys():
                self.edges[k] = self.edges[k][active]
        return

    def count_partners(self):
        self.lifetime_partners
        return

    def step(self):
        self.end_pairs()
        self.set_network_states(upper_age=self.t.dt_year)
        self.add_pairs()
        self.count_partners()
        return


class StructuredCondomUse(CondomUse):
    """CondomUse with DataFrame/dict condom_data and disease.pars.eff_condom."""

    def __init__(self, condom_data=None, name="condomuse", **kwargs):
        if sc.isnumber(condom_data) or condom_data is None:
            super().__init__(condom_data=condom_data, name=name, **kwargs)
        else:
            super().__init__(condom_data=None, name=name, **kwargs)
            self.condom_data = self._process_condom_data(condom_data)

    @staticmethod
    def _process_condom_data(condom_data):
        if sc.isnumber(condom_data):
            return condom_data
        elif isinstance(condom_data, pd.DataFrame):
            df = condom_data
            if "variable" not in df.columns or "value" not in df.columns:
                df = df.melt(id_vars=["partnership"])
            dd = dict()
            for pcombo in df.partnership.unique():
                key = tuple(map(int, pcombo[1:-1].split(","))) if pcombo != "(fsw,client)" else ("fsw", "client")
                thisdf = df.loc[df.partnership == pcombo]
                dd[key] = dict()
                dd[key]["year"] = thisdf.variable.values.astype(int)
                dd[key]["val"] = thisdf.value.values
            return dd
        return condom_data

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.condom_data is not None and isinstance(self.condom_data, dict):
            for rgtuple, valdict in self.condom_data.items():
                yearvec = self.t.yearvec
                self.condom_data[rgtuple]["simvals"] = sc.smoothinterp(yearvec, valdict["year"], valdict["val"])

    def _set_condom_probabilities(self, net):
        if isinstance(self.condom_data, dict):
            for rgm in range(net.pars.n_risk_groups):
                for rgf in range(net.pars.n_risk_groups):
                    risk_pairing = (net.risk_group[net.p1] == rgm) & (net.risk_group[net.p2] == rgf)
                    net.edges.condoms[risk_pairing] = self.condom_data[(rgm, rgf)]["simvals"][net.ti]
            net.edges.condoms[net.edges.sw] = self.condom_data[("fsw", "client")]["simvals"][net.ti]
        elif sc.isnumber(self.condom_data):
            net.edges.condoms[:] = self.condom_data

    def relative_risk(self, network, disease_beta, disease, uids=None, direction=None):
        return 1 - disease.pars.eff_condom
