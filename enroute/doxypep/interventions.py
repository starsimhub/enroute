"""
Doxycycline Post-Exposure Prophylaxis (Doxy-PEP) interventions for STIsim.

Two intervention classes:
    DoxyPEPEnrollment: Manages enrollment, eligibility, and pill dispensing.
    DoxyPEP: EdgeIntervention that reduces per-act transmission risk.

One analyzer:
    DoxyPEPAnalyzer: Tracks prescriptions, doses consumed, enrollment counts.
"""

import starsim as ss
import numpy as np
from ..core import EdgeIntervention

ss_float_ = ss.dtypes.float
ss_int_ = ss.dtypes.int


class DoxyPEPEnrollment(ss.Intervention):
    """
    Manages the clinical side of Doxy-PEP: enrollment, eligibility, and pill dispensing.

    Each timestep: update eligibility, un-enroll agents who lost eligibility
    (they keep remaining doses), enroll new eligible agents, and refill those
    below threshold.

    Args:
        eligibility:       Callable(sim) -> uids of eligible agents.
        enroll_prob:       Per-timestep probability of enrolling an eligible, non-enrolled agent.
        prescribed_doses:  Number of doses added on enrollment or refill.
        refill_threshold:  Agents with doses below this get a refill opportunity.
        refill_prob:       Probability of refilling when below threshold.
        start_year:        Year (float) when enrollment begins; no enrollment before this.
    """

    def __init__(
        self,
        eligibility=None,
        enroll_prob=0.1,
        prescribed_doses=30,
        refill_threshold=5,
        refill_prob=0.8,
        start_year=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_eligibility = eligibility
        self.start_year = start_year
        self.define_pars(
            enroll_prob=enroll_prob,
            prescribed_doses=prescribed_doses,
            refill_threshold=refill_threshold,
            refill_prob=refill_prob,
        )

        self.enroll_dist = ss.bernoulli(p=0)
        self.refill_dist = ss.bernoulli(p=0)

        self.define_states(
            ss.BoolArr("enrolled", default=False),
            ss.BoolArr("eligible", default=False),
            ss.BoolArr("ever_enrolled", default=False),
            ss.FloatArr("doses", default=0),
            ss.FloatArr("ti_enrolled"),
            ss.FloatArr("ti_unenrolled"),
        )

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("n_enrolled", dtype=int, label="Currently enrolled"),
            ss.Result("n_eligible", dtype=int, label="Currently eligible"),
            ss.Result("new_enrollments", dtype=int, label="New enrollments"),
            ss.Result("new_unenrollments", dtype=int, label="Unenrollments"),
            ss.Result("new_refills", dtype=int, label="Prescription refills"),
            ss.Result("total_doses_dispensed", dtype=int, label="Doses dispensed"),
        )

    def step(self):
        sim = self.sim
        ti = sim.ti

        # Skip enrollment before start year
        if self.start_year is not None and sim.t.yearvec[ti] < self.start_year:
            return

        # Update eligibility
        if self.default_eligibility is not None:
            eligible_uids = self.default_eligibility(sim)
            if isinstance(eligible_uids, ss.BoolArr):
                eligible_uids = eligible_uids.uids
            self.eligible[:] = False
            self.eligible[eligible_uids] = True
        else:
            # Everyone alive is eligible if no function provided
            self.eligible[:] = True

        # Unenroll agents who lost eligibility
        to_unenroll = (self.enrolled & ~self.eligible).uids
        if len(to_unenroll):
            self.enrolled[to_unenroll] = False
            self.ti_unenrolled[to_unenroll] = ti

        # Enroll new eligible agents
        candidates = (self.eligible & ~self.enrolled).uids
        if len(candidates):
            self.enroll_dist.set(p=self.pars.enroll_prob)
            new_enrolled = self.enroll_dist.filter(candidates)
            if len(new_enrolled):
                self.enrolled[new_enrolled] = True
                self.ever_enrolled[new_enrolled] = True
                self.ti_enrolled[new_enrolled] = ti
                self.doses[new_enrolled] += self.pars.prescribed_doses

        # Refill enrolled agents below threshold
        self._n_refills_this_step = 0
        needs_refill = (self.enrolled & (self.doses < self.pars.refill_threshold)).uids
        if len(needs_refill):
            self.refill_dist.set(p=self.pars.refill_prob)
            refilled = self.refill_dist.filter(needs_refill)
            if len(refilled):
                self.doses[refilled] += self.pars.prescribed_doses
                self._n_refills_this_step = len(refilled)

        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results["n_enrolled"][ti] = np.count_nonzero(self.enrolled.raw)
        self.results["n_eligible"][ti] = np.count_nonzero(self.eligible.raw)
        self.results["new_enrollments"][ti] = np.count_nonzero(self.ti_enrolled.raw == ti)
        self.results["new_unenrollments"][ti] = np.count_nonzero(self.ti_unenrolled.raw == ti)

        new_enrolled_count = np.count_nonzero(self.ti_enrolled.raw == ti)
        n_refills = getattr(self, "_n_refills_this_step", 0)
        self.results["new_refills"][ti] = n_refills
        self.results["total_doses_dispensed"][ti] = (new_enrolled_count + n_refills) * self.pars.prescribed_doses
        return


class DoxyPEP(EdgeIntervention):
    """
    Per-act edge-level intervention that reduces STI transmission risk when
    an agent with Doxy-PEP doses takes a pill after condomless sex.

    Reads enrollment and dose state from a DoxyPEPEnrollment intervention.
    Operates on edges in an EdgeStructuredSexual network. Each timestep,
    computes unprotected acts per edge, draws Bernoulli uses at the given
    uptake rate, deducts doses from whichever partner is enrolled (preferring
    p1 if both are), and stores uses in the edge column for net_beta().

    Args:
        uptake:      Probability of taking doxy-PEP per unprotected act.
                     Can be a float or dict keyed by edge_type name.
        rel_risk:    Dict of disease-specific relative risks.
                     Keys are disease names (e.g. 'ct', 'ng', 'syph').
        enrollment:  Name or instance of DoxyPEPEnrollment intervention.
                     Auto-detected if only one exists.
    """

    def __init__(
        self,
        uptake=0.6,
        rel_risk=None,
        enrollment=None,
        **kwargs,
    ):
        super().__init__(asymmetric=True, **kwargs)
        self._enrollment_ref = enrollment

        if rel_risk is None:
            rel_risk = dict(ct=0.12, syph=0.13, ng=0.45)
        self.rel_risk = rel_risk

        self.uptake = uptake

    def init_pre(self, sim):
        super().init_pre(sim)

        if self._enrollment_ref is None:
            enrollments = [iv for iv in sim.interventions.values() if isinstance(iv, DoxyPEPEnrollment)]
            if len(enrollments) == 0:
                raise ValueError("DoxyPEP requires a DoxyPEPEnrollment intervention in the simulation.")
            if len(enrollments) > 1:
                raise ValueError("Multiple DoxyPEPEnrollment found. Pass enrollment=<name> to disambiguate.")
            self._enrollment = enrollments[0]
        elif isinstance(self._enrollment_ref, str):
            self._enrollment = sim.interventions[self._enrollment_ref]
        else:
            self._enrollment = self._enrollment_ref
        return

    def _get_uptake_per_edge(self, network):
        """Return per-edge uptake probability array."""
        n_edges = len(network.edges.p1)
        if isinstance(self.uptake, dict):
            prob = np.zeros(n_edges, dtype=ss_float_)
            for type_name, type_val in network.edge_types.items():
                mask = network.edges.edge_type == type_val
                if type_name in self.uptake:
                    prob[mask] = self.uptake[type_name]
            return prob
        else:
            return np.full(n_edges, fill_value=self.uptake, dtype=ss_float_)

    def _deduct_doses(self, network, uses, agent_col):
        """
        Deduct doses for agents in a given edge column (p1 or p2),
        capping uses so no agent exceeds their available doses.

        Caller must zero uses for agents without doses before calling.

        Args:
            network: The EdgeStructuredSexual network.
            uses:    Mutable array of per-edge uses (modified in-place).
            agent_col: 'p1' or 'p2' — which side of the edge to process.

        Returns:
            None (modifies uses in-place and deducts from enrollment.doses).
        """
        enrollment = self._enrollment
        agents = network.edges[agent_col]
        has_doses = enrollment.doses[ss.uids(agents)] > 0

        if not has_doses.any():
            return

        # Sum uses per agent
        agent_uids = agents[has_doses]
        edge_uses = uses[has_doses]

        unique_agents, inverse = np.unique(agent_uids, return_inverse=True)
        total_per_agent = np.bincount(inverse, weights=edge_uses).astype(ss_int_)

        # Cap at available doses
        available = np.round(enrollment.doses[ss.uids(unique_agents)]).astype(ss_int_)
        excess = total_per_agent - available
        needs_cap = excess > 0

        if needs_cap.any():
            # For agents that need capping, scale down proportionally
            for i in np.where(needs_cap)[0]:
                agent_uid = unique_agents[i]
                agent_edges = np.where((agents == agent_uid) & has_doses)[0]
                edge_vals = uses[agent_edges]
                total = edge_vals.sum()
                if total > 0:
                    avail = int(available[i])
                    # Scale proportionally, rounding down
                    scaled = np.floor(edge_vals * (avail / total)).astype(ss_int_)
                    # Distribute remainder to first edges
                    remainder = avail - scaled.sum()
                    for j in range(int(remainder)):
                        scaled[j] += 1
                    uses[agent_edges] = scaled
            # Recompute totals after capping
            edge_uses = uses[has_doses]
            total_per_agent = np.bincount(inverse, weights=edge_uses).astype(ss_int_)

        uids = ss.uids(unique_agents)
        enrollment.doses[uids] -= total_per_agent
        enrollment.doses[uids] = np.maximum(enrollment.doses[uids], 0)  # clamp against float drift
        return

    def update_uses(self, network):
        """Compute per-edge doxy-PEP protected act counts.

        Each partner independently decides whether to take doxy-PEP on each
        unprotected act (one Bernoulli draw per partner per act, executed as a
        single RNG call).  Doses are deducted from each partner's own supply.
        The edge is protected on any act where at least one partner took it.
        """
        n_edges = len(network.edges.p1)
        if n_edges == 0:
            self.edge_uses = np.empty(0, dtype=ss_int_)
            return

        acts = network.edges.acts

        prior_uses = np.zeros_like(acts)
        for iv in network._edge_intvs:
            if iv is not self and iv.edge_col in network.meta:
                prior_uses += network.edges[iv.edge_col]
        unprotected = np.maximum(acts - prior_uses, 0)

        uptake_prob = self._get_uptake_per_edge(network)

        # Independent per-partner Bernoulli draws in a single RNG call
        unprotected_int = unprotected.astype(ss_int_)
        total_acts = int(unprotected_int.sum())
        uses_p1 = np.zeros(n_edges, dtype=ss_int_)
        uses_p2 = np.zeros(n_edges, dtype=ss_int_)

        if total_acts > 0:
            has_acts = unprotected_int > 0
            sub_acts = unprotected_int[has_acts]
            sub_prob = uptake_prob[has_acts]
            per_act_prob = np.repeat(sub_prob, sub_acts)

            doubled_prob = np.concatenate([per_act_prob, per_act_prob])
            self.uses_dist.set(p=doubled_prob)
            all_draws = self.uses_dist.rvs(2 * total_acts)
            draws_p1 = all_draws[:total_acts]
            draws_p2 = all_draws[total_acts:]

            offsets = np.concatenate([[0], np.cumsum(sub_acts[:-1])])
            uses_p1[has_acts] = np.add.reduceat(draws_p1.astype(ss_int_), offsets)
            uses_p2[has_acts] = np.add.reduceat(draws_p2.astype(ss_int_), offsets)

        # Zero uses for partners without doses, then deduct
        enrollment = self._enrollment
        p1_agents = np.asarray(network.edges['p1'])
        p2_agents = np.asarray(network.edges['p2'])
        p1_has = enrollment.doses[ss.uids(p1_agents)] > 0
        p2_has = enrollment.doses[ss.uids(p2_agents)] > 0
        uses_p1[~p1_has] = 0
        uses_p2[~p2_has] = 0
        self._deduct_doses(network, uses_p1, "p1")
        self._deduct_doses(network, uses_p2, "p2")

        self.edge_uses_p1 = uses_p1
        self.edge_uses_p2 = uses_p2
        self.edge_uses = np.maximum(uses_p1, uses_p2)  # for prior-uses subtraction and analyzer compat
        return

    def relative_risk(self, network, disease_beta, disease, uids=None, direction=None):
        """Return per-edge relative risk for the given disease."""
        disease_name = disease.name if hasattr(disease, "name") else str(disease)
        return self.rel_risk.get(disease_name, 1.0)


class DoxyPEPAnalyzer(ss.Analyzer):
    """
    Analyzer that tracks Doxy-PEP related outcomes each timestep.

    Results:
        n_enrolled:           Number of agents currently enrolled
        n_eligible:           Number of agents currently eligible
        n_with_doses:         Number of agents with doses > 0
        total_doses_in_pop:   Total doses remaining in the population
        doses_consumed:       Doses consumed this timestep
        total_acts:           Total sex acts across all edges
        condom_protected:     Acts protected by condoms
        doxypep_protected:    Acts protected by Doxy-PEP
        unprotected:          Acts with no protection
    """

    def __init__(self, enrollment=None, **kwargs):
        super().__init__(**kwargs)
        self._enrollment_ref = enrollment
        self._prev_total_doses = 0

    def init_pre(self, sim):
        super().init_pre(sim)

        if self._enrollment_ref is None:
            enrollments = [iv for iv in sim.interventions.values() if isinstance(iv, DoxyPEPEnrollment)]
            if len(enrollments) == 0:
                raise ValueError("DoxyPEPAnalyzer requires a DoxyPEPEnrollment intervention.")
            self._enrollment = enrollments[0]
        elif isinstance(self._enrollment_ref, str):
            self._enrollment = sim.interventions[self._enrollment_ref]
        else:
            self._enrollment = self._enrollment_ref

        # Cache network reference for step()
        self._network = None
        for n in sim.networks.values():
            if hasattr(n, "edges") and hasattr(n.edges, "acts"):
                self._network = n
                break
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result("n_enrolled", dtype=int, label="Enrolled in Doxy-PEP"),
            ss.Result("n_eligible", dtype=int, label="Eligible for Doxy-PEP"),
            ss.Result("n_with_doses", dtype=int, label="Agents with doses"),
            ss.Result("total_doses_in_pop", dtype=int, label="Total doses in population"),
            ss.Result("doses_consumed", dtype=int, label="Doses consumed"),
            ss.Result("total_acts", dtype=int, label="Total acts", auto_plot=False),
            ss.Result("condom_protected", dtype=int, label="Condom-protected acts", auto_plot=False),
            ss.Result("doxypep_protected", dtype=int, label="Doxy-PEP-protected acts", auto_plot=False),
            ss.Result("unprotected", dtype=int, label="Unprotected acts", auto_plot=False),
        )

    def step(self):
        enrollment = self._enrollment
        sim = self.sim
        ti = sim.ti

        total_doses = int(enrollment.doses.raw[sim.people.alive.raw].sum())

        self.results["n_enrolled"][ti] = np.count_nonzero(enrollment.enrolled.raw)
        self.results["n_eligible"][ti] = np.count_nonzero(enrollment.eligible.raw)
        self.results["n_with_doses"][ti] = np.count_nonzero(enrollment.doses.raw > 0)
        self.results["total_doses_in_pop"][ti] = total_doses

        if ti > 0:
            self.results["doses_consumed"][ti] = max(0, self._prev_total_doses - total_doses)
        self._prev_total_doses = total_doses

        net = self._network
        if net is not None and len(net.edges.p1) > 0:
            acts = net.edges.acts

            # Discover protection columns from registered edge interventions
            condom_sum = 0
            doxypep_sum = 0
            for iv in getattr(net, "_edge_intvs", []):
                col = iv.edge_col
                if col in net.meta:
                    col_sum = int(net.edges[col].sum())
                    if isinstance(iv, DoxyPEP):
                        doxypep_sum += col_sum
                    else:
                        condom_sum += col_sum

            acts_sum = int(acts.sum())
            self.results["total_acts"][ti] = acts_sum
            self.results["condom_protected"][ti] = condom_sum
            self.results["doxypep_protected"][ti] = doxypep_sum
            self.results["unprotected"][ti] = max(0, acts_sum - condom_sum - doxypep_sum)
        return
