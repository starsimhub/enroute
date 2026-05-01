"""Analyzers for edge-network simulations. CondomTracker records per-timestep condom state for plotting."""

import starsim as ss
import sciris as sc
import numpy as np

from .core import EdgeNetwork

ss_float_ = ss.dtypes.float


class CondomTracker(ss.Analyzer):
    """Record per-timestep condom probability, act counts, and edge-age snapshots.

    Stores data in self.data (sc.objdict of lists), one entry per timestep.
    Designed for post-hoc extraction and plotting via the plotting module.

    Args:
        network (str/EdgeNetwork): network to track; None auto-discovers the single EdgeNetwork
    """

    def __init__(self, network=None, uses_col=None, name="condomtracker", **kwargs):
        super().__init__(name=name, **kwargs)
        self._network_ref = network
        self._net = None
        self._uses_col_ref = uses_col  # explicit override; None means auto-discover
        self._uses_col = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)

        # resolve network reference (same guard pattern as EdgeIntervention)
        if self._network_ref is None:
            edge_nets = [n for n in sim.networks.values() if isinstance(n, EdgeNetwork)]
            if len(edge_nets) == 0:
                raise ValueError(f"{self.__class__.__name__} requires at least one EdgeNetwork in the simulation.")
            if len(edge_nets) > 1:
                raise ValueError(f"{self.__class__.__name__} found {len(edge_nets)} EdgeNetworks but no network was specified. Pass network=<name_or_instance> to disambiguate.")
            self._net = edge_nets[0]
        elif isinstance(self._network_ref, str):
            self._net = sim.networks[self._network_ref]
        else:
            self._net = self._network_ref

        # discover the uses column: explicit override, or auto-discover from network interventions
        if self._uses_col_ref is not None:
            self._uses_col = self._uses_col_ref
        else:
            self._uses_col = self._discover_uses_col()

        etype_names = list(self._net.edge_types.keys()) if hasattr(self._net, "edge_types") else []

        self.data = sc.objdict(
            ti=[], t=[], n_edges=[],
            total_acts=[], total_uses=[], mean_condom=[],
            condom_by_type=sc.objdict({et: [] for et in etype_names}),
            acts_by_type=sc.objdict({et: [] for et in etype_names}),
            uses_by_type=sc.objdict({et: [] for et in etype_names}),
            rel_ages_by_type=sc.objdict({et: [] for et in etype_names}),
            rel_condoms_by_type=sc.objdict({et: [] for et in etype_names}),
        )
        return

    def _discover_uses_col(self):
        """Find the uses column from the network's edge interventions. Returns None if none found."""
        if not hasattr(self._net, "_edge_intvs") or not self._net._edge_intvs:
            return None
        # Prefer an intervention with "condom" in the column name; fall back to the first
        for iv in self._net._edge_intvs:
            if "condom" in iv.edge_col.lower():
                return iv.edge_col
        return self._net._edge_intvs[0].edge_col

    def step(self):
        """Snapshot condom/act/edge-age data for the current timestep."""
        net = self._net
        ti = self.sim.ti

        # Lazy re-discovery: if _uses_col was None at init but interventions registered later
        if self._uses_col is None and self._uses_col_ref is None:
            self._uses_col = self._discover_uses_col()

        acts = np.asarray(net.edges.acts)
        condoms = np.asarray(net.edges.condoms) if "condoms" in net.meta else np.zeros(len(acts), dtype=ss_float_)
        uses = np.asarray(net.edges[self._uses_col]) if (self._uses_col and self._uses_col in net.meta) else np.zeros(len(acts), dtype=int)
        etypes = np.asarray(net.edges.edge_type) if "edge_type" in net.meta else np.zeros(len(acts), dtype=int)

        total_acts = int(acts.sum())
        total_uses = int(uses.sum())

        if total_acts > 0:
            mean_condom = float(np.dot(condoms, acts) / total_acts)
        elif len(condoms) > 0:
            mean_condom = float(condoms.mean())
        else:
            mean_condom = 0.0

        self.data.ti.append(ti)
        self.data.t.append(float(self.sim.t.yearvec[ti]))
        self.data.n_edges.append(len(acts))
        self.data.total_acts.append(total_acts)
        self.data.total_uses.append(total_uses)
        self.data.mean_condom.append(mean_condom)

        has_start_ti = "start_ti" in net.meta
        if has_start_ti:
            start_ti = np.asarray(net.edges["start_ti"])
            ages_months = (ti - start_ti).astype(float) * self.sim.t.dt_year * 12

        etype_map = net.edge_types if hasattr(net, "edge_types") else {}
        for etype_name, etype_code in etype_map.items():
            if etype_name not in self.data.condom_by_type:
                continue
            mask = etypes == etype_code
            type_acts = int(acts[mask].sum())
            type_uses = int(uses[mask].sum())
            if type_acts > 0:
                type_mean = float(np.dot(condoms[mask], acts[mask]) / type_acts)
            elif mask.any():
                type_mean = float(condoms[mask].mean())
            else:
                type_mean = 0.0

            self.data.condom_by_type[etype_name].append(type_mean)
            self.data.acts_by_type[etype_name].append(type_acts)
            self.data.uses_by_type[etype_name].append(type_uses)

            if has_start_ti:
                self.data.rel_ages_by_type[etype_name].append(ages_months[mask].copy())
                self.data.rel_condoms_by_type[etype_name].append(condoms[mask].copy())
            else:
                self.data.rel_ages_by_type[etype_name].append(np.array([]))
                self.data.rel_condoms_by_type[etype_name].append(np.array([]))
        return
