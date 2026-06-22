"""stisim-compatible edge interface: EdgeStructuredSexual, StructuredCondomUse."""

import sciris as sc
import numpy as np
import stisim as sti
import pandas as pd

from .core import EdgeNetwork, CondomUse


class EdgeStructuredSexual(EdgeNetwork, sti.StructuredSexual):
    """stisim's StructuredSexual with per-act, per-edge intervention support.

    Subclasses :class:`stisim.StructuredSexual` (risk groups, concurrency,
    age-assortative pairing, sex work) and mixes in :class:`enroute.EdgeNetwork`'s
    intervention-aware ``net_beta`` so per-act interventions (condoms, doxy-PEP,
    PrEP) stack multiplicatively. All network behavior tracks upstream stisim,
    including the default matcher and the age-window sex-work model.

    Condom use is deliberately *not* handled by the network here: attach a
    :class:`StructuredCondomUse` intervention instead. Passing ``condom_data``
    raises, since the inherited network condom column is ignored by
    ``EdgeNetwork.net_beta`` (condoms are netted through the intervention's
    per-act ``_uses`` columns).
    """

    def __init__(self, pars=None, name="structuredsexual", **kwargs):
        has_condom_data = kwargs.get("condom_data") is not None or (isinstance(pars, dict) and pars.get("condom_data") is not None)
        if has_condom_data:
            errormsg = "EdgeStructuredSexual does not handle condoms directly; attach a StructuredCondomUse intervention instead of passing condom_data."
            raise ValueError(errormsg)
        # super() resolves through EdgeNetwork (no __init__) to sti.StructuredSexual
        super().__init__(pars=pars, name=name, **kwargs)
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
            if "sw" in net.edge_types:
                sw_mask = net.edges.edge_type == net.edge_types["sw"]
                net.edges.condoms[sw_mask] = self.condom_data[("fsw", "client")]["simvals"][net.ti]
        elif sc.isnumber(self.condom_data):
            net.edges.condoms[:] = self.condom_data

    def relative_risk(self, network, disease_beta, disease, uids=None, direction=None):
        return 1 - disease.pars.eff_condom
