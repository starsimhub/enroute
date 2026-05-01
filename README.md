# enroute

Per-act, per-edge intervention framework for [starsim](https://github.com/starsimhub/starsim) and [stisim](https://github.com/starsimhub/stisim) sexual networks.

## Installation

```bash
# From GitHub
pip install git+https://github.com/starsimhub/edge-networks.git

# Local development
git clone https://github.com/starsimhub/edge-networks.git
cd edge-networks
pip install -e .
```

## Architecture

```
Core package (enroute/)
═══════════════════════
  Starsim-only layer (core.py)                  No stisim dependency
  ──────────────────────────────────────
    ss.SexualNetwork  →  EdgeNetwork            Intervention-aware net_beta()
    ss.Intervention   →  EdgeIntervention       Base class: update_uses + relative_risk
                          └── CondomUse          Scalar condom probability

  stisim layer (sti.py)                         Requires stisim
  ────────────────────────────────────
    EdgeNetwork   →  EdgeStructuredSexual       Risk groups, sex work, age-assortative pairing
    CondomUse     →  StructuredCondomUse        DataFrame/dict condom_data, eff_condom from disease

  Doxy-PEP sub-package (doxypep/)
  ───────────────────────────────
    DoxyPEPEnrollment                           Clinical enrollment, eligibility, pill dispensing
    DoxyPEP (EdgeIntervention)                  Per-act transmission risk reduction
    DoxyPEPAnalyzer                             Program-level outcome tracking

Examples package (enroute_examples/)
════════════════════════════════════
  BehavioralCondomUse                           Agent preferences, negotiation, exponential decay
  demo_*.py                                     Runnable demo scripts
```

## Code navigation

| If you want to... | Look at |
|---|---|
| **Write a per-act intervention** for a plain starsim sim | Subclass `EdgeIntervention` from `enroute.core`. Implement `update_uses()` and `relative_risk()`. See quick-start below. |
| **Write a per-act intervention** that needs risk groups, diagnosis, or structured partnerships (PrEP, doxy-PEP, etc.) | Same base class, but use `EdgeStructuredSexual` from `enroute.sti` as your network. See `enroute/doxypep/` for a full example. |
| **Use the framework with plain starsim** | `EdgeNetwork` + `CondomUse` from `enroute.core`. |
| **Use with stisim** structured sexual networks | `EdgeStructuredSexual` + `StructuredCondomUse` from `enroute.sti`. |
| **Customize condom use** (time-varying, risk-group-specific rates) | `StructuredCondomUse` in `enroute.sti` -- accepts a DataFrame or dict keyed by `(risk_group, risk_group)` tuples. |
| **Add behavioral dynamics** (decay, boosts, negotiation) | `BehavioralCondomUse` in `enroute_examples`. |
| **Understand how multiple interventions stack** | `EdgeNetwork.net_beta()` in `enroute/core.py`. |
| **See a complete intervention package** | `enroute/doxypep/`. |

## Quick-start: writing a custom EdgeIntervention

This example extends `CondomUse` with per-agent inventory tracking. Agents receive condoms from a separate distribution intervention that fills `sim.people.condom_supply`; this class draws per-act usage as usual, then caps by available supply and deducts.

```python
import starsim as ss
import numpy as np
from enroute import CondomUse, EdgeNetwork

class CondomSupply(CondomUse):
    """CondomUse with per-agent inventory tracking."""

    def __init__(self, condom_data=0.5, **kwargs):
        super().__init__(condom_data=condom_data, name='condomsupply', **kwargs)
        self.define_states(ss.FloatArr('condom_supply', default=0))

    def update_uses(self, network):
        super().update_uses(network)

        # Cap by p1's available supply
        p1 = network.edges.p1
        uses = self.edge_uses.copy()
        supply = self.condom_supply.raw

        # Aggregate demand per agent across edges
        demand = np.bincount(p1, weights=uses, minlength=len(supply))

        # Scale down for agents whose demand exceeds supply
        over = demand > supply
        if over.any():
            scale = np.where(over, supply / np.maximum(demand, 1), 1.0)
            uses = np.floor(uses * scale[p1]).astype(int)
            self.edge_uses = uses

        # Deduct consumed condoms
        consumed = np.bincount(p1, weights=self.edge_uses, minlength=len(supply))
        supply -= consumed
        np.maximum(supply, 0, out=supply)

```

The network auto-discovers all `EdgeIntervention` instances at `net_beta()` time. Adding a second intervention (e.g., a prophylaxis) requires no changes -- they stack automatically.
