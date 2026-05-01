"""Reusable plotting utilities for edge-network condom interventions.

Primitives: seed_band, seed_stats, stacked_area, savefig, style.
Domain plots: prevalence comparison, condom prob by edge type, acts protection,
              decay curve, network graph, Beta preference distributions.
Compound summaries: plot_structured_summary, plot_behavioral_summary.
"""

import numpy as np
import sciris as sc
import starsim as ss
import pylab as pl
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist

from .core import EdgeNetwork

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

EDGE_TYPE_COLORS = {"stable": "#2166ac", "casual": "#4dac26", "onetime": "#f1a340", "sw": "#d6604d"}
EDGE_TYPE_LABELS = {"stable": "Stable", "casual": "Casual", "onetime": "One-time", "sw": "Sex work"}
RG_COLORS = {0: "#2166ac", 1: "#4dac26", 2: "#d6604d"}  # risk group palette
RG_LABELS = {0: "Low risk", 1: "Medium risk", 2: "High risk"}


# %% Primitives

def style():
    """Return ss.style context manager with project-standard font."""
    return ss.style(font="Raleway", fontsize=12)


def seed_stats(arr_2d):
    """Compute mean, min, max across seed axis.

    Args:
        arr_2d (ndarray): shape (n_seeds, n_timepoints)

    Returns:
        tuple: (mean, lo, hi) each shape (n_timepoints,)
    """
    arr_2d = np.asarray(arr_2d, dtype=float)
    mean = arr_2d.mean(axis=0)
    lo = arr_2d.min(axis=0)
    hi = arr_2d.max(axis=0)
    return mean, lo, hi


def seed_band(ax, tvec, mean, lo, hi, label=None, color=None, alpha=0.12, **kw):
    """Draw a mean line with a shaded min/max band.

    Args:
        ax    (Axes):  target axes
        tvec  (array): x values
        mean  (array): central line
        lo    (array): lower bound
        hi    (array): upper bound
        label (str):   legend label
        color (str):   matplotlib color
        alpha (float): fill opacity

    Returns:
        line handle
    """
    line, = ax.plot(tvec, mean, color=color, label=label, **kw)
    ax.fill_between(tvec, lo, hi, color=line.get_color(), alpha=alpha)
    return line


def stacked_area(ax, x, layers, labels=None, colors=None, alpha=0.7):
    """Stacked fill_between from y=0.

    Args:
        ax     (Axes):        target axes
        x      (array):       x values
        layers (list[array]): height of each layer (not cumulative)
        labels (list[str]):   one per layer
        colors (list[str]):   one per layer
        alpha  (float):       fill opacity

    Returns:
        list of PolyCollection handles
    """
    bottom = np.zeros_like(x, dtype=float)
    handles = []
    for i, layer in enumerate(layers):
        top = bottom + np.asarray(layer, dtype=float)
        c = colors[i] if colors else None
        lab = labels[i] if labels else None
        h = ax.fill_between(x, bottom, top, color=c, alpha=alpha, label=lab)
        handles.append(h)
        bottom = top
    return handles


def savefig(fig, name, path="examples"):
    """Save figure to path/name.png at 150 dpi.

    Args:
        fig  (Figure): matplotlib figure
        name (str):    filename without extension
        path (str):    directory; created if absent
    """
    sc.path(path).mkdir(parents=True, exist_ok=True)
    fig.savefig(sc.path(path) / f"{name}.png", dpi=150, bbox_inches="tight")
    return


# %% Domain plots

def plot_prevalence_comparison(scenarios, disease_key="ct", ax=None, title=None):
    """Multi-seed prevalence comparison across scenarios.

    Args:
        scenarios   (dict): label -> list[Sim] (one Sim per seed)
        disease_key (str):  disease name ("ct", "ng", etc.)
        ax          (Axes): existing axes; None creates a figure
        title       (str):  axes title

    Returns:
        Figure
    """
    with style():
        if ax is None:
            fig, ax = pl.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        colors = pl.cm.tab10(np.arange(max(len(scenarios), 1)) % 10)
        for i, (label, sims) in enumerate(scenarios.items()):
            tvec = np.array(sims[0].t.yearvec)
            prev_mat = np.array([np.array(s.diseases[disease_key].results.prevalence) for s in sims])
            mean, lo, hi = seed_stats(prev_mat)
            seed_band(ax, tvec, mean, lo, hi, label=label, color=colors[i], linewidth=2)

        ax.set_xlabel("Year")
        ax.set_ylabel("Prevalence")
        ax.set_title(title or f"{disease_key.upper()} prevalence")
        ax.legend(loc="best")
        fig.tight_layout()
    return fig


def plot_condom_prob_by_type(trackers, title=None):
    """Per-edge-type condom probability time series with seed bands.

    Args:
        trackers (list): CondomTracker objects (one per seed)
        title    (str):  figure suptitle

    Returns:
        Figure (2x2 grid)
    """
    with style():
        fig, axes = pl.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        etype_order = ["stable", "casual", "onetime", "sw"]

        for etype_name, ax in zip(etype_order, axes.flat):
            tvec = np.array(trackers[0].data.t)
            series = np.array([np.array(tr.data.condom_by_type[etype_name]) for tr in trackers])
            mean, lo, hi = seed_stats(series)
            seed_band(ax, tvec, mean, lo, hi, color=EDGE_TYPE_COLORS[etype_name], linewidth=2)
            ax.set_title(EDGE_TYPE_LABELS[etype_name])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Condom probability")
            ax.set_xlabel("Year")

        fig.suptitle(title or "Condom probability by edge type", fontsize=14)
        fig.tight_layout()
    return fig


def plot_acts_protection(trackers, ax=None, title=None):
    """Stacked area: condom-protected vs unprotected acts over time.

    Args:
        trackers (list): CondomTracker objects (one per seed)
        ax       (Axes): existing axes; None creates a figure
        title    (str):  axes title

    Returns:
        Figure
    """
    with style():
        if ax is None:
            fig, ax = pl.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        tvec = np.array(trackers[0].data.t)
        acts_mat = np.array([np.array(tr.data.total_acts) for tr in trackers])
        uses_mat = np.array([np.array(tr.data.total_uses) for tr in trackers])
        mean_acts = acts_mat.mean(axis=0)
        mean_uses = uses_mat.mean(axis=0)
        unprotected = np.maximum(0, mean_acts - mean_uses)

        stacked_area(ax, tvec, [mean_uses, unprotected],
                     labels=["Condom-protected", "Unprotected"],
                     colors=["#2166ac", "#d6604d"], alpha=0.75)
        ax.plot(tvec, mean_acts, "--k", linewidth=1.5, alpha=0.6, label="Total acts")
        ax.set_xlabel("Year")
        ax.set_ylabel("Acts per timestep")
        ax.set_title(title or "Act-level protection over time")
        ax.legend(loc="best")
        fig.tight_layout()
    return fig


def plot_decay_curve(trackers, etype="stable", ax=None, title=None):
    """Scatter of condom probability vs relationship age, with fitted decay.

    Only meaningful when start_ti is tracked (BehavioralCondomUse / ExtendedCondomUse with decay).

    Args:
        trackers (list): CondomTracker objects (one per seed)
        etype    (str):  edge type to show
        ax       (Axes): existing axes; None creates a figure
        title    (str):  axes title

    Returns:
        Figure
    """
    # pool final-timestep data across seeds
    all_ages = []
    all_probs = []
    for tr in trackers:
        ages_list = tr.data.rel_ages_by_type.get(etype, [])
        probs_list = tr.data.rel_condoms_by_type.get(etype, [])
        if ages_list and len(ages_list[-1]) > 0:
            all_ages.append(ages_list[-1])
            all_probs.append(probs_list[-1])

    if not all_ages:
        sc.warn(f"plot_decay_curve: no relationship-age data available for edge type '{etype}'")
        return None

    ages = np.concatenate(all_ages)
    probs = np.concatenate(all_probs)

    with style():
        if ax is None:
            fig, ax = pl.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        ax.scatter(ages, probs, alpha=0.15, s=12, color=EDGE_TYPE_COLORS.get(etype, "C0"), rasterized=True)

        # fit exponential decay: p(t) = floor + (p0 - floor) * exp(-lambda * t)
        def _decay(t, p0, floor, lam):
            return floor + (p0 - floor) * np.exp(-lam * t)

        try:
            popt, _ = curve_fit(_decay, ages, probs, p0=[0.5, 0.1, 0.04], bounds=([0, 0, 0], [1, 1, 10]), maxfev=5000)
            t_fit = np.linspace(0, ages.max(), 200)
            ax.plot(t_fit, _decay(t_fit, *popt), "k-", linewidth=2, label="Fitted decay")
            half_life = np.log(2) / popt[2] if popt[2] > 0 else float("inf")
            ax.annotate(f"Half-life: {half_life:.1f} months\nFloor: {popt[1]:.2f}", xy=(0.65, 0.85), xycoords="axes fraction", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        except RuntimeError:
            sc.warn("plot_decay_curve: exponential fit did not converge")

        ax.set_xlabel("Relationship age (months)")
        ax.set_ylabel("Condom probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title or f"Condom decay — {EDGE_TYPE_LABELS.get(etype, etype)} partnerships")
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="best")
        fig.tight_layout()
    return fig


def _packed_layout(G, seed=42, pad=0.08):
    """Lay out each connected component separately and shelf-pack them tightly.

    Spring layout with disconnected components pushes them to the boundary,
    leaving empty space in the centre. This function instead:
      1. Lays out each component with spring_layout (tight k).
      2. Scales each component proportional to sqrt(n_nodes).
      3. Shelf-packs left-to-right; starts a new row when the shelf is full.

    Args:
        G   (nx.Graph): the graph
        seed (int):     RNG seed for spring_layout reproducibility
        pad  (float):   gap between components (in layout units)

    Returns:
        dict: node -> (x, y) positions
    """
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    if not components:
        return {}

    # pre-compute each component's local layout and bounding size
    layouts = []  # list of (sub_pos_normalised, width, height, comp)
    for idx, comp in enumerate(components):
        subG = G.subgraph(comp)
        n = len(comp)
        k = 0.8 / np.sqrt(max(n, 2))
        sub_pos = nx.spring_layout(subG, seed=seed + idx, k=k, iterations=50)

        # scale proportional to sqrt(n_nodes) so big components get more room
        target_size = np.sqrt(n) * 0.15
        xs = np.array([sub_pos[node][0] for node in sub_pos])
        ys = np.array([sub_pos[node][1] for node in sub_pos])
        x_range = xs.max() - xs.min() if len(xs) > 1 else 1.0
        y_range = ys.max() - ys.min() if len(ys) > 1 else 1.0
        scale = max(x_range, y_range) if max(x_range, y_range) > 0 else 1.0

        normed = {}
        for node in sub_pos:
            normed[node] = ((sub_pos[node][0] - xs.min()) / scale * target_size,
                            (sub_pos[node][1] - ys.min()) / scale * target_size)

        layouts.append((normed, target_size, target_size, comp))

    # shelf-pack: fill rows left-to-right, wrap when exceeding target width
    total_area = sum(w * h for _, w, h, _ in layouts)
    shelf_width = np.sqrt(total_area) * 1.8  # aim for roughly square output

    pos = {}
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0

    for normed, w, h, comp in layouts:
        if cursor_x + w > shelf_width and cursor_x > 0:
            # start new row
            cursor_y -= row_height + pad
            cursor_x = 0.0
            row_height = 0.0

        for node in normed:
            pos[node] = (normed[node][0] + cursor_x, normed[node][1] + cursor_y)

        cursor_x += w + pad
        row_height = max(row_height, h)

    return pos


def plot_network_graph(sim, ax=None, title=None):
    """Network graph with edges colored by condom probability.

    Requires networkx (soft dependency). Returns None if unavailable.

    Args:
        sim   (Sim):  completed sim
        ax    (Axes): existing axes; None creates a figure
        title (str):  axes title

    Returns:
        Figure, or None if networkx unavailable
    """
    if not _HAS_NX:
        sc.warn("plot_network_graph requires networkx — install it for network visualizations")
        return None

    net = None
    for n in sim.networks.values():
        if isinstance(n, EdgeNetwork):
            net = n
            break
    if net is None:
        sc.warn("plot_network_graph: no EdgeNetwork found in sim")
        return None

    p1 = np.asarray(net.edges.p1)
    p2 = np.asarray(net.edges.p2)
    condoms = np.asarray(net.edges.condoms) if "condoms" in net.meta else np.zeros(len(p1))
    etypes = np.asarray(net.edges.edge_type) if "edge_type" in net.meta else np.zeros(len(p1), dtype=int)

    etype_styles = {"stable": "-", "casual": "--", "onetime": ":", "sw": "-."}
    etype_map = net.edge_types if hasattr(net, "edge_types") else {}
    if not etype_map:
        sc.warn("plot_network_graph: network has no edge_types; all edges will render as 'stable' style")
    reverse_etype = {v: k for k, v in etype_map.items()}

    G = nx.Graph()
    edge_data = []
    for i in range(len(p1)):
        u, v = int(p1[i]), int(p2[i])
        G.add_edge(u, v)
        edge_data.append((u, v, float(condoms[i]), int(etypes[i])))

    if len(G.nodes) == 0:
        sc.warn("plot_network_graph: network has no edges at this timestep")
        return None

    with style():
        if ax is None:
            fig, ax = pl.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        # lay out each connected component separately, then pack into a grid
        pos = _packed_layout(G, seed=42)
        cmap = pl.cm.RdYlGn

        # draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color="#888888", alpha=0.7)

        # draw edges grouped by type for different linestyles
        for etype_code in set(int(e[3]) for e in edge_data):
            etype_name = reverse_etype.get(etype_code, "stable")
            ls = etype_styles.get(etype_name, "-")
            subset = [(u, v, c) for u, v, c, et in edge_data if et == etype_code]
            if not subset:
                continue
            edges_list = [(u, v) for u, v, _ in subset]
            edge_colors = [cmap(c) for _, _, c in subset]
            nx.draw_networkx_edges(G, pos, edgelist=edges_list, edge_color=edge_colors, style=ls, alpha=0.8, width=1.8, ax=ax)

        ax.set_axis_off()

        # colorbar
        sm = pl.cm.ScalarMappable(cmap=cmap, norm=pl.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.6, label="Condom probability")

        # edge-type legend
        legend_handles = [Line2D([0], [0], color="gray", linestyle=etype_styles[et], linewidth=1.5, label=EDGE_TYPE_LABELS[et])
                          for et in etype_styles if et in etype_map]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

        ax.set_title(title or "Sexual network — condom protection", fontsize=13)
        fig.tight_layout()
    return fig


def plot_beta_preference_dist(beta_params, ax=None, title=None):
    """Beta preference PDF curves by risk group and sex.

    Args:
        beta_params (dict): (risk_group, 'f'|'m') -> (a, b)
        ax          (Axes): existing axes; None creates a figure
        title       (str):  axes title

    Returns:
        Figure
    """
    with style():
        if ax is None:
            fig, ax = pl.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        x = np.linspace(0.001, 0.999, 300)
        sex_styles = {"f": "-", "m": "--"}
        sex_labels = {"f": "Female", "m": "Male"}

        for (rg, sex), (a, b) in sorted(beta_params.items()):
            pdf = beta_dist.pdf(x, a, b)
            color = RG_COLORS.get(rg, "C0")
            ls = sex_styles.get(sex, "-")
            label = f"{RG_LABELS.get(rg, f'RG{rg}')} — {sex_labels.get(sex, sex)}"
            ax.plot(x, pdf, color=color, linestyle=ls, linewidth=1.8, label=label)

        ax.set_xlabel("Condom preference")
        ax.set_ylabel("Density")
        ax.set_title(title or "Per-agent condom preference distributions")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
    return fig


# %% Compound summaries

def plot_structured_summary(scenarios, trackers_by_scenario, disease_key="ct"):
    """4-panel summary for StructuredCondomUse runs.

    Panels: prevalence comparison | condom prob by type (first tracked scenario) |
            acts protection | scenario labels.

    Args:
        scenarios            (dict): label -> list[Sim]
        trackers_by_scenario (dict): label -> list[CondomTracker]
        disease_key          (str):  "ct" or "ng"

    Returns:
        Figure (2x2)
    """
    # find first scenario with trackers
    tracked_label = None
    for label, trs in trackers_by_scenario.items():
        if trs:
            tracked_label = label
            break

    with style():
        fig = pl.figure(figsize=(14, 10))

        # panel 1: prevalence comparison
        ax1 = fig.add_subplot(2, 2, 1)
        plot_prevalence_comparison(scenarios, disease_key=disease_key, ax=ax1)

        # panel 2: condom prob by type (inline 2x2 would be awkward, use a single summary line per type)
        ax2 = fig.add_subplot(2, 2, 2)
        if tracked_label:
            trs = trackers_by_scenario[tracked_label]
            tvec = np.array(trs[0].data.t)
            for etype_name in ["stable", "casual", "onetime", "sw"]:
                series = np.array([np.array(tr.data.condom_by_type[etype_name]) for tr in trs])
                mean, lo, hi = seed_stats(series)
                seed_band(ax2, tvec, mean, lo, hi, label=EDGE_TYPE_LABELS[etype_name], color=EDGE_TYPE_COLORS[etype_name], linewidth=1.5)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("Condom probability")
            ax2.set_xlabel("Year")
            ax2.set_title(f"Condom prob by type — {tracked_label}")
            ax2.legend(loc="best", fontsize=9)
        else:
            ax2.text(0.5, 0.5, "No tracker data", ha="center", va="center", transform=ax2.transAxes)
            ax2.set_title("Condom prob by type")

        # panel 3: acts protection
        ax3 = fig.add_subplot(2, 2, 3)
        if tracked_label:
            plot_acts_protection(trackers_by_scenario[tracked_label], ax=ax3)
        else:
            ax3.text(0.5, 0.5, "No tracker data", ha="center", va="center", transform=ax3.transAxes)
            ax3.set_title("Act-level protection")

        # panel 4: scenario summary text
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_axis_off()
        lines = [f"Disease: {disease_key.upper()}", ""]
        for label, sims in scenarios.items():
            prev = np.array([np.array(s.diseases[disease_key].results.prevalence)[-1] for s in sims])
            lines.append(f"{label}: prev = {prev.mean():.3f}")
        ax4.text(0.1, 0.7, "\n".join(lines), transform=ax4.transAxes, fontsize=11, verticalalignment="top", family="monospace")
        ax4.set_title("Scenario summary")

        fig.suptitle(f"StructuredCondomUse — {disease_key.upper()}", fontsize=15)
        fig.tight_layout()
    return fig


def plot_behavioral_summary(trackers, beta_params, decay_etype="stable"):
    """4-panel summary for BehavioralCondomUse runs.

    Panels: Beta preferences | condom prob by type | acts protection | decay curve.

    Args:
        trackers    (list): CondomTracker objects (one per seed)
        beta_params (dict): passed to plot_beta_preference_dist
        decay_etype (str):  edge type for decay panel

    Returns:
        Figure (2x2)
    """
    with style():
        fig, axes = pl.subplots(2, 2, figsize=(14, 10))

        # panel 1: Beta preferences
        plot_beta_preference_dist(beta_params, ax=axes[0, 0])

        # panel 2: condom prob by type (all types on one axes)
        ax2 = axes[0, 1]
        tvec = np.array(trackers[0].data.t)
        for etype_name in ["stable", "casual", "onetime", "sw"]:
            series = np.array([np.array(tr.data.condom_by_type[etype_name]) for tr in trackers])
            mean, lo, hi = seed_stats(series)
            seed_band(ax2, tvec, mean, lo, hi, label=EDGE_TYPE_LABELS[etype_name], color=EDGE_TYPE_COLORS[etype_name], linewidth=1.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Condom probability")
        ax2.set_xlabel("Year")
        ax2.set_title("Condom prob by type")
        ax2.legend(loc="best", fontsize=9)

        # panel 3: acts protection
        plot_acts_protection(trackers, ax=axes[1, 0])

        # panel 4: decay curve
        result = plot_decay_curve(trackers, etype=decay_etype, ax=axes[1, 1])
        if result is None:
            axes[1, 1].text(0.5, 0.5, "No decay data", ha="center", va="center", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Condom decay")

        fig.suptitle("BehavioralCondomUse summary", fontsize=15)
        fig.tight_layout()
    return fig
