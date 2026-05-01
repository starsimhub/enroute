"""
Demo: Doxy-PEP intervention in STIsim.

Runs two scenarios side-by-side:
    1. Baseline: no Doxy-PEP
    2. Doxy-PEP: enrollment + per-act prophylaxis

Compares STI prevalence with and without Doxy-PEP.
"""

import numpy as np
import sciris as sc
import pylab as pl
import starsim as ss
import stisim as sti

sc.options(interactive=False)
from enroute import EdgeStructuredSexual, StructuredCondomUse
from enroute.doxypep import DoxyPEPEnrollment, DoxyPEP, DoxyPEPAnalyzer


def make_sim(use_doxypep=False, n_agents=2_000, dt=1 / 12, start=2000, dur=30, seed=0):
    """Create a simulation with or without Doxy-PEP."""

    # -- Diseases --
    ng = sti.Gonorrhea(
        beta_m2f=0.038,
        rel_beta_f2m=0.5,
        init_prev=0.01,
    )
    ct = sti.Chlamydia(
        beta_m2f=0.05,
        rel_beta_f2m=0.5,
        init_prev=0.04,
    )
    tv = sti.Trichomoniasis(
        beta_m2f=0.02,
        rel_beta_f2m=0.5,
        init_prev=0.05,
    )

    # -- Treatment pipeline --
    def seeking_care(sim):
        ng_care = sim.diseases.ng.symptomatic & (sim.diseases.ng.ti_seeks_care == sim.ti)
        ct_care = sim.diseases.ct.symptomatic & (sim.diseases.ct.ti_seeks_care == sim.ti)
        tv_care = sim.diseases.tv.symptomatic & (sim.diseases.tv.ti_seeks_care == sim.ti)
        return (ng_care | ct_care | tv_care).uids

    ng_tx = sti.GonorrheaTreatment()
    ct_tx = sti.STITreatment(diseases="ct", name="ct_tx", label="CT treatment")
    metro = sti.STITreatment(diseases="tv", name="metro", label="Metronidazole")
    syndromic = sti.SymptomaticTesting(
        diseases=[ng, ct, tv],
        eligibility=seeking_care,
        treatments=[ng_tx, ct_tx, metro],
        disease_treatment_map={"ng": ng_tx, "ct": ct_tx, "tv": metro},
        sens=dict(ng=[0.98, 0.98], ct=[0.98, 0.98], tv=[0.98, 0.98]),
        spec=dict(ng=[0.95, 0.95], ct=[0.95, 0.95], tv=[0.95, 0.95]),
        negative_treatments=[],
    )

    # -- Demographics --
    pregnancy = sti.Pregnancy(fertility_rate=10)
    death = ss.Deaths(death_rate=10)

    # -- Network + condoms --
    network = EdgeStructuredSexual()
    condoms = StructuredCondomUse(condom_data=0.3)

    # -- Interventions --
    interventions = [condoms, syndromic, ng_tx, ct_tx, metro]
    analyzers = []

    if use_doxypep:
        # Eligibility: all alive agents
        def dpep_eligible(sim):
            return sim.people.alive

        enrollment = DoxyPEPEnrollment(
            name="dpep_enrollment",
            eligibility=dpep_eligible,
            enroll_prob=0.05,
            prescribed_doses=30,
            refill_threshold=5,
            refill_prob=0.8,
        )
        doxypep = DoxyPEP(name="doxypep", uptake=0.8)
        analyzer = DoxyPEPAnalyzer(name="dpep_analyzer")

        interventions.extend([enrollment, doxypep])
        analyzers.append(analyzer)

    return ss.Sim(
        label="Doxy-PEP" if use_doxypep else "Baseline",
        rand_seed=seed,
        dt=dt,
        start=start,
        dur=dur,
        n_agents=n_agents,
        diseases=[ng, ct, tv],
        networks=network,
        demographics=[pregnancy, death],
        interventions=interventions,
        analyzers=analyzers,
    )


def plot_prevalence(msim):
    """Plot prevalence with dimmed per-run lines and bold mean, colored by label."""
    diseases = [("ng", "Gonorrhea"), ("ct", "Chlamydia"), ("tv", "Trichomoniasis")]
    scenarios = [("Baseline", "C0"), ("Doxy-PEP", "C1")]

    with ss.style(font="Raleway", fontsize=12):
        fig, axes = pl.subplots(1, len(diseases), figsize=(5 * len(diseases), 5), sharey=True)

        for ax, (dkey, dlabel) in zip(axes, diseases):
            for scenario, color in scenarios:
                sims = [s for s in msim.sims if s.label == scenario]
                prev_all = []
                for sim in sims:
                    prev = sim.results[dkey]["prevalence"].values
                    tvec = sim.results[dkey]["prevalence"].timevec
                    prev_all.append(prev)
                    ax.plot(tvec, prev, color=color, alpha=0.2, lw=0.8)

                prev_mean = np.mean(prev_all, axis=0)
                ax.plot(tvec, prev_mean, color=color, lw=2.5, label=scenario)

            ax.set_title(dlabel)
            ax.set_xlabel("Year")
            ax.legend()

        axes[0].set_ylabel("Prevalence")
        fig.suptitle("Impact of Doxy-PEP on STI Prevalence", fontsize=14)
        fig.tight_layout()
    return fig


def plot_edge_bars(sim):
    """
    Viz 1: Stacked bar chart showing act breakdown at the last timestep.

    For each edge type (stable, casual, onetime, sw), shows the aggregate
    split of acts into condom-protected, doxy-PEP-protected, and unprotected.
    """
    net = sim.networks.structuredsexual
    edges = net.edges

    acts = edges.acts
    condom = edges.get("condomuse_uses", np.zeros_like(acts))
    doxypep = edges.get("doxypep_uses", np.zeros_like(acts))
    unprotected = np.maximum(acts - condom - doxypep, 0)
    edge_type = edges.edge_type

    type_names = {v: k.capitalize() for k, v in net.edge_types.items()}
    categories = sorted(type_names.keys())
    labels = [type_names[c] for c in categories]

    condom_totals = [int(condom[edge_type == c].sum()) for c in categories]
    doxy_totals = [int(doxypep[edge_type == c].sum()) for c in categories]
    unprot_totals = [int(unprotected[edge_type == c].sum()) for c in categories]

    # Add an "All" column
    labels.append("All")
    condom_totals.append(int(condom.sum()))
    doxy_totals.append(int(doxypep.sum()))
    unprot_totals.append(int(unprotected.sum()))

    x = np.arange(len(labels))
    width = 0.6

    with ss.style(font="Raleway", fontsize=12):
        fig, ax = pl.subplots(figsize=(8, 5))
        ax.bar(x, condom_totals, width, label="Condom-protected", color="C0")
        ax.bar(x, doxy_totals, width, bottom=condom_totals, label="Doxy-PEP-protected", color="C1")
        bottoms = np.array(condom_totals) + np.array(doxy_totals)
        ax.bar(x, unprot_totals, width, bottom=bottoms, label="Unprotected", color="C3", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Number of acts")
        ax.set_title("Per-act protection breakdown (last timestep)")
        ax.legend()
        fig.tight_layout()
    return fig


def plot_network_edges(sim, filtered=True):
    """
    Network graph with segmented edges showing protection breakdown.

    Each edge is drawn as a segmented line: blue = condom, orange = doxy-PEP,
    red = unprotected, with segment lengths proportional to act counts.

    Args:
        sim: A completed Sim object.
        filtered: If True, exclude nodes whose only edge is a single stable
            relationship (highlights the interesting part of the network).
            If False, show all nodes/edges.
    """
    import networkx as nx
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors

    net = sim.networks.structuredsexual
    edges = net.edges

    acts = edges.acts
    condom = edges.get("condomuse_uses", np.zeros_like(acts))
    doxypep = edges.get("doxypep_uses", np.zeros_like(acts))

    # Determine which nodes to exclude
    nodes_to_exclude = set()
    if filtered:
        edge_type = edges.edge_type
        stable_type = net.edge_types["stable"]
        p1_arr = np.asarray(edges.p1)
        p2_arr = np.asarray(edges.p2)
        all_uids = np.unique(np.concatenate([p1_arr, p2_arr]))
        for uid in all_uids:
            mask = (p1_arr == uid) | (p2_arr == uid)
            if mask.sum() == 1 and edge_type[mask][0] == stable_type:
                nodes_to_exclude.add(int(uid))

    # Build graph and store per-edge protection data
    G = nx.Graph()
    edge_data = {}  # (p1, p2) -> dict(acts, condom, doxy, unprot)
    for i in range(len(edges.p1)):
        p1, p2 = int(edges.p1[i]), int(edges.p2[i])
        if p1 in nodes_to_exclude or p2 in nodes_to_exclude:
            continue
        a = int(acts[i])
        c = int(condom[i])
        d = int(doxypep[i])
        u = max(a - c - d, 0)
        G.add_edge(p1, p2)
        edge_data[(p1, p2)] = dict(acts=a, condom=c, doxy=d, unprot=u)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) == 0:
        return pl.figure()

    # Node colors: enrolled = orange, not enrolled = gray
    enrollment = sim.interventions.get("dpep_enrollment")
    node_colors = []
    for uid in G.nodes():
        if enrollment is not None and enrollment.enrolled.raw[uid]:
            node_colors.append("C1")
        else:
            node_colors.append("lightgray")

    # Spring layout
    pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(max(len(G.nodes()), 1)))

    # Protection colors
    prot_colors = {
        "condom": mcolors.to_rgba("C0"),
        "doxy": mcolors.to_rgba("C1"),
        "unprot": mcolors.to_rgba("C3"),
    }

    # Build segmented edges as a LineCollection
    segments = []
    seg_colors = []
    seg_widths = []

    for u_node, v_node in G.edges():
        data = edge_data.get((u_node, v_node)) or edge_data.get((v_node, u_node))
        p_start = np.array(pos[u_node])
        p_end = np.array(pos[v_node])
        a = data["acts"]

        if a == 0:
            segments.append([p_start, p_end])
            seg_colors.append((0.7, 0.7, 0.7, 0.5))
            seg_widths.append(3)
            continue

        # Build ordered fractions: condom, doxy, unprotected
        fracs = []
        for key in ("condom", "doxy", "unprot"):
            f = data[key] / a
            if f > 0:
                fracs.append((key, f))

        # Walk along the edge, placing colored segments
        t = 0.0
        for key, f in fracs:
            t_end = t + f
            seg_start = p_start + t * (p_end - p_start)
            seg_end = p_start + t_end * (p_end - p_start)
            segments.append([seg_start, seg_end])
            seg_colors.append(prot_colors[key])
            seg_widths.append(3)
            t = t_end

    label = "filtered" if filtered else "full"
    with ss.style(font="Raleway", fontsize=12):
        fig, ax = pl.subplots(figsize=(10, 10))

        # Draw segmented edges
        lc = LineCollection(segments, colors=seg_colors, linewidths=seg_widths, alpha=0.8)
        ax.add_collection(lc)

        # Draw nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=30,
            node_color=node_colors,
            edgecolors="gray",
            linewidths=0.5,
        )

        # Legend
        legend_elements = [
            Line2D([0], [0], color="C0", lw=2, label="Condom-protected"),
            Line2D([0], [0], color="C1", lw=2, label="Doxy-PEP-protected"),
            Line2D([0], [0], color="C3", lw=2, label="Unprotected"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="C1", markersize=8, label="Enrolled in Doxy-PEP"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray", markersize=8, label="Not enrolled"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_title(f"Sexual network — {label}: edge protection status (last timestep)")
        ax.autoscale_view()
        ax.axis("off")
        fig.tight_layout()
    return fig


def plot_acts_timeseries(sim):
    """
    Viz 3: Stacked area chart of act-level protection over time.
    """
    res = sim.results.dpep_analyzer
    tvec = res.total_acts.timevec

    total = res.total_acts.values
    condom = res.condom_protected.values
    doxypep = res.doxypep_protected.values
    unprotected = res.unprotected.values

    with ss.style(font="Raleway", fontsize=12):
        fig, ax = pl.subplots(figsize=(10, 5))
        ax.stackplot(
            tvec,
            condom,
            doxypep,
            unprotected,
            labels=["Condom-protected", "Doxy-PEP-protected", "Unprotected"],
            colors=["C0", "C1", "C3"],
            alpha=0.8,
        )
        ax.plot(tvec, total, color="black", lw=1.5, ls="--", label="Total acts")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of acts per timestep")
        ax.set_title("Per-act protection over time")
        ax.legend(loc="upper left")
        fig.tight_layout()
    return fig


def run_sims(seeds=range(4), save=None):
    """Run baseline and Doxy-PEP simulations, save to datafile."""
    all_sims = [make_sim(use_doxypep=False, seed=s) for s in seeds] + [make_sim(use_doxypep=True, seed=s) for s in seeds]

    print("Running all scenarios...")
    msim = ss.parallel(*all_sims, verbose=0)

    if save is not None:
        print(f"Saving to {save}...")
        ss.save(save, msim)
    return msim


def load_sims(datafile):
    """Load previously saved simulation results."""
    print(f"Loading from {datafile}...")
    return ss.load(datafile)


def plot_sims(msim):
    """Generate all plots from a completed MultiSim."""
    plot_prevalence(msim)

    dpep_sim = [s for s in msim.sims if s.label == "Doxy-PEP"][0]

    plot_edge_bars(dpep_sim)
    plot_network_edges(dpep_sim, filtered=False)
    plot_acts_timeseries(dpep_sim)

    dpep_sims = [s for s in msim.sims if s.label == "Doxy-PEP"]
    dpep_msim = ss.MultiSim(sims=dpep_sims)
    analyzer_keys = [f"dpep_analyzer.{k}" for k in ["n_enrolled", "n_with_doses", "doses_consumed", "n_eligible"]]
    dpep_msim.plot(key=analyzer_keys, figsize=(12, 8), font="Raleway", fontsize=12)
    return


if __name__ == "__main__":
    load = False
    plot = True
    filename = "results/demo_doxypep.msim"

    if not load:
        msim = run_sims(save=filename)
    else:
        msim = load_sims(filename)

    if plot:
        plot_sims(msim)
