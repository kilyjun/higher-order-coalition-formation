"""
visualize_toy.py — Visualizations for the 3-agent complete-graph toy example.

Produces 5 figures:
  1. Network diagram with skill vectors and coalition synergies
  2. Effort allocation comparison (NE vs SO) — grouped bar chart
  3. Parameter sweeps (rho, tau, lam, beta) — 4-panel line plots
  4. Breakthrough probability decomposition (NE vs SO)
  5. 2D heatmap: rho x tau for market failure gap and generalist disadvantage

Usage:
    python visualize_toy.py                # all figures
    python visualize_toy.py --plot network
    python visualize_toy.py --plot effort
    python visualize_toy.py --plot sweeps
    python visualize_toy.py --plot breakthrough
    python visualize_toy.py --plot heatmap
    python visualize_toy.py --save         # save to figures/
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from toy_example import analyze, create_agents, build_all_coalitions, coalition_label, AGENT_LABELS

# ── Style ────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "font.family": "serif",
})

C_S1 = "#2980b9"
C_S2 = "#27ae60"
C_G  = "#e74c3c"
C_NE = "#34495e"
C_SO = "#8e44ad"
C_GAP = "#e67e22"

COAL_COLORS = {
    frozenset([0]):       C_S1,
    frozenset([1]):       C_S2,
    frozenset([2]):       C_G,
    frozenset([0, 1]):    "#95a5a6",
    frozenset([0, 2]):    "#d4a017",
    frozenset([1, 2]):    "#1abc9c",
    frozenset([0, 1, 2]): "#8e44ad",
}


def _save(fig, name, save):
    if save:
        os.makedirs("figures", exist_ok=True)
        path = os.path.join("figures", name)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  Saved: {path}")


# =====================================================================
# Figure 1: Network diagram + skill vectors
# =====================================================================

def plot_network(save=False):
    agents = create_agents()
    coalitions = build_all_coalitions(agents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: Skill space ---
    ax1.set_xlim(-0.15, 1.15)
    ax1.set_ylim(-0.15, 1.15)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Skill dimension 1")
    ax1.set_ylabel("Skill dimension 2")
    ax1.set_title("Skill Vectors on Unit Circle")

    # Draw unit circle arc
    theta_arc = np.linspace(0, np.pi / 2, 100)
    ax1.plot(np.cos(theta_arc), np.sin(theta_arc), "k-", alpha=0.15, linewidth=1)

    colors = [C_S1, C_S2, C_G]
    labels = ["S1: (1, 0)", "S2: (0, 1)", "G: (0.71, 0.71)"]
    for a, col, lab in zip(agents, colors, labels):
        v = a.skill_vector
        ax1.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax1.plot(v[0], v[1], "o", color=col, markersize=10, zorder=5)
        offset = (0.04, 0.04)
        if a.agent_id == 0:
            offset = (0.04, -0.08)
        elif a.agent_id == 1:
            offset = (-0.12, 0.04)
        ax1.annotate(lab, xy=(v[0], v[1]),
                     xytext=(v[0] + offset[0], v[1] + offset[1]),
                     fontsize=10, color=col, fontweight="bold")

    ax1.axhline(0, color="gray", linewidth=0.3)
    ax1.axvline(0, color="gray", linewidth=0.3)
    ax1.grid(True, alpha=0.15)

    # --- Right panel: Network graph ---
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.3, 1.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Complete Network (All Links Active)")

    # Node positions (triangle)
    pos = {0: (-1.0, -0.7), 1: (1.0, -0.7), 2: (0.0, 1.0)}
    node_colors = {0: C_S1, 1: C_S2, 2: C_G}
    node_labels = {0: "S1\n(1, 0)", 1: "S2\n(0, 1)", 2: "G\n(.71, .71)"}

    # Edges with similarity labels
    for i in range(3):
        for j in range(i + 1, 3):
            sim = agents[i].similarity(agents[j])
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]
            ax2.plot(x, y, "-", color="gray", linewidth=2, alpha=0.5, zorder=1)
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            # Offset label slightly
            dx, dy = y[1] - y[0], -(x[1] - x[0])
            norm = max(np.sqrt(dx**2 + dy**2), 1e-9)
            dx, dy = dx / norm * 0.15, dy / norm * 0.15
            ax2.text(mx + dx, my + dy, f"sim={sim:.2f}",
                     ha="center", va="center", fontsize=9, color="gray",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # Nodes
    for aid in [0, 1, 2]:
        x, y = pos[aid]
        circle = plt.Circle((x, y), 0.3, color=node_colors[aid], alpha=0.2, zorder=2)
        ax2.add_patch(circle)
        ax2.plot(x, y, "o", color=node_colors[aid], markersize=20, zorder=3)
        ax2.text(x, y - 0.55, node_labels[aid], ha="center", va="top",
                 fontsize=10, fontweight="bold", color=node_colors[aid])

    # Coalition synergy table below
    rho = 0.5
    coalitions_multi = [c for c in coalitions if c.size >= 2]
    table_y = -1.15
    ax2.text(0, table_y, "Coalition synergies (rho=0.5):", ha="center",
             fontsize=9, fontstyle="italic", color="gray")
    entries = []
    for c in coalitions_multi:
        label = coalition_label(c.key)
        entries.append(f"{label}: G={c.synergy(rho):.3f}")
    ax2.text(0, table_y - 0.2, "   ".join(entries), ha="center", fontsize=8, color="gray")

    fig.suptitle("Toy Example: 3-Agent Setup", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "toy_network.png", save)
    plt.show()


# =====================================================================
# Figure 2: Effort allocation NE vs SO
# =====================================================================

def plot_effort(save=False, rho=0.5, tau=1.0, lam=0.0, beta=0.0):
    res = analyze(rho=rho, tau=tau, lam=lam, beta=beta, verbose=False)

    agents = create_agents()
    coalitions = build_all_coalitions(agents)
    coal_labels = [coalition_label(c.key) for c in coalitions]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (aid, agent_name) in zip(axes, AGENT_LABELS.items()):
        ne_vals = []
        so_vals = []
        colors = []
        labels_used = []
        for c in coalitions:
            if aid in c.member_ids:
                ne_vals.append(res["efforts_ne"][aid].get(c.key, 0.0))
                so_vals.append(res["efforts_so"][aid].get(c.key, 0.0))
                colors.append(COAL_COLORS[c.key])
                labels_used.append(coalition_label(c.key))

        x = np.arange(len(labels_used))
        w = 0.35
        bars_ne = ax.bar(x - w/2, ne_vals, w, color=colors, alpha=0.6,
                         edgecolor="black", linewidth=0.5, label="NE")
        bars_so = ax.bar(x + w/2, so_vals, w, color=colors, alpha=1.0,
                         edgecolor="black", linewidth=0.5, label="SO",
                         hatch="///")

        ax.set_xticks(x)
        ax.set_xticklabels(labels_used, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Effort r_{iC}")
        ax.set_title(f"{agent_name.strip()}")
        ax.grid(True, alpha=0.2, axis="y")

        # Add total effort annotation
        total_ne = sum(ne_vals)
        total_so = sum(so_vals)
        ax.text(0.95, 0.95, f"R_NE={total_ne:.3f}\nR_SO={total_so:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

    # Legend
    ne_patch = mpatches.Patch(facecolor="gray", alpha=0.6, label="Nash Eq.")
    so_patch = mpatches.Patch(facecolor="gray", alpha=1.0, hatch="///", label="Social Opt.")
    axes[1].legend(handles=[ne_patch, so_patch], loc="upper center", fontsize=10)

    fig.suptitle(f"Effort Allocation: NE vs Social Optimum  "
                 f"(rho={rho}, tau={tau}, lam={lam})", fontsize=13)
    plt.tight_layout()
    _save(fig, "toy_effort.png", save)
    plt.show()


# =====================================================================
# Figure 3: Parameter sweeps (4 panels)
# =====================================================================

def plot_sweeps(save=False):
    sweeps = [
        ("rho", np.linspace(0, 1, 21), "Breadth-depth weight (rho)"),
        ("tau", np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
         "Threshold (tau)"),
        ("lam", np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
         "Friction (lambda)"),
        ("beta", np.linspace(0, 1, 11), "Group-size penalty (beta)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (pname, vals, xlabel) in zip(axes, sweeps):
        gaps = []
        u_diffs = []
        w_nes = []
        w_sos = []

        defaults = dict(rho=0.5, alpha=0.5, beta=0.0, lam=0.0, tau=1.0)
        for v in vals:
            kwargs = dict(defaults)
            kwargs[pname] = v
            res = analyze(verbose=False, **kwargs)
            gaps.append(res["gap"])
            u_diffs.append(res["payoffs_ne"][2] - res["payoffs_ne"][0])
            w_nes.append(res["w_ne"])
            w_sos.append(res["w_so"])

        ax2 = ax.twinx()

        l1, = ax.plot(vals, gaps, "o-", color=C_GAP, linewidth=2, markersize=4,
                      label="Market failure gap (W_SO - W_NE)")
        l2, = ax2.plot(vals, u_diffs, "s--", color=C_G, linewidth=2, markersize=4,
                       label="Generalist disadvantage (u_G - u_S1)")

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Market failure gap", color=C_GAP)
        ax2.set_ylabel("u_G - u_S1", color=C_G)
        ax.tick_params(axis="y", labelcolor=C_GAP)
        ax2.tick_params(axis="y", labelcolor=C_G)

        lines = [l1, l2]
        ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Toy Example: Parameter Sweeps", fontsize=14)
    plt.tight_layout()
    _save(fig, "toy_sweeps.png", save)
    plt.show()


# =====================================================================
# Figure 4: Breakthrough probability decomposition
# =====================================================================

def plot_breakthrough(save=False, rho=0.5, tau=1.0, lam=0.0, beta=0.0):
    res = analyze(rho=rho, tau=tau, lam=lam, beta=beta, verbose=False)

    agents = create_agents()
    coalitions = build_all_coalitions(agents)

    coal_labels = [coalition_label(c.key) for c in coalitions]
    ne_probs = [res["ne_outputs"][c.key]["prob"] for c in coalitions]
    so_probs = [res["so_outputs"][c.key]["prob"] for c in coalitions]
    colors = [COAL_COLORS[c.key] for c in coalitions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # --- Left: bar chart comparison ---
    x = np.arange(len(coal_labels))
    w = 0.35
    ax1.bar(x - w/2, ne_probs, w, color=colors, alpha=0.6,
            edgecolor="black", linewidth=0.5)
    ax1.bar(x + w/2, so_probs, w, color=colors, alpha=1.0,
            edgecolor="black", linewidth=0.5, hatch="///")
    ax1.set_xticks(x)
    ax1.set_xticklabels(coal_labels, rotation=35, ha="right", fontsize=9)
    ax1.set_ylabel("P(breakthrough)")
    ax1.set_title("Breakthrough Probability by Coalition")
    ax1.grid(True, alpha=0.2, axis="y")

    ne_patch = mpatches.Patch(facecolor="gray", alpha=0.6, label="Nash Eq.")
    so_patch = mpatches.Patch(facecolor="gray", alpha=1.0, hatch="///", label="Social Opt.")
    ax1.legend(handles=[ne_patch, so_patch], fontsize=10)

    # --- Right: stacked waterfall showing where SO gains come from ---
    # Difference in P(BT) per coalition: SO - NE
    diffs = np.array(so_probs) - np.array(ne_probs)
    # Sort by magnitude for clarity
    order = np.argsort(-np.abs(diffs))
    sorted_labels = [coal_labels[i] for i in order]
    sorted_diffs = diffs[order]
    sorted_colors = [colors[i] for i in order]

    cumulative = np.cumsum(sorted_diffs)
    ax2.bar(range(len(sorted_labels)), sorted_diffs, color=sorted_colors,
            edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.plot(range(len(sorted_labels)), cumulative, "k-o", markersize=5,
             linewidth=1.5, label="Cumulative gain")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xticks(range(len(sorted_labels)))
    ax2.set_xticklabels(sorted_labels, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Change in P(BT): SO - NE")
    ax2.set_title("Where Does the Planner Gain?")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"Breakthrough Decomposition  (rho={rho}, tau={tau}, lam={lam})",
                 fontsize=13)
    plt.tight_layout()
    _save(fig, "toy_breakthrough.png", save)
    plt.show()


# =====================================================================
# Figure 5: rho x tau heatmap
# =====================================================================

def plot_heatmap(save=False):
    rhos = np.linspace(0, 1, 21)
    taus = np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])

    gap_matrix = np.zeros((len(taus), len(rhos)))
    udiff_matrix = np.zeros((len(taus), len(rhos)))

    for i, tau in enumerate(taus):
        for j, rho in enumerate(rhos):
            res = analyze(rho=rho, tau=tau, verbose=False)
            gap_matrix[i, j] = res["gap"]
            udiff_matrix[i, j] = res["payoffs_ne"][2] - res["payoffs_ne"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Market failure gap ---
    im1 = ax1.imshow(gap_matrix, cmap="YlOrRd", aspect="auto", origin="lower",
                     extent=[rhos[0], rhos[-1], 0, len(taus) - 1])
    ax1.set_yticks(range(len(taus)))
    ax1.set_yticklabels([f"{t:.2f}" for t in taus])
    ax1.set_xlabel("rho (breadth-depth)")
    ax1.set_ylabel("tau (threshold)")
    ax1.set_title("Market Failure Gap (W_SO - W_NE)")
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cb1.set_label("Gap")

    # Contour lines
    ax1.contour(np.linspace(0, len(rhos) - 1, len(rhos)),
                np.arange(len(taus)),
                gap_matrix, levels=5, colors="black", linewidths=0.5, alpha=0.4)

    # --- Generalist payoff disadvantage ---
    vmax = max(abs(udiff_matrix.min()), abs(udiff_matrix.max()))
    im2 = ax2.imshow(udiff_matrix, cmap="RdBu_r", aspect="auto", origin="lower",
                     vmin=-vmax, vmax=vmax,
                     extent=[rhos[0], rhos[-1], 0, len(taus) - 1])
    ax2.set_yticks(range(len(taus)))
    ax2.set_yticklabels([f"{t:.2f}" for t in taus])
    ax2.set_xlabel("rho (breadth-depth)")
    ax2.set_ylabel("tau (threshold)")
    ax2.set_title("Generalist Disadvantage (u_G - u_S1)")
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cb2.set_label("u_G - u_S1  (blue = disadvantage)")

    # Zero contour
    ax2.contour(np.linspace(0, len(rhos) - 1, len(rhos)),
                np.arange(len(taus)),
                udiff_matrix, levels=[0], colors="black", linewidths=2)

    fig.suptitle("Toy Example: rho x tau Landscape (complete graph, no friction)",
                 fontsize=14)
    plt.tight_layout()
    _save(fig, "toy_heatmap.png", save)
    plt.show()


# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Toy Example Visualization")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["network", "effort", "sweeps", "breakthrough",
                                 "heatmap", "all"])
    # Note: "heatmap" is disabled from "all" for performance. Run it explicitly.
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    plot_map = {
        "network":      ("Network diagram", lambda: plot_network(args.save)),
        "effort":       ("Effort allocation", lambda: plot_effort(
                             args.save, args.rho, args.tau, args.lam, args.beta)),
        "sweeps":       ("Parameter sweeps", lambda: plot_sweeps(args.save)),
        "breakthrough": ("Breakthrough decomposition", lambda: plot_breakthrough(
                             args.save, args.rho, args.tau, args.lam, args.beta)),
        # "heatmap" excluded from "all" — too slow. Run with --plot heatmap explicitly.
        # "heatmap":      ("rho x tau heatmap", lambda: plot_heatmap(args.save)),
    }

    if args.plot == "heatmap":
        print("\n--- rho x tau heatmap ---")
        plot_heatmap(args.save)
        print("\nDone.")
        return

    targets = list(plot_map.keys()) if args.plot == "all" else [args.plot]

    for name in targets:
        label, func = plot_map[name]
        print(f"\n--- {label} ---")
        func()

    print("\nDone.")


if __name__ == "__main__":
    main()
