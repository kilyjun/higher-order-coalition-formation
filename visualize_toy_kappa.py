"""
visualize_toy_kappa.py — Visualizations for the kappa sweep toy example.

Produces 3 figures:
  1. Expected payoffs E[u_S1], E[u_S2], E[u_G] vs kappa
  2. Optimal subsidy sigma*(kappa) showing divergence
  3. Realization probabilities and expected active coalitions vs kappa

Usage:
    python visualize_toy_kappa.py
    python visualize_toy_kappa.py --rho 0.8 --tau 0.25 --beta 0.75
    python visualize_toy_kappa.py --save
    python visualize_toy_kappa.py --plot subsidy
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from toy_kappa_sweep import sweep_kappa

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


def _save(fig, name, save):
    if save:
        os.makedirs("figures", exist_ok=True)
        path = os.path.join("figures", name)
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"  Saved: {path}")


# =====================================================================
# Figure 1: Expected Payoffs vs Kappa
# =====================================================================

def plot_payoffs(data, save=False):
    """E[u_i] vs kappa with generalist penalty shaded."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kappa = data["kappa"]

    # --- Left: payoffs ---
    ax1.plot(kappa, data["E_u_S1"], "-o", color=C_S1, ms=4, label="E[u_S1]")
    ax1.plot(kappa, data["E_u_S2"], "-s", color=C_S2, ms=4, label="E[u_S2]")
    ax1.plot(kappa, data["E_u_G"],  "-^", color=C_G,  ms=4, label="E[u_G]")

    # Shade generalist penalty
    u_S = np.maximum(data["E_u_S1"], data["E_u_S2"])
    penalty_mask = u_S > data["E_u_G"]
    if np.any(penalty_mask):
        ax1.fill_between(kappa, data["E_u_G"], u_S,
                         where=penalty_mask, alpha=0.15, color=C_G,
                         label="Generalist penalty")

    ax1.set_xlabel(r"$\kappa$ (homophily)")
    ax1.set_ylabel("Expected payoff")
    ax1.set_title("Expected Payoffs under NE")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", lw=0.5)

    # --- Right: edge probability ---
    ax2.plot(kappa, data["p"], "-", color=C_NE, lw=2)
    ax2.set_xlabel(r"$\kappa$ (homophily)")
    ax2.set_ylabel(r"$p = (1/\sqrt{2})^{\kappa}$")
    ax2.set_title("S-G Edge Formation Probability")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Mark key thresholds
    for threshold in [0.5, 0.1]:
        kappa_thresh = np.log(threshold) / np.log(1 / np.sqrt(2))
        if kappa_thresh <= kappa[-1]:
            ax2.axhline(threshold, color="gray", ls="--", lw=0.5)
            ax2.axvline(kappa_thresh, color="gray", ls="--", lw=0.5)
            ax2.annotate(f"p={threshold}", (kappa_thresh, threshold),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color="gray")

    fig.suptitle("Effect of Homophily on Agent Payoffs", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "toy_kappa_payoffs.png", save)
    plt.show()


# =====================================================================
# Figure 2: Optimal Subsidy sigma*(kappa)
# =====================================================================

def plot_subsidy(data, save=False):
    """Three classes of subsidy vs kappa: type, formation, output-contingent."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    kappa = data["kappa"]
    sig_type = data["sigma_type"]
    sig_form = np.array(data["sigma_form"])
    sig_out = np.array(data["sigma_output"])

    # Cap inf values for plotting
    cap = np.max(sig_form[np.isfinite(sig_form)]) * 1.5 if np.any(np.isfinite(sig_form)) else 10
    sig_form_plot = np.where(np.isfinite(sig_form), sig_form, cap)
    sig_out_plot = np.where(np.isfinite(sig_out), sig_out, cap * 2)

    # --- Left panel: linear scale ---
    ax1.plot(kappa, sig_type, "-o", color=C_S1, ms=4, lw=2,
             label=r"$\sigma_{\mathrm{type}}$ (lump-sum)")
    ax1.plot(kappa, sig_form_plot, "-s", color=C_GAP, ms=4, lw=2,
             label=r"$\sigma_{\mathrm{form}}$ (formation-contingent)")
    ax1.plot(kappa, sig_out_plot, "-^", color=C_G, ms=4, lw=2,
             label=r"$\sigma_{\mathrm{output}}$ (output-contingent)")

    ax1.set_xlabel(r"$\kappa$ (homophily)")
    ax1.set_ylabel(r"Required subsidy $\sigma$")
    ax1.set_title("Three Classes of Subsidy (linear)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Right panel: log scale to show divergence ---
    # Only plot points where sigma > 0
    mask_type = sig_type > 1e-10
    mask_form = (sig_form > 1e-10) & np.isfinite(sig_form)
    mask_out = (sig_out > 1e-10) & np.isfinite(sig_out)

    if np.any(mask_type):
        ax2.semilogy(kappa[mask_type], sig_type[mask_type], "-o", color=C_S1,
                     ms=4, lw=2, label=r"$\sigma_{\mathrm{type}}$ (bounded)")
    if np.any(mask_form):
        ax2.semilogy(kappa[mask_form], sig_form[mask_form], "-s", color=C_GAP,
                     ms=4, lw=2, label=r"$\sigma_{\mathrm{form}} = \Delta / p^2$ (diverges)")
    if np.any(mask_out):
        ax2.semilogy(kappa[mask_out], sig_out[mask_out], "-^", color=C_G,
                     ms=4, lw=2, label=r"$\sigma_{\mathrm{output}} = \Delta / (p^2 h)$ (diverges faster)")

    ax2.set_xlabel(r"$\kappa$ (homophily)")
    ax2.set_ylabel(r"Required subsidy $\sigma$ (log scale)")
    ax2.set_title("Subsidy Divergence (log scale)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Type-contingent subsidy is bounded; downstream subsidies diverge",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "toy_kappa_subsidy.png", save)
    plt.show()


# =====================================================================
# Figure 3: Realization probs + welfare decomposition
# =====================================================================

def plot_realizations(data, save=False):
    """Realization probabilities and welfare vs kappa."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kappa = data["kappa"]
    p = data["p"]

    # Realization probabilities
    prob_R0 = (1 - p) ** 2
    prob_R1 = p * (1 - p)
    prob_R2 = (1 - p) * p
    prob_R3 = p ** 2

    ax1.stackplot(kappa,
                  prob_R3, prob_R1 + prob_R2, prob_R0,
                  labels=[
                      "R3: both edges (full coalition)",
                      "R1+R2: one edge (partial)",
                      "R0: no edges (solo only)",
                  ],
                  colors=[C_SO, C_GAP, "#bdc3c7"],
                  alpha=0.8)
    ax1.set_xlabel(r"$\kappa$ (homophily)")
    ax1.set_ylabel("Probability")
    ax1.set_title("Network Realization Probabilities")
    ax1.legend(loc="center right", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Expected active coalitions: 3 singletons always
    # + P(R1)*1 + P(R2)*1 + P(R3)*3 = 2p(1-p)*1 + p^2*3
    E_active = 3 + 2 * p * (1 - p) + 3 * p ** 2
    ax1_twin = ax1.twinx()
    ax1_twin.plot(kappa, E_active, "--", color="black", lw=1.5,
                  label="E[# active coalitions]")
    ax1_twin.set_ylabel("E[# active coalitions]", color="black")
    ax1_twin.set_ylim(2.5, 7.5)
    ax1_twin.legend(loc="upper right", fontsize=8)

    # Welfare
    ax2.plot(kappa, data["E_W_NE"], "-o", color=C_NE, ms=4, lw=2, label="E[W_NE]")
    ax2.plot(kappa, data["E_W_SO"], "-s", color=C_SO, ms=4, lw=2, label="E[W_SO]")
    ax2.fill_between(kappa, data["E_W_NE"], data["E_W_SO"],
                     alpha=0.15, color=C_GAP, label="Market failure gap")
    ax2.set_xlabel(r"$\kappa$ (homophily)")
    ax2.set_ylabel("Expected welfare")
    ax2.set_title("Expected Welfare: NE vs Social Optimum")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Network Formation and Welfare under Homophily", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, "toy_kappa_realizations.png", save)
    plt.show()


# =====================================================================
# Combined: all three
# =====================================================================

def plot_all(data, save=False):
    """3-row figure combining all analyses."""
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    kappa = data["kappa"]
    p = data["p"]

    # --- Row 1, Left: Payoffs ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(kappa, data["E_u_S1"], "-o", color=C_S1, ms=3, label="E[u_S1]")
    ax1.plot(kappa, data["E_u_S2"], "-s", color=C_S2, ms=3, label="E[u_S2]")
    ax1.plot(kappa, data["E_u_G"],  "-^", color=C_G,  ms=3, label="E[u_G]")
    u_S = np.maximum(data["E_u_S1"], data["E_u_S2"])
    ax1.fill_between(kappa, data["E_u_G"], u_S,
                     where=(u_S > data["E_u_G"]),
                     alpha=0.15, color=C_G, label="G penalty")
    ax1.set_xlabel(r"$\kappa$")
    ax1.set_ylabel("Expected payoff")
    ax1.set_title("(a) Expected Payoffs under NE")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", lw=0.5)

    # --- Row 1, Right: Edge probability ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(kappa, p, "-", color=C_NE, lw=2)
    ax2.set_xlabel(r"$\kappa$")
    ax2.set_ylabel(r"$p = (1/\sqrt{2})^{\kappa}$")
    ax2.set_title("(b) Edge Formation Probability")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # --- Row 2, Left: Subsidy (log scale, all three classes) ---
    ax3 = fig.add_subplot(gs[1, 0])
    sig_type = data["sigma_type"]
    sig_form = np.array(data["sigma_form"])
    sig_out = np.array(data["sigma_output"])
    mask_t = sig_type > 1e-10
    mask_f = (sig_form > 1e-10) & np.isfinite(sig_form)
    mask_o = (sig_out > 1e-10) & np.isfinite(sig_out)
    if np.any(mask_t):
        ax3.semilogy(kappa[mask_t], sig_type[mask_t], "-o", color=C_S1,
                     ms=3, lw=2, label=r"$\sigma_{\mathrm{type}}$")
    if np.any(mask_f):
        ax3.semilogy(kappa[mask_f], sig_form[mask_f], "-s", color=C_GAP,
                     ms=3, lw=2, label=r"$\sigma_{\mathrm{form}}$")
    if np.any(mask_o):
        ax3.semilogy(kappa[mask_o], sig_out[mask_o], "-^", color=C_G,
                     ms=3, lw=2, label=r"$\sigma_{\mathrm{output}}$")
    ax3.set_xlabel(r"$\kappa$")
    ax3.set_ylabel(r"Subsidy (log)")
    ax3.set_title("(c) Three Subsidy Classes")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3, which="both")

    # --- Row 2, Right: Realization probs ---
    ax4 = fig.add_subplot(gs[1, 1])
    prob_R0 = (1 - p) ** 2
    prob_R1 = p * (1 - p)
    prob_R2 = (1 - p) * p
    prob_R3 = p ** 2
    ax4.stackplot(kappa,
                  prob_R3, prob_R1 + prob_R2, prob_R0,
                  labels=["Both edges", "One edge", "No edges"],
                  colors=[C_SO, C_GAP, "#bdc3c7"], alpha=0.8)
    ax4.set_xlabel(r"$\kappa$")
    ax4.set_ylabel("Probability")
    ax4.set_title("(d) Network Realization Probabilities")
    ax4.legend(loc="center right", fontsize=8)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # --- Row 3, Left: Welfare ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(kappa, data["E_W_NE"], "-o", color=C_NE, ms=3, lw=2, label="E[W_NE]")
    ax5.plot(kappa, data["E_W_SO"], "-s", color=C_SO, ms=3, lw=2, label="E[W_SO]")
    ax5.fill_between(kappa, data["E_W_NE"], data["E_W_SO"],
                     alpha=0.15, color=C_GAP, label="Gap")
    ax5.set_xlabel(r"$\kappa$")
    ax5.set_ylabel("Expected welfare")
    ax5.set_title("(e) Welfare: NE vs SO")
    ax5.legend(loc="best", fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- Row 3, Right: Payoff gap u_G - u_S ---
    ax6 = fig.add_subplot(gs[2, 1])
    u_diff = data["E_u_G"] - u_S
    ax6.plot(kappa, u_diff, "-o", color=C_G, ms=3, lw=2)
    ax6.fill_between(kappa, 0, u_diff, where=(u_diff < 0),
                     alpha=0.15, color=C_G, label="G disadvantage")
    ax6.fill_between(kappa, 0, u_diff, where=(u_diff >= 0),
                     alpha=0.15, color=C_S1, label="G advantage")
    ax6.axhline(0, color="gray", lw=1, ls="--")
    ax6.set_xlabel(r"$\kappa$")
    ax6.set_ylabel(r"$E[u_G] - E[u_S]$")
    ax6.set_title("(f) Generalist Payoff Gap")
    ax6.legend(loc="best", fontsize=8)
    ax6.grid(True, alpha=0.3)

    fig.suptitle("Homophily, Generalist Disadvantage, and Optimal Subsidy",
                 fontsize=15, y=1.01)
    _save(fig, "toy_kappa_combined.png", save)
    plt.show()


# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Toy Kappa Sweep Visualization")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["payoffs", "subsidy", "realizations", "all"])
    parser.add_argument("--kappa-max", type=float, default=8.0)
    parser.add_argument("--n-kappa", type=int, default=21)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    kappas = np.linspace(0, args.kappa_max, args.n_kappa)

    print("Computing kappa sweep...")
    data = sweep_kappa(kappas, rho=args.rho, alpha=args.alpha, beta=args.beta,
                      lam=args.lam, tau=args.tau)

    plot_map = {
        "payoffs":       ("Payoffs vs kappa", lambda: plot_payoffs(data, args.save)),
        "subsidy":       ("Optimal subsidy", lambda: plot_subsidy(data, args.save)),
        "realizations":  ("Realization probs", lambda: plot_realizations(data, args.save)),
    }

    if args.plot == "all":
        # Individual figures
        for name, (label, func) in plot_map.items():
            print(f"\n--- {label} ---")
            func()
        # Combined figure
        print("\n--- Combined figure ---")
        plot_all(data, args.save)
    else:
        label, func = plot_map[args.plot]
        print(f"\n--- {label} ---")
        func()

    print("\nDone.")


if __name__ == "__main__":
    main()
