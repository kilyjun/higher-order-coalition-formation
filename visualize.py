"""
visualize.py — Visualization script for coalition formation simulation results.

Usage:
    python visualize.py                        # all plots, default settings
    python visualize.py --plot sweep           # single-parameter sweep plots
    python visualize.py --plot heatmap         # kappa x tau heatmap
    python visualize.py --plot friction        # friction analysis
    python visualize.py --plot payoffs         # individual payoff comparison
    python visualize.py --plot all             # everything
    python visualize.py --draws 500            # increase MC precision
    python visualize.py --save                 # save figures to figures/
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict

from config import ModelParams
from simulation import Simulation, SimulationResult


# --- Style ---
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

COLORS = {
    "with_g": "#2980b9",
    "without_g": "#e74c3c",
    "gap": "#27ae60",
    "surplus_ne": "#2c3e50",
    "surplus_so": "#8e44ad",
    "specialist": "#3498db",
    "generalist": "#e67e22",
}


# --- Helpers ---

def run_sweep_data(param_name: str, values: list, draws: int = 200,
                   seed: int = 42, fixed: dict = None) -> Tuple[list, list, list]:
    """Run a 1D sweep and return (values, results_with_G, results_without_G)."""
    results_with = []
    results_without = []
    base = fixed or {}
    for val in values:
        kwargs = {param_name: val, "n_network_draws": draws, "seed": seed, **base}
        p_with = ModelParams(generalist_present=True, **kwargs)
        p_without = ModelParams(generalist_present=False, **kwargs)
        results_with.append(Simulation(p_with).run())
        results_without.append(Simulation(p_without).run())
    return values, results_with, results_without


def extract(results: List[SimulationResult], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in results])


# =====================================================================
# Plot 1: Single-parameter sweeps (4 panels)
# =====================================================================

def plot_sweeps(draws: int = 200, seed: int = 42, save: bool = False):
    """Four-panel plot: sweep kappa, tau, rho, beta."""
    sweeps = [
        ("kappa", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], "Homophily (kappa)"),
        ("tau", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0], "Threshold (tau)"),
        ("rho", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], "Breadth-depth (rho)"),
        ("lam", [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], "Friction (lambda)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (pname, vals, label) in zip(axes, sweeps):
        print(f"  Sweeping {pname}...")
        vs, rw, rwo = run_sweep_data(pname, vals, draws=draws, seed=seed)

        # Generalist value-add
        delta_w = extract(rw, "expected_surplus_ne") - extract(rwo, "expected_surplus_ne")
        # Market failure gap (with G)
        gap = extract(rw, "market_failure_gap")

        ax.plot(vs, delta_w, "o-", color=COLORS["with_g"], label="Generalist value-add (dW)")
        ax.plot(vs, gap, "s--", color=COLORS["gap"], label="Market failure gap")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel(label)
        ax.set_ylabel("Welfare difference")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Parameter Sweeps: Generalist Value-Add & Market Failure Gap", fontsize=14)
    plt.tight_layout()
    if save:
        _save_fig(fig, "sweeps.png")
    plt.show()


# =====================================================================
# Plot 2: Kappa x Tau heatmap
# =====================================================================

def plot_heatmap(draws: int = 200, seed: int = 42, save: bool = False):
    """Heatmap of generalist value-add in (kappa, tau) space."""
    kappas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    taus = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    delta_matrix = np.zeros((len(taus), len(kappas)))
    gap_matrix = np.zeros((len(taus), len(kappas)))

    for i, tau in enumerate(taus):
        for j, kappa in enumerate(kappas):
            print(f"  kappa={kappa}, tau={tau}...")
            p_w = ModelParams(kappa=kappa, tau=tau, generalist_present=True,
                              n_network_draws=draws, seed=seed)
            p_wo = ModelParams(kappa=kappa, tau=tau, generalist_present=False,
                               n_network_draws=draws, seed=seed)
            rw = Simulation(p_w).run()
            rwo = Simulation(p_wo).run()
            delta_matrix[i, j] = rw.expected_surplus_ne - rwo.expected_surplus_ne
            gap_matrix[i, j] = rw.market_failure_gap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Generalist value-add heatmap
    vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    im1 = ax1.imshow(delta_matrix, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax, origin="lower")
    ax1.set_xticks(range(len(kappas)))
    ax1.set_xticklabels([f"{k:.1f}" for k in kappas])
    ax1.set_yticks(range(len(taus)))
    ax1.set_yticklabels([f"{t:.2f}" for t in taus])
    ax1.set_xlabel("Homophily (kappa)")
    ax1.set_ylabel("Threshold (tau)")
    ax1.set_title("Generalist Value-Add (dW_NE)")
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    # Annotate cells
    for i in range(len(taus)):
        for j in range(len(kappas)):
            ax1.text(j, i, f"{delta_matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(delta_matrix[i,j]) > vmax*0.5 else "black")

    # Market failure gap heatmap
    im2 = ax2.imshow(gap_matrix, cmap="Greens", aspect="auto", origin="lower")
    ax2.set_xticks(range(len(kappas)))
    ax2.set_xticklabels([f"{k:.1f}" for k in kappas])
    ax2.set_yticks(range(len(taus)))
    ax2.set_yticklabels([f"{t:.2f}" for t in taus])
    ax2.set_xlabel("Homophily (kappa)")
    ax2.set_ylabel("Threshold (tau)")
    ax2.set_title("Market Failure Gap (W_SO - W_NE)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    for i in range(len(taus)):
        for j in range(len(kappas)):
            ax2.text(j, i, f"{gap_matrix[i,j]:.3f}", ha="center", va="center",
                     fontsize=8, color="white" if gap_matrix[i,j] > gap_matrix.max()*0.5 else "black")

    fig.suptitle("Kappa x Tau: Where Generalists Matter Most", fontsize=14)
    plt.tight_layout()
    if save:
        _save_fig(fig, "heatmap_kappa_tau.png")
    plt.show()


# =====================================================================
# Plot 3: Friction analysis (lambda vs kappa interaction)
# =====================================================================

def plot_friction(draws: int = 200, seed: int = 42, save: bool = False):
    """Show how friction interacts with homophily and threshold."""
    lambdas = [0.0, 0.5, 1.0, 2.0, 4.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Fix tau=2 (hard problem), vary kappa across friction levels
    kappas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    cmap = plt.cm.viridis
    for idx, lam in enumerate(lambdas):
        color = cmap(idx / (len(lambdas) - 1))
        deltas = []
        for kappa in kappas:
            p_w = ModelParams(kappa=kappa, tau=2.0, lam=lam, generalist_present=True,
                              n_network_draws=draws, seed=seed)
            p_wo = ModelParams(kappa=kappa, tau=2.0, lam=lam, generalist_present=False,
                               n_network_draws=draws, seed=seed)
            rw = Simulation(p_w).run()
            rwo = Simulation(p_wo).run()
            deltas.append(rw.expected_surplus_ne - rwo.expected_surplus_ne)
        ax1.plot(kappas, deltas, "o-", color=color, label=f"lam={lam:.1f}")

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Homophily (kappa)")
    ax1.set_ylabel("Generalist value-add (dW)")
    ax1.set_title("Friction x Homophily (tau=2.0)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Fix kappa=2 (high homophily), vary tau across friction levels
    taus = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    for idx, lam in enumerate(lambdas):
        color = cmap(idx / (len(lambdas) - 1))
        deltas = []
        for tau in taus:
            p_w = ModelParams(kappa=2.0, tau=tau, lam=lam, generalist_present=True,
                              n_network_draws=draws, seed=seed)
            p_wo = ModelParams(kappa=2.0, tau=tau, lam=lam, generalist_present=False,
                               n_network_draws=draws, seed=seed)
            rw = Simulation(p_w).run()
            rwo = Simulation(p_wo).run()
            deltas.append(rw.expected_surplus_ne - rwo.expected_surplus_ne)
        ax2.plot(taus, deltas, "o-", color=color, label=f"lam={lam:.1f}")

    ax2.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Threshold (tau)")
    ax2.set_ylabel("Generalist value-add (dW)")
    ax2.set_title("Friction x Threshold (kappa=2.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Communication Friction: Extensive vs Intensive Margin Trade-off", fontsize=14)
    plt.tight_layout()
    if save:
        _save_fig(fig, "friction_analysis.png")
    plt.show()


# =====================================================================
# Plot 4: Individual payoff comparison
# =====================================================================

def plot_payoffs(draws: int = 200, seed: int = 42, save: bool = False):
    """Compare specialist vs generalist individual payoffs across regimes."""
    kappas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Payoffs under NE
    specialist_payoffs = []
    generalist_payoffs = []
    for kappa in kappas:
        p = ModelParams(kappa=kappa, generalist_present=True,
                        n_network_draws=draws, seed=seed)
        res = Simulation(p).run()
        # Agent 0,1 are specialists, agent 2 is generalist
        specialist_payoffs.append(res.expected_payoffs_ne[0])
        generalist_payoffs.append(res.expected_payoffs_ne[2])

    ax1.plot(kappas, specialist_payoffs, "o-", color=COLORS["specialist"],
             label="Specialist (agent 0)")
    ax1.plot(kappas, generalist_payoffs, "s-", color=COLORS["generalist"],
             label="Generalist (agent 2)")
    ax1.fill_between(kappas, specialist_payoffs, generalist_payoffs,
                     alpha=0.15, color=COLORS["generalist"])
    ax1.set_xlabel("Homophily (kappa)")
    ax1.set_ylabel("Expected payoff E[u_i]")
    ax1.set_title("Individual Payoffs under NE (tau=1.0)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Payoffs across tau
    taus = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    specialist_payoffs_t = []
    generalist_payoffs_t = []
    for tau in taus:
        p = ModelParams(tau=tau, kappa=2.0, generalist_present=True,
                        n_network_draws=draws, seed=seed)
        res = Simulation(p).run()
        specialist_payoffs_t.append(res.expected_payoffs_ne[0])
        generalist_payoffs_t.append(res.expected_payoffs_ne[2])

    ax2.plot(taus, specialist_payoffs_t, "o-", color=COLORS["specialist"],
             label="Specialist (agent 0)")
    ax2.plot(taus, generalist_payoffs_t, "s-", color=COLORS["generalist"],
             label="Generalist (agent 2)")
    ax2.fill_between(taus, specialist_payoffs_t, generalist_payoffs_t,
                     alpha=0.15, color=COLORS["generalist"])
    ax2.set_xlabel("Threshold (tau)")
    ax2.set_ylabel("Expected payoff E[u_i]")
    ax2.set_title("Individual Payoffs under NE (kappa=2.0)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Career Incentives: Specialist vs Generalist Payoffs", fontsize=14)
    plt.tight_layout()
    if save:
        _save_fig(fig, "payoff_comparison.png")
    plt.show()


# =====================================================================
# Plot 5: Effort allocation decomposition
# =====================================================================

def plot_effort_shares(draws: int = 200, seed: int = 42, save: bool = False):
    """Stacked bar chart of effort allocation by coalition size."""
    kappas = [0.0, 0.5, 1.0, 2.0, 4.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: With generalist
    solo_shares, pair_shares, triad_shares = [], [], []
    for kappa in kappas:
        p = ModelParams(kappa=kappa, generalist_present=True,
                        n_network_draws=draws, seed=seed)
        res = Simulation(p).run()
        solo_shares.append(res.effort_shares.get(1, 0))
        pair_shares.append(res.effort_shares.get(2, 0))
        triad_shares.append(res.effort_shares.get(3, 0))

    x = np.arange(len(kappas))
    width = 0.6
    ax1.bar(x, solo_shares, width, label="Solo", color="#3498db")
    ax1.bar(x, pair_shares, width, bottom=solo_shares, label="Pair", color="#2ecc71")
    ax1.bar(x, triad_shares, width,
            bottom=np.array(solo_shares) + np.array(pair_shares),
            label="Triad", color="#e74c3c")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{k:.1f}" for k in kappas])
    ax1.set_xlabel("Homophily (kappa)")
    ax1.set_ylabel("Effort share")
    ax1.set_title("With Generalist")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Panel 2: Without generalist
    solo_shares2, pair_shares2, triad_shares2 = [], [], []
    for kappa in kappas:
        p = ModelParams(kappa=kappa, generalist_present=False,
                        n_network_draws=draws, seed=seed)
        res = Simulation(p).run()
        solo_shares2.append(res.effort_shares.get(1, 0))
        pair_shares2.append(res.effort_shares.get(2, 0))
        triad_shares2.append(res.effort_shares.get(3, 0))

    ax2.bar(x, solo_shares2, width, label="Solo", color="#3498db")
    ax2.bar(x, pair_shares2, width, bottom=solo_shares2, label="Pair", color="#2ecc71")
    ax2.bar(x, triad_shares2, width,
            bottom=np.array(solo_shares2) + np.array(pair_shares2),
            label="Triad", color="#e74c3c")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{k:.1f}" for k in kappas])
    ax2.set_xlabel("Homophily (kappa)")
    ax2.set_ylabel("Effort share")
    ax2.set_title("Without Generalist")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Effort Allocation by Coalition Size", fontsize=14)
    plt.tight_layout()
    if save:
        _save_fig(fig, "effort_shares.png")
    plt.show()


# =====================================================================
# Utilities
# =====================================================================

def _save_fig(fig, filename: str):
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualization for Coalition Simulation")
    parser.add_argument("--plot", type=str, default="all",
                        choices=["sweep", "heatmap", "friction", "payoffs",
                                 "effort", "all"],
                        help="Which plot(s) to generate")
    parser.add_argument("--draws", type=int, default=200,
                        help="Number of network Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true",
                        help="Save figures to figures/ directory")
    args = parser.parse_args()

    plot_funcs = {
        "sweep": ("Parameter sweeps", plot_sweeps),
        "heatmap": ("Kappa x Tau heatmap", plot_heatmap),
        "friction": ("Friction analysis", plot_friction),
        "payoffs": ("Payoff comparison", plot_payoffs),
        "effort": ("Effort shares", plot_effort_shares),
    }

    if args.plot == "all":
        targets = list(plot_funcs.keys())
    else:
        targets = [args.plot]

    for name in targets:
        label, func = plot_funcs[name]
        print(f"\n--- {label} ---")
        func(draws=args.draws, seed=args.seed, save=args.save)

    print("\nDone.")


if __name__ == "__main__":
    main()
