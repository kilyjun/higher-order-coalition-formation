"""
experiment_generalist_ratio.py — Vary generalist fraction p and plot E[W(p)].

Setup: N agents. Each agent drawn as:
  - With prob p:       Generalist,       v = (1/sqrt(2), 1/sqrt(2)),  theta = pi/4
  - With prob (1-p)/2: Specialist type 1, v = (1, 0),                theta = 0
  - With prob (1-p)/2: Specialist type 2, v = (0, 1),                theta = pi/2

For each p, we draw many type profiles, and for each profile draw many
networks, solve NE, and compute E[W].

Goal: plot E[W] vs p, find p* = argmax E[W(p)].

Usage:
    python experiment_generalist_ratio.py
    python experiment_generalist_ratio.py --n_agents 6 --draws 100 --type_draws 50
    python experiment_generalist_ratio.py --save
"""

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from agent import Agent
from network import Network
from coalition import Coalition
from reward import RewardModel
from equilibrium import EquilibriumSolver


def draw_type_profile(n_agents: int, p: float,
                      rng: np.random.Generator) -> List[float]:
    """Draw theta angles for n agents given generalist fraction p.

    Returns list of theta values:
        pi/4  for generalist
        0     for specialist type 1
        pi/2  for specialist type 2
    """
    thetas = []
    for _ in range(n_agents):
        u = rng.random()
        if u < p:
            thetas.append(np.pi / 4)       # generalist
        elif u < p + (1 - p) / 2:
            thetas.append(0.0)              # specialist type 1
        else:
            thetas.append(np.pi / 2)        # specialist type 2
    return thetas


def run_single(agents: List[Agent], kappa: float, rho: float, alpha: float,
               beta: float, lam: float, tau: float, V: float, c: float,
               rng: np.random.Generator) -> float:
    """Run stages 1-3 for one network draw. Returns welfare under NE."""
    net = Network(agents, kappa=kappa, rng=rng)
    net.realize()

    active_keys = net.active_coalitions(max_size=len(agents))
    agent_map = {a.agent_id: a for a in agents}
    coalitions = [
        Coalition([agent_map[aid] for aid in sorted(key)], member_ids=key)
        for key in active_keys
    ]

    if not coalitions:
        return 0.0

    reward_model = RewardModel(tau=tau, V=V, c=c)
    solver = EquilibriumSolver(agents, coalitions, reward_model,
                               rho=rho, alpha=alpha, beta=beta, lam=lam)
    solver.solve_nash()
    return solver.compute_welfare()


def experiment(n_agents: int = 6, p_values: List[float] = None,
               n_type_draws: int = 30, n_network_draws: int = 100,
               kappa: float = 2.0, tau: float = 1.0, rho: float = 0.5,
               alpha: float = 0.5, beta: float = 0.0, lam: float = 0.0,
               V: float = 1.0, c: float = 1.0, seed: int = 42):
    """Run the full p-sweep experiment.

    Returns: (p_values, mean_welfare, std_welfare)
    """
    if p_values is None:
        p_values = np.linspace(0, 1, 21)

    rng = np.random.default_rng(seed)
    mean_welfare = []
    std_welfare = []

    for p in p_values:
        welfare_samples = []
        for _ in range(n_type_draws):
            thetas = draw_type_profile(n_agents, p, rng)
            # Create fresh agents
            agents = [Agent(agent_id=i, theta=t) for i, t in enumerate(thetas)]

            for _ in range(n_network_draws):
                # Fresh agents each network draw (reset efforts)
                agents_fresh = [Agent(agent_id=i, theta=t)
                                for i, t in enumerate(thetas)]
                w = run_single(agents_fresh, kappa, rho, alpha, beta, lam,
                               tau, V, c, rng)
                welfare_samples.append(w)

        mean_welfare.append(np.mean(welfare_samples))
        std_welfare.append(np.std(welfare_samples) / np.sqrt(len(welfare_samples)))
        print(f"  p={p:.2f}  E[W]={mean_welfare[-1]:.4f} +/- {std_welfare[-1]:.4f}")

    return np.array(p_values), np.array(mean_welfare), np.array(std_welfare)


def plot_results(p_values, mean_w, std_w, params: dict, save: bool = False):
    """Plot E[W] vs p with confidence band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(p_values, mean_w, "o-", color="#2980b9", linewidth=2, markersize=5,
            label="E[W(p)] under NE")
    ax.fill_between(p_values, mean_w - 1.96 * std_w, mean_w + 1.96 * std_w,
                    alpha=0.2, color="#2980b9")

    # Mark optimum
    p_star_idx = np.argmax(mean_w)
    p_star = p_values[p_star_idx]
    w_star = mean_w[p_star_idx]
    ax.axvline(p_star, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.plot(p_star, w_star, "*", color="#e74c3c", markersize=15, zorder=5)
    ax.annotate(f"p* = {p_star:.2f}", xy=(p_star, w_star),
                xytext=(p_star + 0.05, w_star + 0.01),
                fontsize=12, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_xlabel("Generalist fraction (p)", fontsize=13)
    ax.set_ylabel("E[W(p)]  (expected surplus under NE)", fontsize=13)
    ax.set_title(
        f"Optimal Generalist Ratio  "
        f"(N={params['n_agents']}, kappa={params['kappa']}, "
        f"tau={params['tau']}, rho={params['rho']}, lam={params['lam']})",
        fontsize=14
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        import os
        os.makedirs("figures", exist_ok=True)
        path = "figures/generalist_ratio.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.show()


def plot_multi_regime(n_agents: int = 6, n_type_draws: int = 30,
                      n_network_draws: int = 100, seed: int = 42,
                      save: bool = False):
    """Plot E[W] vs p for multiple (kappa, tau) regimes side by side."""
    regimes = [
        {"kappa": 0.5, "tau": 0.5, "label": "Low kappa, Easy (tau=0.5)"},
        {"kappa": 0.5, "tau": 2.0, "label": "Low kappa, Hard (tau=2.0)"},
        {"kappa": 2.0, "tau": 0.5, "label": "High kappa, Easy (tau=0.5)"},
        {"kappa": 2.0, "tau": 2.0, "label": "High kappa, Hard (tau=2.0)"},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    p_values = np.linspace(0, 1, 15)

    for ax, regime in zip(axes, regimes):
        print(f"\n--- {regime['label']} ---")
        ps, mw, sw = experiment(
            n_agents=n_agents, p_values=p_values,
            n_type_draws=n_type_draws, n_network_draws=n_network_draws,
            kappa=regime["kappa"], tau=regime["tau"], seed=seed
        )

        ax.plot(ps, mw, "o-", color="#2980b9", linewidth=2, markersize=4)
        ax.fill_between(ps, mw - 1.96 * sw, mw + 1.96 * sw,
                        alpha=0.2, color="#2980b9")

        p_star_idx = np.argmax(mw)
        p_star = ps[p_star_idx]
        ax.axvline(p_star, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.plot(p_star, mw[p_star_idx], "*", color="#e74c3c", markersize=12, zorder=5)
        ax.set_title(f"{regime['label']}\np* = {p_star:.2f}", fontsize=11)
        ax.set_xlabel("Generalist fraction (p)")
        ax.set_ylabel("E[W(p)]")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Optimal Generalist Ratio Across Regimes (N={n_agents})", fontsize=14)
    plt.tight_layout()
    if save:
        import os
        os.makedirs("figures", exist_ok=True)
        path = "figures/generalist_ratio_regimes.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved: {path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment: Optimal generalist ratio p*"
    )
    parser.add_argument("--n_agents", type=int, default=6)
    parser.add_argument("--type_draws", type=int, default=30,
                        help="MC draws over type profiles per p")
    parser.add_argument("--draws", type=int, default=100,
                        help="MC draws over network per type profile")
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi", action="store_true",
                        help="Run 4-regime comparison panel")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    t0 = time.time()

    if args.multi:
        plot_multi_regime(
            n_agents=args.n_agents,
            n_type_draws=args.type_draws,
            n_network_draws=args.draws,
            seed=args.seed, save=args.save
        )
    else:
        params = dict(n_agents=args.n_agents, kappa=args.kappa, tau=args.tau,
                      rho=args.rho, lam=args.lam)
        print(f"\nExperiment: Generalist ratio sweep")
        print(f"  N={args.n_agents}, kappa={args.kappa}, tau={args.tau}, "
              f"rho={args.rho}, lam={args.lam}")
        print(f"  type_draws={args.type_draws}, network_draws={args.draws}\n")

        p_values = np.linspace(0, 1, 21)
        ps, mw, sw = experiment(
            n_agents=args.n_agents, p_values=p_values,
            n_type_draws=args.type_draws, n_network_draws=args.draws,
            kappa=args.kappa, tau=args.tau, rho=args.rho, lam=args.lam,
            seed=args.seed
        )

        p_star_idx = np.argmax(mw)
        print(f"\n  p* = {ps[p_star_idx]:.2f}  (E[W*] = {mw[p_star_idx]:.4f})")

        plot_results(ps, mw, sw, params, save=args.save)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
