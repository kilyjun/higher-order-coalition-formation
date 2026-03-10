"""
toy_example.py — Deterministic 3-agent complete-network analysis.

Agents:
    S1: specialist type 1, v = (1, 0),              theta = 0
    S2: specialist type 2, v = (0, 1),              theta = pi/2
    G:  generalist,        v = (1/sqrt2, 1/sqrt2),  theta = pi/4

Network: complete graph (all 3 links exist). No randomness.

Coalitions (all active): {S1}, {S2}, {G}, {S1,S2}, {S1,G}, {S2,G}, {S1,S2,G}

For each parameter configuration, prints a full decomposition:
    - Coalition-level: synergy, friction, effective output, breakthrough prob
    - Agent-level: effort allocation, payoffs under NE vs SO
    - System-level: welfare, market failure gap

Usage:
    python toy_example.py
    python toy_example.py --sweep rho
    python toy_example.py --sweep tau
    python toy_example.py --sweep lam
"""

import argparse
import numpy as np
from typing import List, Dict

from agent import Agent
from coalition import Coalition
from reward import RewardModel
from equilibrium import EquilibriumSolver


# ── Agent setup ──────────────────────────────────────────────────────

def create_agents() -> List[Agent]:
    return [
        Agent(agent_id=0, theta=0.0),           # S1: (1, 0)
        Agent(agent_id=1, theta=np.pi / 2),     # S2: (0, 1)
        Agent(agent_id=2, theta=np.pi / 4),     # G:  (1/sqrt2, 1/sqrt2)
    ]


AGENT_LABELS = {0: "S1", 1: "S2", 2: "G "}


# ── Build all 7 coalitions on the complete graph ─────────────────────

def build_all_coalitions(agents: List[Agent]) -> List[Coalition]:
    """All 2^3 - 1 = 7 non-empty subsets, all active on complete graph."""
    from itertools import combinations
    coalitions = []
    ids = [a.agent_id for a in agents]
    agent_map = {a.agent_id: a for a in agents}
    for size in range(1, len(ids) + 1):
        for combo in combinations(ids, size):
            key = frozenset(combo)
            members = [agent_map[aid] for aid in sorted(combo)]
            coalitions.append(Coalition(members, member_ids=key))
    return coalitions


def coalition_label(key) -> str:
    names = [AGENT_LABELS[i].strip() for i in sorted(key)]
    return "{" + ",".join(names) + "}"


# ── Analysis for one parameter setting ───────────────────────────────

def analyze(rho: float = 0.5, alpha: float = 0.5, beta: float = 0.0,
            lam: float = 0.0, tau: float = 1.0, V: float = 1.0,
            c: float = 1.0, verbose: bool = True) -> Dict:
    """Full analysis of the 3-agent toy example. Returns results dict."""

    agents = create_agents()
    coalitions = build_all_coalitions(agents)
    reward = RewardModel(tau=tau, V=V, c=c)
    solver = EquilibriumSolver(agents, coalitions, reward,
                               rho=rho, alpha=alpha, beta=beta, lam=lam)

    # ── Static coalition properties (before effort) ──
    if verbose:
        print("=" * 72)
        print(f"  PARAMETERS: rho={rho:.2f}  tau={tau:.2f}  lam={lam:.2f}  "
              f"alpha={alpha:.2f}  beta={beta:.2f}  c={c:.2f}  V={V:.2f}")
        print("=" * 72)

        print("\n  Agents:")
        for a in agents:
            v = a.skill_vector
            print(f"    {AGENT_LABELS[a.agent_id]}: v=({v[0]:.3f}, {v[1]:.3f})  "
                  f"solo_depth={a.solo_depth:.3f}  solo_breadth={a.solo_breadth:.3f}")

        print(f"\n  Pairwise similarities:")
        for i in range(3):
            for j in range(i + 1, 3):
                sim = agents[i].similarity(agents[j])
                print(f"    sim({AGENT_LABELS[i].strip()}, {AGENT_LABELS[j].strip()}) = {sim:.4f}")

        print(f"\n  Coalition properties (static):")
        print(f"    {'Coalition':<16} {'Breadth':>8} {'Depth':>8} "
              f"{'Synergy':>8} {'AvgSim':>8} {'Friction':>8}")
        print(f"    {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for coal in coalitions:
            label = coalition_label(coal.key)
            print(f"    {label:<16} {coal.breadth:8.4f} {coal.depth:8.4f} "
                  f"{coal.synergy(rho):8.4f} {coal.avg_similarity:8.4f} "
                  f"{coal.friction(lam):8.4f}")

    # ── Nash Equilibrium ──
    solver.solve_nash()
    w_ne = solver.compute_welfare()
    payoffs_ne = solver.compute_payoffs()
    efforts_ne = {a.agent_id: dict(a.efforts) for a in agents}
    total_efforts_ne = {a.agent_id: a.total_effort for a in agents}

    # Collect NE coalition outputs
    ne_outputs = {}
    for coal in coalitions:
        q = coal.effective_output(rho, alpha, beta, lam)
        prob = reward.breakthrough_prob(q)
        ne_outputs[coal.key] = {"q": q, "prob": prob}

    if verbose:
        print(f"\n  --- NASH EQUILIBRIUM ---")
        print(f"\n  Effort allocations:")
        print(f"    {'Agent':<6} ", end="")
        for coal in coalitions:
            print(f"{coalition_label(coal.key):>12}", end="")
        print(f"  {'Total':>8}")
        print(f"    {'-'*6} ", end="")
        for _ in coalitions:
            print(f"  {'-'*10}", end="")
        print(f"  {'-'*8}")
        for a in agents:
            print(f"    {AGENT_LABELS[a.agent_id]:<6} ", end="")
            for coal in coalitions:
                e = efforts_ne[a.agent_id].get(coal.key, 0.0)
                if a.agent_id in coal.member_ids:
                    print(f"{e:12.6f}", end="")
                else:
                    print(f"{'---':>12}", end="")
            print(f"  {total_efforts_ne[a.agent_id]:8.4f}")

        print(f"\n  Coalition outputs under NE:")
        print(f"    {'Coalition':<16} {'q_C':>10} {'P(BT)':>10} {'h*V':>10} {'per-member':>12}")
        print(f"    {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
        for coal in coalitions:
            d = ne_outputs[coal.key]
            hv = d["prob"] * V
            per = hv / coal.size
            print(f"    {coalition_label(coal.key):<16} {d['q']:10.6f} "
                  f"{d['prob']:10.6f} {hv:10.6f} {per:12.6f}")

        print(f"\n  Agent payoffs under NE:")
        for a in agents:
            print(f"    {AGENT_LABELS[a.agent_id]}: u = {payoffs_ne[a.agent_id]:+.6f}  "
                  f"(effort cost = {a.effort_cost(c):.6f})")
        print(f"\n  Total surplus (NE):  W_NE = {w_ne:+.6f}")

    # ── Social Optimum ──
    agents_so = create_agents()
    coalitions_so = build_all_coalitions(agents_so)
    solver_so = EquilibriumSolver(agents_so, coalitions_so, reward,
                                  rho=rho, alpha=alpha, beta=beta, lam=lam)
    solver_so.solve_social_optimum()
    w_so = solver_so.compute_welfare()
    payoffs_so = solver_so.compute_payoffs()
    efforts_so = {a.agent_id: dict(a.efforts) for a in agents_so}
    total_efforts_so = {a.agent_id: a.total_effort for a in agents_so}

    so_outputs = {}
    for coal in coalitions_so:
        q = coal.effective_output(rho, alpha, beta, lam)
        prob = reward.breakthrough_prob(q)
        so_outputs[coal.key] = {"q": q, "prob": prob}

    if verbose:
        print(f"\n  --- SOCIAL OPTIMUM ---")
        print(f"\n  Effort allocations:")
        print(f"    {'Agent':<6} ", end="")
        for coal in coalitions_so:
            print(f"{coalition_label(coal.key):>12}", end="")
        print(f"  {'Total':>8}")
        print(f"    {'-'*6} ", end="")
        for _ in coalitions_so:
            print(f"  {'-'*10}", end="")
        print(f"  {'-'*8}")
        for a in agents_so:
            print(f"    {AGENT_LABELS[a.agent_id]:<6} ", end="")
            for coal in coalitions_so:
                e = efforts_so[a.agent_id].get(coal.key, 0.0)
                if a.agent_id in coal.member_ids:
                    print(f"{e:12.6f}", end="")
                else:
                    print(f"{'---':>12}", end="")
            print(f"  {total_efforts_so[a.agent_id]:8.4f}")

        print(f"\n  Coalition outputs under SO:")
        print(f"    {'Coalition':<16} {'q_C':>10} {'P(BT)':>10} {'h*V':>10} {'per-member':>12}")
        print(f"    {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
        for coal in coalitions_so:
            d = so_outputs[coal.key]
            hv = d["prob"] * V
            per = hv / coal.size
            print(f"    {coalition_label(coal.key):<16} {d['q']:10.6f} "
                  f"{d['prob']:10.6f} {hv:10.6f} {per:12.6f}")

        print(f"\n  Agent payoffs under SO:")
        for a in agents_so:
            print(f"    {AGENT_LABELS[a.agent_id]}: u = {payoffs_so[a.agent_id]:+.6f}  "
                  f"(effort cost = {a.effort_cost(c):.6f})")
        print(f"\n  Total surplus (SO):  W_SO = {w_so:+.6f}")

        # ── Summary ──
        gap = w_so - w_ne
        print(f"\n  {'='*50}")
        print(f"  MARKET FAILURE GAP:  W_SO - W_NE = {gap:+.6f}")
        print(f"  Generalist payoff disadvantage (NE): "
              f"u_G - u_S1 = {payoffs_ne[2] - payoffs_ne[0]:+.6f}")
        print(f"  {'='*50}\n")

    return {
        "w_ne": w_ne, "w_so": w_so, "gap": w_so - w_ne,
        "payoffs_ne": payoffs_ne, "payoffs_so": payoffs_so,
        "efforts_ne": efforts_ne, "efforts_so": efforts_so,
        "ne_outputs": ne_outputs, "so_outputs": so_outputs,
    }


# ── Parameter sweeps with compact table output ──────────────────────

def sweep(param_name: str, values: list, base: dict = None):
    """Sweep one parameter, print compact comparison table."""
    base = base or {}
    defaults = dict(rho=0.5, alpha=0.5, beta=0.0, lam=0.0, tau=1.0, V=1.0, c=1.0)
    defaults.update(base)

    print(f"\n{'='*80}")
    print(f"  SWEEP: {param_name}")
    print(f"  Base: " + "  ".join(f"{k}={v}" for k, v in defaults.items() if k != param_name))
    print(f"{'='*80}")
    print(f"\n  {param_name:>8}  {'W_NE':>10}  {'W_SO':>10}  {'Gap':>10}  "
          f"{'u_S1':>10}  {'u_S2':>10}  {'u_G':>10}  {'u_G-u_S1':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for val in values:
        kwargs = dict(defaults)
        kwargs[param_name] = val
        res = analyze(verbose=False, **kwargs)
        pne = res["payoffs_ne"]
        print(f"  {val:8.3f}  {res['w_ne']:+10.6f}  {res['w_so']:+10.6f}  "
              f"{res['gap']:+10.6f}  {pne[0]:+10.6f}  {pne[1]:+10.6f}  "
              f"{pne[2]:+10.6f}  {pne[2]-pne[0]:+10.6f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Toy 3-agent complete-graph analysis")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["rho", "tau", "lam", "beta", "all"],
                        help="Parameter to sweep (or 'all')")
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    args = parser.parse_args()

    base = dict(rho=args.rho, tau=args.tau, lam=args.lam, beta=args.beta)

    if args.sweep is None:
        # Single detailed analysis
        analyze(**base)

    elif args.sweep == "all":
        sweep("rho", [0.0, 0.25, 0.5, 0.75, 1.0], base)
        sweep("tau", [0.25, 0.5, 1.0, 2.0, 4.0], base)
        sweep("lam", [0.0, 0.5, 1.0, 2.0, 4.0], base)
        sweep("beta", [0.0, 0.25, 0.5, 0.75, 1.0], base)

    else:
        ranges = {
            "rho": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "tau": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            "lam": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            "beta": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
        }
        sweep(args.sweep, ranges[args.sweep], base)


if __name__ == "__main__":
    main()
