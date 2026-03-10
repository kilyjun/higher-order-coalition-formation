"""
toy_kappa_sweep.py — 3-agent toy with homophilic network formation (kappa).

Agents: S1 (1,0), S2 (0,1), G (1/sqrt2, 1/sqrt2)
Edges:  P(S1-S2) = 0^kappa = 0  (always absent for kappa > 0)
        P(S1-G) = P(S2-G) = (1/sqrt2)^kappa

With S1-S2 always absent, there are 4 network realizations:
  R0: no edges           prob = (1-p)^2
  R1: S1-G only          prob = p(1-p)
  R2: S2-G only          prob = (1-p)p
  R3: S1-G + S2-G        prob = p^2

Active coalitions per realization:
  R0: {S1}, {S2}, {G}
  R1: {S1}, {S2}, {G}, {S1,G}
  R2: {S1}, {S2}, {G}, {S2,G}
  R3: {S1}, {S2}, {G}, {S1,G}, {S2,G}, {S1,S2,G}

All 4 realizations are enumerated exactly (no Monte Carlo).

Subsidy sigma*(kappa) = max(0, u_S - u_G) under NE, i.e., the lump-sum
transfer to the generalist that equalizes expected payoffs.

Usage:
    python toy_kappa_sweep.py
    python toy_kappa_sweep.py --kappa-max 15 --n-kappa 30
    python toy_kappa_sweep.py --rho 0.8 --tau 0.25 --beta 0.75
"""

import argparse
import numpy as np
from typing import List, Dict, Tuple, FrozenSet

from agent import Agent
from coalition import Coalition
from reward import RewardModel
from equilibrium import EquilibriumSolver


AGENT_LABELS = {0: "S1", 1: "S2", 2: "G"}


def create_agents() -> List[Agent]:
    return [
        Agent(agent_id=0, theta=0.0),           # S1: (1, 0)
        Agent(agent_id=1, theta=np.pi / 2),     # S2: (0, 1)
        Agent(agent_id=2, theta=np.pi / 4),     # G:  (1/sqrt2, 1/sqrt2)
    ]


# -- Network realizations for 3 agents with P(S1-S2) = 0 --

# Each realization is a list of active coalition keys (frozensets)
SINGLETONS = [frozenset({0}), frozenset({1}), frozenset({2})]


def _realization_active_keys() -> List[Tuple[str, List[FrozenSet[int]]]]:
    """Return the 4 realizations and their active coalition keys."""
    return [
        ("R0: no edges",    SINGLETONS),
        ("R1: S1-G only",   SINGLETONS + [frozenset({0, 2})]),
        ("R2: S2-G only",   SINGLETONS + [frozenset({1, 2})]),
        ("R3: S1-G + S2-G", SINGLETONS + [frozenset({0, 2}), frozenset({1, 2}),
                                           frozenset({0, 1, 2})]),
    ]


def _realization_probs(p: float) -> List[float]:
    """Probabilities for R0..R3 given edge prob p = (1/sqrt2)^kappa."""
    return [
        (1 - p) ** 2,      # R0
        p * (1 - p),        # R1
        (1 - p) * p,        # R2
        p ** 2,             # R3
    ]


def solve_realization(active_keys: List[FrozenSet[int]],
                      rho: float, alpha: float, beta: float, lam: float,
                      tau: float, V: float, c: float) -> Dict:
    """Solve NE for one network realization. Returns payoffs and welfare."""
    agents = create_agents()
    agent_map = {a.agent_id: a for a in agents}

    coalitions = []
    for key in active_keys:
        members = [agent_map[aid] for aid in sorted(key)]
        coalitions.append(Coalition(members, member_ids=key))

    reward = RewardModel(tau=tau, V=V, c=c)
    solver = EquilibriumSolver(agents, coalitions, reward,
                               rho=rho, alpha=alpha, beta=beta, lam=lam)

    solver.solve_nash()
    welfare_ne = solver.compute_welfare()
    payoffs_ne = solver.compute_payoffs()
    efforts_ne = {a.agent_id: dict(a.efforts) for a in agents}

    # Social optimum (fresh agents)
    agents_so = create_agents()
    agent_map_so = {a.agent_id: a for a in agents_so}
    coalitions_so = []
    for key in active_keys:
        members = [agent_map_so[aid] for aid in sorted(key)]
        coalitions_so.append(Coalition(members, member_ids=key))

    solver_so = EquilibriumSolver(agents_so, coalitions_so, reward,
                                  rho=rho, alpha=alpha, beta=beta, lam=lam)
    solver_so.solve_social_optimum()
    welfare_so = solver_so.compute_welfare()
    payoffs_so = solver_so.compute_payoffs()

    return {
        "payoffs_ne": payoffs_ne,
        "payoffs_so": payoffs_so,
        "welfare_ne": welfare_ne,
        "welfare_so": welfare_so,
        "efforts_ne": efforts_ne,
    }


def analyze_kappa(kappa: float, rho: float = 0.5, alpha: float = 0.5,
                  beta: float = 0.0, lam: float = 0.0, tau: float = 1.0,
                  V: float = 1.0, c: float = 1.0,
                  verbose: bool = False) -> Dict:
    """Analyze expected payoffs at a given kappa by exact enumeration."""
    # Edge probability
    sim_SG = 1.0 / np.sqrt(2)  # sim(S1,G) = sim(S2,G)
    if kappa == 0:
        p = 1.0  # 0^0 = 1, all edges form
    else:
        p = sim_SG ** kappa

    realizations = _realization_active_keys()
    probs = _realization_probs(p)

    # Expected payoffs
    E_payoffs_ne = {0: 0.0, 1: 0.0, 2: 0.0}
    E_payoffs_so = {0: 0.0, 1: 0.0, 2: 0.0}
    E_welfare_ne = 0.0
    E_welfare_so = 0.0

    realization_details = []

    for (label, active_keys), prob in zip(realizations, probs):
        if prob < 1e-15:
            realization_details.append({
                "label": label, "prob": prob,
                "payoffs_ne": {0: 0.0, 1: 0.0, 2: 0.0},
                "welfare_ne": 0.0,
            })
            continue

        res = solve_realization(active_keys, rho, alpha, beta, lam, tau, V, c)

        for aid in [0, 1, 2]:
            E_payoffs_ne[aid] += prob * res["payoffs_ne"][aid]
            E_payoffs_so[aid] += prob * res["payoffs_so"][aid]
        E_welfare_ne += prob * res["welfare_ne"]
        E_welfare_so += prob * res["welfare_so"]

        realization_details.append({
            "label": label, "prob": prob,
            "payoffs_ne": res["payoffs_ne"],
            "payoffs_so": res["payoffs_so"],
            "welfare_ne": res["welfare_ne"],
            "welfare_so": res["welfare_so"],
            "efforts_ne": res["efforts_ne"],
        })

    # --- Three classes of subsidy ---
    # Gap: how much more specialists earn than the generalist in expectation
    u_S_max = max(E_payoffs_ne[0], E_payoffs_ne[1])
    delta = max(0.0, u_S_max - E_payoffs_ne[2])

    # (1) Type-contingent: lump-sum to G regardless of coalition formation
    sigma_type = delta

    # (2) Formation-contingent: paid only when grand coalition {S1,S2,G} forms
    #     E[sigma_form] = sigma_form * P_form = delta  =>  sigma_form = delta / P_form
    P_form = p ** 2  # both G-S1 and G-S2 edges must realize
    sigma_form = delta / P_form if P_form > 1e-15 else float("inf")

    # (3) Output-contingent: paid only when grand coalition forms AND breakthrough
    #     Need P(breakthrough | grand coalition formed) from R3 realization
    #     sigma_output = delta / (P_form * P_success_given_form)
    P_success_given_form = 0.0
    for d in realization_details:
        if d["label"].startswith("R3") and "efforts_ne" in d:
            # Compute breakthrough prob of grand coalition under R3's NE
            agents_r3 = create_agents()
            agent_map_r3 = {a.agent_id: a for a in agents_r3}
            # Restore R3 efforts
            for aid, eff_dict in d["efforts_ne"].items():
                for ckey, eff_val in eff_dict.items():
                    agents_r3[aid].efforts[ckey] = eff_val
            grand_key = frozenset({0, 1, 2})
            grand_members = [agent_map_r3[aid] for aid in sorted(grand_key)]
            grand_coal = Coalition(grand_members, member_ids=grand_key)
            q_grand = grand_coal.effective_output(rho, alpha, beta, lam)
            reward_model = RewardModel(tau=tau, V=V, c=c)
            P_success_given_form = reward_model.breakthrough_prob(q_grand)
            break

    denom_output = P_form * P_success_given_form
    sigma_output = delta / denom_output if denom_output > 1e-15 else float("inf")

    if verbose:
        print(f"\n  kappa = {kappa:.2f},  p(S-G edge) = {p:.6f},  "
              f"P_form(grand) = {P_form:.6f}")
        print(f"  {'Realization':<20} {'Prob':>8}  {'u_S1':>10} {'u_S2':>10} "
              f"{'u_G':>10} {'W_NE':>10}")
        print(f"  {'-'*20} {'-'*8}  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for d in realization_details:
            pne = d["payoffs_ne"]
            print(f"  {d['label']:<20} {d['prob']:8.6f}  {pne[0]:+10.6f} "
                  f"{pne[1]:+10.6f} {pne[2]:+10.6f} "
                  f"{d.get('welfare_ne', 0.0):+10.6f}")
        print(f"\n  Expected:  E[u_S1]={E_payoffs_ne[0]:+.6f}  "
              f"E[u_S2]={E_payoffs_ne[1]:+.6f}  E[u_G]={E_payoffs_ne[2]:+.6f}")
        print(f"  E[W_NE]={E_welfare_ne:+.6f}  E[W_SO]={E_welfare_so:+.6f}  "
              f"Gap={E_welfare_so - E_welfare_ne:+.6f}")
        print(f"  Subsidies:  type={sigma_type:.6f}  "
              f"formation={sigma_form:.6f}  output={sigma_output:.6f}")

    return {
        "kappa": kappa,
        "p": p,
        "P_form": P_form,
        "P_success_given_form": P_success_given_form,
        "E_payoffs_ne": E_payoffs_ne,
        "E_payoffs_so": E_payoffs_so,
        "E_welfare_ne": E_welfare_ne,
        "E_welfare_so": E_welfare_so,
        "sigma_type": sigma_type,
        "sigma_form": sigma_form,
        "sigma_output": sigma_output,
        "realization_details": realization_details,
    }


def sweep_kappa(kappa_values: np.ndarray, rho: float = 0.5, alpha: float = 0.5,
                beta: float = 0.0, lam: float = 0.0, tau: float = 1.0,
                V: float = 1.0, c: float = 1.0) -> Dict:
    """Sweep kappa and collect results. Returns dict of arrays."""
    results = []
    arrays = {
        "kappa": [], "p": [], "P_form": [],
        "E_u_S1": [], "E_u_S2": [], "E_u_G": [],
        "E_W_NE": [], "E_W_SO": [], "gap": [],
        "sigma_type": [], "sigma_form": [], "sigma_output": [],
    }

    print(f"\n{'='*120}")
    print(f"  KAPPA SWEEP  (rho={rho:.2f}  tau={tau:.2f}  beta={beta:.2f}  "
          f"lam={lam:.2f}  alpha={alpha:.2f}  c={c:.2f}  V={V:.2f})")
    print(f"{'='*120}")
    print(f"\n  {'kappa':>6}  {'p':>8}  {'P_form':>8}  {'E[u_S]':>10}  "
          f"{'E[u_G]':>10}  {'delta':>10}  "
          f"{'sig_type':>10}  {'sig_form':>10}  {'sig_output':>10}  "
          f"{'E[W_NE]':>10}  {'Gap':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for kappa in kappa_values:
        res = analyze_kappa(kappa, rho=rho, alpha=alpha, beta=beta,
                           lam=lam, tau=tau, V=V, c=c)
        results.append(res)

        pne = res["E_payoffs_ne"]
        gap_val = res["E_welfare_so"] - res["E_welfare_ne"]
        delta = res["sigma_type"]
        sf = res["sigma_form"]
        so = res["sigma_output"]

        sf_str = f"{sf:10.4f}" if sf < 1e6 else f"{'inf':>10}"
        so_str = f"{so:10.4f}" if so < 1e6 else f"{'inf':>10}"

        print(f"  {kappa:6.2f}  {res['p']:8.6f}  {res['P_form']:8.6f}  "
              f"{max(pne[0], pne[1]):+10.6f}  {pne[2]:+10.6f}  {delta:10.6f}  "
              f"{delta:10.6f}  {sf_str}  {so_str}  "
              f"{res['E_welfare_ne']:+10.6f}  {gap_val:+10.6f}")

        arrays["kappa"].append(kappa)
        arrays["p"].append(res["p"])
        arrays["P_form"].append(res["P_form"])
        arrays["E_u_S1"].append(pne[0])
        arrays["E_u_S2"].append(pne[1])
        arrays["E_u_G"].append(pne[2])
        arrays["E_W_NE"].append(res["E_welfare_ne"])
        arrays["E_W_SO"].append(res["E_welfare_so"])
        arrays["gap"].append(gap_val)
        arrays["sigma_type"].append(res["sigma_type"])
        arrays["sigma_form"].append(res["sigma_form"])
        arrays["sigma_output"].append(res["sigma_output"])

    print()

    # Convert to numpy
    for k in arrays:
        arrays[k] = np.array(arrays[k])

    return arrays


def main():
    parser = argparse.ArgumentParser(
        description="3-agent toy with kappa (homophilic network)")
    parser.add_argument("--kappa-min", type=float, default=0.0)
    parser.add_argument("--kappa-max", type=float, default=8.0)
    parser.add_argument("--n-kappa", type=int, default=21)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-realization details for each kappa")
    args = parser.parse_args()

    kappas = np.linspace(args.kappa_min, args.kappa_max, args.n_kappa)

    if args.verbose:
        for k in kappas:
            analyze_kappa(k, rho=args.rho, alpha=args.alpha, beta=args.beta,
                         lam=args.lam, tau=args.tau, verbose=True)
    else:
        sweep_kappa(kappas, rho=args.rho, alpha=args.alpha, beta=args.beta,
                   lam=args.lam, tau=args.tau)


if __name__ == "__main__":
    main()
