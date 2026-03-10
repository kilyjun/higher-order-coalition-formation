"""
main.py — Entry point for running coalition formation experiments.

Usage:
    python main.py                     # run baseline 3-agent case
    python main.py --sweep kappa       # sweep over homophily
    python main.py --sweep tau         # sweep over threshold
    python main.py --sweep rho         # sweep over breadth-depth
    python main.py --sweep factorial   # full kappa x tau factorial
    python main.py --draws 500         # number of network Monte Carlo draws
"""

import argparse
import time
import numpy as np

from config import (
    ModelParams, BASELINE, KAPPA_SWEEP, TAU_SWEEP,
    RHO_SWEEP, BETA_SWEEP, LAMBDA_SWEEP, full_factorial,
)
from simulation import Simulation, SimulationResult


def print_result(res: SimulationResult) -> None:
    """Pretty-print a single simulation result."""
    p = res.params
    tag = "w/ G" if p.generalist_present else "w/o G"
    print(f"  [{tag}] kappa={p.kappa:.1f}  tau={p.tau:.1f}  rho={p.rho:.2f}  beta={p.beta:.2f}  lam={p.lam:.2f}")
    print(f"    E[W_NE] = {res.expected_surplus_ne:+.4f}   "
          f"E[W_SO] = {res.expected_surplus_so:+.4f}   "
          f"Gap = {res.market_failure_gap:+.4f}")
    print(f"    E[BT_NE] = {res.mean_breakthroughs_ne:.4f}   "
          f"E[BT_SO] = {res.mean_breakthroughs_so:.4f}")
    payoffs_str = "  ".join(
        f"u_{aid}={pay:+.4f}" for aid, pay in sorted(res.expected_payoffs_ne.items())
    )
    print(f"    Payoffs: {payoffs_str}")
    if res.effort_shares:
        shares_str = "  ".join(
            f"size-{sz}={sh:.1%}" for sz, sh in res.effort_shares.items()
        )
        print(f"    Effort shares: {shares_str}")
    print()


def run_comparison(params_with: ModelParams, params_without: ModelParams) -> None:
    """Run with/without generalist and compute value-add."""
    sim_with = Simulation(params_with)
    sim_without = Simulation(params_without)

    res_with = sim_with.run()
    res_without = sim_without.run()

    delta = res_with.expected_surplus_ne - res_without.expected_surplus_ne
    res_with.generalist_value_add = delta

    print_result(res_with)
    print_result(res_without)
    print(f"  >> Generalist value-add (dW_NE) = {delta:+.4f}")
    print(f"  >> Market failure gap (w/ G)     = {res_with.market_failure_gap:+.4f}")
    print("-" * 60)


def run_sweep(name: str, configs: list, n_draws: int) -> None:
    """Run a parameter sweep, comparing with/without generalist."""
    print(f"\n{'='*60}")
    print(f"  SWEEP: {name}")
    print(f"{'='*60}\n")

    for cfg in configs:
        cfg.n_network_draws = n_draws

    # Group by parameter values (with vs without generalist)
    seen = {}
    for cfg in configs:
        key = (cfg.kappa, cfg.tau, cfg.rho, cfg.beta, cfg.lam)
        seen.setdefault(key, {})[cfg.generalist_present] = cfg

    for key, variants in sorted(seen.items()):
        if True in variants and False in variants:
            run_comparison(variants[True], variants[False])
        else:
            for cfg in variants.values():
                sim = Simulation(cfg)
                print_result(sim.run())


def main():
    parser = argparse.ArgumentParser(description="Coalition Formation Simulation")
    parser.add_argument("--sweep", type=str, default=None,
                        choices=["kappa", "tau", "rho", "beta", "lam", "factorial"],
                        help="Parameter sweep to run")
    parser.add_argument("--draws", type=int, default=200,
                        help="Number of network Monte Carlo draws")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.time()

    if args.sweep is None:
        # Baseline comparison: with vs without generalist
        print("\n" + "=" * 60)
        print("  BASELINE: 3 agents (2 specialists + 1 generalist vs 3 specialists)")
        print("=" * 60 + "\n")
        p_with = ModelParams(generalist_present=True, n_network_draws=args.draws, seed=args.seed)
        p_without = ModelParams(generalist_present=False, n_network_draws=args.draws, seed=args.seed)
        run_comparison(p_with, p_without)

    elif args.sweep == "kappa":
        configs = []
        for k in [0.0, 0.5, 1.0, 2.0, 4.0]:
            configs.append(ModelParams(kappa=k, generalist_present=True, seed=args.seed))
            configs.append(ModelParams(kappa=k, generalist_present=False, seed=args.seed))
        run_sweep("Homophily (kappa)", configs, args.draws)

    elif args.sweep == "tau":
        configs = []
        for t in [0.25, 0.5, 1.0, 2.0, 4.0]:
            configs.append(ModelParams(tau=t, generalist_present=True, seed=args.seed))
            configs.append(ModelParams(tau=t, generalist_present=False, seed=args.seed))
        run_sweep("Threshold (tau)", configs, args.draws)

    elif args.sweep == "rho":
        configs = []
        for r in [0.0, 0.25, 0.5, 0.75, 1.0]:
            configs.append(ModelParams(rho=r, generalist_present=True, seed=args.seed))
            configs.append(ModelParams(rho=r, generalist_present=False, seed=args.seed))
        run_sweep("Breadth-Depth (rho)", configs, args.draws)

    elif args.sweep == "beta":
        configs = []
        for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
            configs.append(ModelParams(beta=b, generalist_present=True, seed=args.seed))
            configs.append(ModelParams(beta=b, generalist_present=False, seed=args.seed))
        run_sweep("Group-size penalty (beta)", configs, args.draws)

    elif args.sweep == "lam":
        configs = []
        for l in [0.0, 0.5, 1.0, 2.0, 4.0]:
            configs.append(ModelParams(lam=l, generalist_present=True, seed=args.seed))
            configs.append(ModelParams(lam=l, generalist_present=False, seed=args.seed))
        run_sweep("Communication friction (lambda)", configs, args.draws)

    elif args.sweep == "factorial":
        configs = full_factorial()
        run_sweep("Full Factorial (kappa x tau)", configs, args.draws)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
