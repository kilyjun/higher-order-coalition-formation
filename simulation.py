"""
simulation.py — Orchestrator for the 4-stage simulation pipeline.

    Stage 0: Agents choose angles (theta) — given by config.
    Stage 1: Network realized from homophilic edge probabilities.
    Stage 2: Effort NE / SO solved on active coalitions.
    Stage 3: Breakthroughs realized, welfare computed.

Averages over many network draws to compute E[W].
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from agent import Agent
from network import Network
from coalition import Coalition
from reward import RewardModel
from equilibrium import EquilibriumSolver
from config import ModelParams


@dataclass
class SimulationResult:
    """Stores all outcome measures from a simulation run."""

    params: ModelParams

    # O1: Expected total surplus under NE
    expected_surplus_ne: float = 0.0
    # O1b: Expected total surplus under SO
    expected_surplus_so: float = 0.0
    # O2: Generalist value-add (requires comparison run)
    generalist_value_add: Optional[float] = None
    # O3: Market failure gap
    market_failure_gap: float = 0.0
    # O4: Effort allocation shares {coalition_size -> mean share}
    effort_shares: Dict[int, float] = field(default_factory=dict)
    # O5: Individual expected payoffs {agent_id -> E[u_i]}
    expected_payoffs_ne: Dict[int, float] = field(default_factory=dict)
    # O6: Optimal subsidy (computed externally)
    optimal_subsidy: Optional[float] = None

    # Auxiliary
    mean_breakthroughs_ne: float = 0.0
    mean_breakthroughs_so: float = 0.0
    n_draws: int = 0


class Simulation:
    """Runs the full 4-stage model for a given parameter configuration."""

    def __init__(self, params: ModelParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)
        self.agents: List[Agent] = []
        self.reward_model = RewardModel(tau=params.tau, V=params.V, c=params.c)

    # ------------------------------------------------------------------
    # Stage 0: Create agents
    # ------------------------------------------------------------------

    def _create_agents(self) -> List[Agent]:
        agents = []
        for i, theta in enumerate(self.params.thetas):
            agents.append(Agent(agent_id=i, theta=theta))
        return agents

    # ------------------------------------------------------------------
    # Stage 1 + 2 + 3: Single network draw
    # ------------------------------------------------------------------

    def _run_single_draw(self, agents: List[Agent]) -> Dict:
        """Run stages 1-3 for one network realization.

        Returns dict with keys: welfare_ne, welfare_so, payoffs_ne,
        efforts_ne, breakthroughs_ne, breakthroughs_so.
        """
        # Stage 1: Realize network
        net = Network(agents, kappa=self.params.kappa, rng=self.rng)
        net.realize()

        # Enumerate active coalitions
        max_size = self.params.max_coalition_size or self.params.n_agents
        active_keys = net.active_coalitions(max_size=max_size)

        # Build Coalition objects
        agent_map = {a.agent_id: a for a in agents}
        coalitions = []
        for key in active_keys:
            members = [agent_map[aid] for aid in sorted(key)]
            coalitions.append(Coalition(members, member_ids=key))

        if not coalitions:
            return {
                "welfare_ne": 0.0, "welfare_so": 0.0,
                "payoffs_ne": {a.agent_id: 0.0 for a in agents},
                "efforts_ne": {a.agent_id: 0.0 for a in agents},
                "breakthroughs_ne": 0.0, "breakthroughs_so": 0.0,
            }

        solver = EquilibriumSolver(
            agents, coalitions, self.reward_model,
            rho=self.params.rho, alpha=self.params.alpha,
            beta=self.params.beta, lam=self.params.lam
        )

        # Stage 2a: Nash Equilibrium
        solver.solve_nash()
        welfare_ne = solver.compute_welfare()
        payoffs_ne = solver.compute_payoffs()
        efforts_ne = {a.agent_id: a.total_effort for a in agents}

        # Breakthroughs under NE
        bt_ne = sum(
            self.reward_model.breakthrough_prob(
                c.effective_output(self.params.rho, self.params.alpha, self.params.beta, self.params.lam)
            )
            for c in coalitions
        )

        # Stage 2b: Social Optimum
        solver.solve_social_optimum()
        welfare_so = solver.compute_welfare()

        bt_so = sum(
            self.reward_model.breakthrough_prob(
                c.effective_output(self.params.rho, self.params.alpha, self.params.beta, self.params.lam)
            )
            for c in coalitions
        )

        return {
            "welfare_ne": welfare_ne,
            "welfare_so": welfare_so,
            "payoffs_ne": payoffs_ne,
            "efforts_ne": efforts_ne,
            "breakthroughs_ne": bt_ne,
            "breakthroughs_so": bt_so,
        }

    # ------------------------------------------------------------------
    # Full run: average over network draws
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """Run full simulation: Stage 0 + many draws of Stages 1-3."""
        # Stage 0
        self.agents = self._create_agents()

        n_draws = self.params.n_network_draws
        welfare_ne_acc = 0.0
        welfare_so_acc = 0.0
        bt_ne_acc = 0.0
        bt_so_acc = 0.0
        payoffs_acc = {a.agent_id: 0.0 for a in self.agents}
        effort_by_size = {}  # coalition_size -> total effort across draws

        for draw in range(n_draws):
            # Create fresh agents each draw (to reset efforts)
            agents = self._create_agents()
            result = self._run_single_draw(agents)

            welfare_ne_acc += result["welfare_ne"]
            welfare_so_acc += result["welfare_so"]
            bt_ne_acc += result["breakthroughs_ne"]
            bt_so_acc += result["breakthroughs_so"]
            for aid, pay in result["payoffs_ne"].items():
                payoffs_acc[aid] += pay

            # Track effort allocation by coalition size
            for a in agents:
                for ckey, eff in a.efforts.items():
                    sz = len(ckey)
                    effort_by_size.setdefault(sz, 0.0)
                    effort_by_size[sz] += eff

        # Averages
        res = SimulationResult(params=self.params, n_draws=n_draws)
        res.expected_surplus_ne = welfare_ne_acc / n_draws
        res.expected_surplus_so = welfare_so_acc / n_draws
        res.market_failure_gap = res.expected_surplus_so - res.expected_surplus_ne
        res.mean_breakthroughs_ne = bt_ne_acc / n_draws
        res.mean_breakthroughs_so = bt_so_acc / n_draws
        res.expected_payoffs_ne = {aid: p / n_draws for aid, p in payoffs_acc.items()}

        # Effort shares by coalition size
        total_eff = sum(effort_by_size.values())
        if total_eff > 0:
            res.effort_shares = {sz: e / total_eff for sz, e in sorted(effort_by_size.items())}

        return res

    def __repr__(self) -> str:
        return f"Simulation(κ={self.params.kappa}, τ={self.params.tau}, ρ={self.params.rho})"
