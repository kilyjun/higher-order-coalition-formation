"""
equilibrium.py — Nash Equilibrium and Social Optimum solvers (Stage 2).

Given a realized network G and active coalitions, solve for:
  - Nash Equilibrium: each agent maximizes own payoff u_i.
  - Social Optimum:   a planner maximizes total surplus W.

Both are solved numerically via scipy.optimize.
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple, FrozenSet

from agent import Agent
from coalition import Coalition
from reward import RewardModel


class EquilibriumSolver:
    """Solves for Nash Equilibrium and Social Optimum effort profiles."""

    def __init__(self, agents: List[Agent], coalitions: List[Coalition],
                 reward_model: RewardModel, rho: float, alpha: float,
                 beta: float, lam: float = 0.0):
        self.agents = agents
        self.coalitions = coalitions
        self.reward = reward_model
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.n_agents = len(agents)

        # Build index: agent_id -> list of coalitions they belong to
        self.agent_coalitions: Dict[int, List[Coalition]] = {
            a.agent_id: [] for a in agents
        }
        for coal in coalitions:
            for aid in coal.member_ids:
                self.agent_coalitions[aid].append(coal)

        # Build variable index: maps (agent_id, coalition_key) -> flat index
        self.var_index: Dict[Tuple[int, FrozenSet[int]], int] = {}
        idx = 0
        for coal in coalitions:
            for aid in sorted(coal.member_ids):
                self.var_index[(aid, coal.key)] = idx
                idx += 1
        self.n_vars = idx

    def _unpack_efforts(self, x: np.ndarray) -> None:
        """Write flat vector x into agents' effort dicts."""
        for a in self.agents:
            a.reset_efforts()
        for (aid, ckey), idx in self.var_index.items():
            agent = self.agents[aid]
            agent.efforts[ckey] = max(x[idx], 0.0)

    def _pack_efforts(self) -> np.ndarray:
        """Read agents' effort dicts into a flat vector."""
        x = np.zeros(self.n_vars)
        for (aid, ckey), idx in self.var_index.items():
            x[idx] = self.agents[aid].efforts.get(ckey, 0.0)
        return x

    # ------------------------------------------------------------------
    # Nash Equilibrium (best-response iteration)
    # ------------------------------------------------------------------

    def solve_nash(self, max_iter: int = 200, tol: float = 1e-8,
                   init: np.ndarray = None) -> np.ndarray:
        """Solve NE via iterated best response.

        Each agent optimizes their own effort variables, holding others fixed.
        Repeats until convergence.

        Returns:
            Flat vector of equilibrium efforts.
        """
        if self.n_vars == 0:
            return np.array([])

        x = init if init is not None else np.full(self.n_vars, 0.1)

        for iteration in range(max_iter):
            x_old = x.copy()

            for agent in self.agents:
                aid = agent.agent_id
                # Indices for this agent's effort variables
                my_vars = [(ckey, self.var_index[(aid, ckey)])
                           for ckey in [c.key for c in self.agent_coalitions[aid]]
                           if (aid, ckey) in self.var_index]
                if not my_vars:
                    continue
                my_indices = [idx for _, idx in my_vars]

                def neg_payoff(e_i, agent_id=aid, indices=my_indices):
                    """Negative payoff for agent (to minimize)."""
                    x_tmp = x.copy()
                    for k, flat_idx in enumerate(indices):
                        x_tmp[flat_idx] = max(e_i[k], 0.0)
                    self._unpack_efforts(x_tmp)
                    payoff = self.reward.agent_payoff(
                        agent_id, self.coalitions, self.rho, self.alpha,
                        self.beta, self.lam
                    )
                    return -payoff

                e0 = np.array([x[idx] for idx in my_indices])
                bounds = [(0, None)] * len(my_indices)
                res = minimize(neg_payoff, e0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 100, "ftol": 1e-12})
                for k, flat_idx in enumerate(my_indices):
                    x[flat_idx] = max(res.x[k], 0.0)

            if np.max(np.abs(x - x_old)) < tol:
                break

        self._unpack_efforts(x)
        return x

    # ------------------------------------------------------------------
    # Social Optimum (joint optimization)
    # ------------------------------------------------------------------

    def solve_social_optimum(self, init: np.ndarray = None) -> np.ndarray:
        """Solve for the social planner's optimal effort profile.

        Maximizes W = sum_C [h(q_C/tau)*V] - sum_i (c/2)*R_i^2.

        Returns:
            Flat vector of socially optimal efforts.
        """
        if self.n_vars == 0:
            return np.array([])

        def neg_surplus(x):
            self._unpack_efforts(x)
            agents_R = {a.agent_id: a.total_effort for a in self.agents}
            W = self.reward.total_surplus(
                self.coalitions, agents_R, self.rho, self.alpha,
                self.beta, self.lam
            )
            return -W

        x0 = init if init is not None else np.full(self.n_vars, 0.1)
        bounds = [(0, None)] * self.n_vars
        res = minimize(neg_surplus, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500, "ftol": 1e-12})

        self._unpack_efforts(res.x)
        return res.x

    # ------------------------------------------------------------------
    # Convenience: compute welfare under NE and SO
    # ------------------------------------------------------------------

    def compute_welfare(self) -> float:
        """Compute total surplus W for the current effort profile."""
        agents_R = {a.agent_id: a.total_effort for a in self.agents}
        return self.reward.total_surplus(
            self.coalitions, agents_R, self.rho, self.alpha, self.beta, self.lam
        )

    def compute_payoffs(self) -> Dict[int, float]:
        """Compute individual payoffs for all agents under current efforts."""
        return {
            a.agent_id: self.reward.agent_payoff(
                a.agent_id, self.coalitions, self.rho, self.alpha,
                self.beta, self.lam
            )
            for a in self.agents
        }

    def __repr__(self) -> str:
        return f"EquilibriumSolver(agents={self.n_agents}, coalitions={len(self.coalitions)}, vars={self.n_vars})"
