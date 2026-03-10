"""
reward.py — Breakthrough probability and individual reward (Stage 3).

    P(breakthrough in C) = h(q_C / tau),   h(x) = 1 - exp(-x)

Individual reward from coalition C:
    pi_C = (1/|C|) * h(q_C / tau) * V

Individual payoff:
    u_i = sum_{C ni i} pi_C  -  (c/2) * R_i^2
"""

import numpy as np
from typing import List

from coalition import Coalition


class RewardModel:
    """Computes breakthrough probabilities, rewards, and payoffs."""

    def __init__(self, tau: float = 1.0, V: float = 1.0, c: float = 1.0):
        """
        Args:
            tau: breakthrough difficulty threshold.
            V: prize value for a breakthrough.
            c: effort cost coefficient.
        """
        self.tau = tau
        self.V = V
        self.c = c

    @staticmethod
    def h(x: float) -> float:
        """Breakthrough probability function: h(x) = 1 - exp(-x)."""
        return 1.0 - np.exp(-x)

    def breakthrough_prob(self, q_C: float) -> float:
        """P(breakthrough in C) = h(q_C / tau)."""
        if self.tau <= 0:
            return 1.0 if q_C > 0 else 0.0
        return self.h(q_C / self.tau)

    def coalition_reward(self, coalition: Coalition, rho: float,
                         alpha: float, beta: float, lam: float = 0.0) -> float:
        """Expected prize from coalition C: h(q_C / tau) * V."""
        q = coalition.effective_output(rho, alpha, beta, lam)
        return self.breakthrough_prob(q) * self.V

    def individual_reward(self, coalition: Coalition, rho: float,
                          alpha: float, beta: float, lam: float = 0.0) -> float:
        """Agent's share of expected prize: (1/|C|) * h(q_C/tau) * V."""
        return self.coalition_reward(coalition, rho, alpha, beta, lam) / coalition.size

    def agent_payoff(self, agent_id: int, coalitions: List[Coalition],
                     rho: float, alpha: float, beta: float,
                     lam: float = 0.0) -> float:
        """Total payoff for agent i: sum of rewards - effort cost.

        u_i = sum_{C ni i} pi_C(q_C, tau)  -  (c/2) * R_i^2
        """
        total_reward = 0.0
        total_effort = 0.0
        for coal in coalitions:
            if agent_id in coal.member_ids:
                total_reward += self.individual_reward(coal, rho, alpha, beta, lam)
                total_effort += coal.members[
                    list(coal.member_ids).index(agent_id)
                ].efforts.get(coal.key, 0.0)
        cost = 0.5 * self.c * total_effort ** 2
        return total_reward - cost

    def total_surplus(self, coalitions: List[Coalition],
                      agents_total_efforts: dict,
                      rho: float, alpha: float, beta: float,
                      lam: float = 0.0) -> float:
        """Social surplus W = sum_C [h(q_C/tau)*V] - sum_i (c/2)*R_i^2.

        Args:
            coalitions: list of active coalitions.
            agents_total_efforts: {agent_id: R_i} total effort per agent.
            rho, alpha, beta, lam: model parameters.
        """
        total_value = sum(
            self.coalition_reward(coal, rho, alpha, beta, lam) for coal in coalitions
        )
        total_cost = sum(
            0.5 * self.c * R_i ** 2 for R_i in agents_total_efforts.values()
        )
        return total_value - total_cost

    def __repr__(self) -> str:
        return f"RewardModel(tau={self.tau}, V={self.V}, c={self.c})"
