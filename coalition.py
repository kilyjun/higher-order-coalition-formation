"""
coalition.py — Coalition synergy, breadth/depth, friction, and effective output.

For coalition C with members' skill vectors {v_i}:

    Breadth_C = (1/d) * sum_k max_{i in C} v_{ik}
    Depth_C   = (1/|C|) * max_k sum_{i in C} v_{ik}

Synergy (CES aggregator):
    Gamma_C = [ rho * Breadth^sigma + (1-rho) * Depth^sigma ]^{1/sigma}
    where sigma controls substitutability (fixed at 1 => linear for now).

Communication friction (intensive margin):
    phi(C) = (avg_sim_C)^lambda
    where avg_sim_C = mean pairwise cosine similarity within C.
    lambda = 0: no friction.  lambda > 0: diverse teams pay a cost.

Effective output:
    q_C = Gamma_C * phi(C) * (prod_{j in C} r_{jC})^{alpha / |C|^beta}
"""

import numpy as np
from itertools import combinations
from typing import List, FrozenSet

from agent import Agent


class Coalition:
    """A coalition of agents with synergy, friction, and output computation."""

    def __init__(self, members: List[Agent], member_ids: FrozenSet[int] = None):
        self.members = members
        self.member_ids = member_ids or frozenset(a.agent_id for a in members)
        self.size = len(members)

        # Cache skill matrix: (|C| x d)
        self._skill_matrix = np.array([a.skill_vector for a in members])

    @property
    def key(self) -> FrozenSet[int]:
        return self.member_ids

    # --- Collective skill measures ---

    @property
    def breadth(self) -> float:
        """Breadth_C = (1/d) sum_k max_{i in C} v_{ik}."""
        d = self._skill_matrix.shape[1]
        return float(np.mean(np.max(self._skill_matrix, axis=0)))

    @property
    def depth(self) -> float:
        """Depth_C = (1/|C|) max_k sum_{i in C} v_{ik}."""
        col_sums = np.sum(self._skill_matrix, axis=0)
        return float(np.max(col_sums) / self.size)

    def synergy(self, rho: float) -> float:
        """Coalition synergy Gamma_C.

        Gamma_C = rho * Breadth + (1 - rho) * Depth
        (linear CES; can be generalized later)

        Args:
            rho: weight on breadth vs depth. rho=1 => pure breadth-rewarding.
        """
        return rho * self.breadth + (1.0 - rho) * self.depth

    @property
    def avg_similarity(self) -> float:
        """Average pairwise cosine similarity within the coalition.

        avg_sim_C = (1 / C(|C|,2)) * sum_{i<j in C} sim_ij

        For singletons, returns 1.0 (no friction).
        """
        if self.size <= 1:
            return 1.0
        total_sim = 0.0
        n_pairs = 0
        for i, j in combinations(range(self.size), 2):
            total_sim += self.members[i].similarity(self.members[j])
            n_pairs += 1
        return total_sim / n_pairs

    def friction(self, lam: float) -> float:
        """Communication friction phi(C) = (avg_sim_C)^lambda.

        Args:
            lam: friction exponent. 0 = no friction, >0 = diverse teams penalized.
        """
        if lam == 0.0 or self.size <= 1:
            return 1.0
        avg_sim = max(self.avg_similarity, 0.0)  # guard negatives
        return avg_sim ** lam

    def effective_output(self, rho: float, alpha: float, beta: float,
                         lam: float = 0.0) -> float:
        """Effective output q_C = Gamma_C * phi(C) * (prod r_{jC})^{alpha / |C|^beta}.

        Reads effort levels from each agent's efforts dict.

        Args:
            rho: breadth-depth mixing parameter.
            alpha: diminishing returns on effort.
            beta: group-size penalty on marginal returns.
            lam: communication friction exponent.
        """
        gamma = self.synergy(rho)
        phi = self.friction(lam)
        efforts = []
        for a in self.members:
            e = a.efforts.get(self.key, 0.0)
            efforts.append(e)

        efforts = np.array(efforts)
        if np.any(efforts <= 0):
            return 0.0

        exponent = alpha / (self.size ** beta)
        prod_efforts = np.prod(efforts)
        q = gamma * phi * (prod_efforts ** exponent)
        return float(q)

    def __repr__(self) -> str:
        ids = sorted(self.member_ids)
        return f"Coalition({ids}, Γ(ρ=0.5)={self.synergy(0.5):.3f})"
