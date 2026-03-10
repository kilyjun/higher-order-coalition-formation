"""
agent.py — Agent with endogenous type choice on the unit circle.

Each agent i chooses angle theta_i in [0, pi/4]:
    v_i(theta_i) = (cos(theta_i), sin(theta_i)),  ||v_i|| = 1

    theta = 0     -> specialist  v = (1, 0)
    theta = pi/4  -> generalist  v = (1/sqrt(2), 1/sqrt(2))
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Agent:
    """A single agent who chooses a skill-vector angle and allocates effort."""

    agent_id: int
    theta: float = 0.0  # angle in [0, pi/4]; 0 = specialist, pi/4 = generalist

    # Effort allocations: coalition_key -> effort level (set during equilibrium solve)
    efforts: Dict[frozenset, float] = field(default_factory=dict)

    # --- Skill vector ---

    @property
    def skill_vector(self) -> np.ndarray:
        """Unit-norm skill vector on the 2D unit circle."""
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    @property
    def is_generalist(self) -> bool:
        return self.theta > np.pi / 8  # midpoint heuristic

    # --- Solo metrics ---

    @property
    def solo_depth(self) -> float:
        """Depth_{i} = max_k v_{ik}  (for a singleton)."""
        return float(np.max(self.skill_vector))

    @property
    def solo_breadth(self) -> float:
        """Breadth_{i} = (1/d) sum_k max_{j in {i}} v_{jk} = mean of skills."""
        return float(np.mean(self.skill_vector))

    # --- Effort accounting ---

    @property
    def total_effort(self) -> float:
        """R_i = sum_C r_{iC}."""
        return sum(self.efforts.values())

    def effort_cost(self, c: float = 1.0) -> float:
        """Quadratic effort cost: (c/2) * R_i^2."""
        return 0.5 * c * self.total_effort ** 2

    def reset_efforts(self) -> None:
        self.efforts = {}

    # --- Similarity ---

    def similarity(self, other: "Agent") -> float:
        """Cosine similarity between two agents' skill vectors."""
        v_i = self.skill_vector
        v_j = other.skill_vector
        # Both are unit vectors, so dot product = cosine similarity
        return float(np.dot(v_i, v_j))

    def __repr__(self) -> str:
        kind = "G" if self.is_generalist else "S"
        return f"Agent({self.agent_id}, θ={self.theta:.3f}, {kind})"
