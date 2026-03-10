"""
config.py — Parameter defaults and experiment presets.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ModelParams:
    """All model parameters in one place."""

    # --- Stage 0: Agent types ---
    n_agents: int = 3                    # number of agents
    thetas: List[float] = None           # angles for each agent; None = auto-assign
    generalist_present: bool = True      # whether to include a generalist

    # --- Stage 1: Network ---
    kappa: float = 1.0                   # homophily exponent

    # --- Stage 2: Effort & production ---
    rho: float = 0.5                     # breadth-depth weight (1 = pure breadth)
    alpha: float = 0.5                   # diminishing returns on effort
    beta: float = 0.0                    # group-size penalty on marginal returns
    lam: float = 0.0                     # communication friction exponent (lambda)
    c: float = 1.0                       # effort cost coefficient

    # --- Stage 3: Breakthrough ---
    tau: float = 1.0                     # breakthrough difficulty threshold
    V: float = 1.0                       # prize value

    # --- Simulation ---
    n_network_draws: int = 1000          # Monte Carlo draws of network realizations
    max_coalition_size: int = None       # limit coalition enumeration (None = n_agents)
    seed: int = 42                       # random seed

    def __post_init__(self):
        if self.thetas is None:
            self._assign_default_thetas()

    def _assign_default_thetas(self):
        """Default: equal specialist angles + one generalist if present."""
        if self.n_agents == 3 and self.generalist_present:
            # Two specialists (theta=0), one generalist (theta=pi/4)
            self.thetas = [0.0, 0.0, np.pi / 4]
        elif self.n_agents == 3 and not self.generalist_present:
            # Three specialists: two at theta=0, one "mild" specialist
            self.thetas = [0.0, 0.0, 0.0]
        else:
            # General: all specialists except last (generalist if flag set)
            self.thetas = [0.0] * self.n_agents
            if self.generalist_present and self.n_agents > 0:
                self.thetas[-1] = np.pi / 4


# --- Experiment presets ---

BASELINE = ModelParams()

# Sweep over homophily
KAPPA_SWEEP = [
    ModelParams(kappa=k) for k in [0.0, 0.5, 1.0, 2.0, 4.0]
]

# Sweep over threshold difficulty
TAU_SWEEP = [
    ModelParams(tau=t) for t in [0.25, 0.5, 1.0, 2.0, 4.0]
]

# Sweep over breadth-depth
RHO_SWEEP = [
    ModelParams(rho=r) for r in [0.0, 0.25, 0.5, 0.75, 1.0]
]

# Sweep over group-size penalty
BETA_SWEEP = [
    ModelParams(beta=b) for b in [0.0, 0.25, 0.5, 0.75, 1.0]
]

# Sweep over communication friction
LAMBDA_SWEEP = [
    ModelParams(lam=l) for l in [0.0, 0.5, 1.0, 2.0, 4.0]
]

# Full factorial (kappa x tau), with and without generalist
def full_factorial(kappas=(0.5, 1.0, 2.0), taus=(0.5, 1.0, 2.0),
                   rhos=(0.5,), betas=(0.0,), lams=(0.0,)) -> List[ModelParams]:
    """Generate all parameter combinations for systematic experiments."""
    configs = []
    for k in kappas:
        for t in taus:
            for r in rhos:
                for b in betas:
                    for l in lams:
                        for g in [True, False]:
                            configs.append(ModelParams(
                                kappa=k, tau=t, rho=r, beta=b, lam=l,
                                generalist_present=g
                            ))
    return configs
