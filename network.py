"""
network.py — Homophilic network formation (Stage 1).

Edge probability:  p_ij = sim_ij^kappa
where sim_ij = cosine similarity of skill vectors.

A coalition C is *active* in realization G iff its members are connected
(a spanning tree exists among them in G).
"""

import numpy as np
from itertools import combinations
from typing import List, Set, FrozenSet, Dict, Tuple

from agent import Agent


class Network:
    """A realized network over a set of agents with homophilic edge formation."""

    def __init__(self, agents: List[Agent], kappa: float, rng: np.random.Generator = None):
        self.agents = agents
        self.n = len(agents)
        self.kappa = kappa
        self.rng = rng or np.random.default_rng()

        # Adjacency matrix (symmetric, no self-loops)
        self.adj: np.ndarray = np.zeros((self.n, self.n), dtype=bool)
        # Edge probabilities (for expected-value computations)
        self.edge_probs: np.ndarray = np.zeros((self.n, self.n))

        self._compute_edge_probs()

    # --- Edge probabilities ---

    def _compute_edge_probs(self) -> None:
        """Compute p_ij = sim_ij^kappa for all pairs."""
        for i, j in combinations(range(self.n), 2):
            sim = self.agents[i].similarity(self.agents[j])
            sim = max(sim, 0.0)  # guard against numerical negatives
            p = sim ** self.kappa if self.kappa > 0 else 1.0
            self.edge_probs[i, j] = p
            self.edge_probs[j, i] = p

    # --- Realization ---

    def realize(self) -> "Network":
        """Draw a random network realization (each edge independently)."""
        self.adj = np.zeros((self.n, self.n), dtype=bool)
        for i, j in combinations(range(self.n), 2):
            if self.rng.random() < self.edge_probs[i, j]:
                self.adj[i, j] = True
                self.adj[j, i] = True
        return self

    def set_complete(self) -> "Network":
        """Set the network to the complete graph (all edges present)."""
        self.adj = np.ones((self.n, self.n), dtype=bool)
        np.fill_diagonal(self.adj, False)
        return self

    # --- Coalition activity ---

    def is_connected(self, members: FrozenSet[int]) -> bool:
        """Check if a subset of agents is connected in the realized network.
        Uses BFS on the subgraph induced by `members`."""
        if len(members) <= 1:
            return True
        members_list = list(members)
        visited = {members_list[0]}
        queue = [members_list[0]]
        while queue:
            node = queue.pop(0)
            for nbr in members_list:
                if nbr not in visited and self.adj[node, nbr]:
                    visited.add(nbr)
                    queue.append(nbr)
        return visited == set(members_list)

    def active_coalitions(self, max_size: int = None) -> List[FrozenSet[int]]:
        """Enumerate all connected subsets (coalitions) of agents in G.

        For small n this is feasible (2^n subsets).
        max_size: if given, only consider coalitions up to this size.
        """
        if max_size is None:
            max_size = self.n
        ids = list(range(self.n))
        active = []
        for size in range(1, max_size + 1):
            for combo in combinations(ids, size):
                fs = frozenset(combo)
                if self.is_connected(fs):
                    active.append(fs)
        return active

    # --- Probability of a coalition being active (analytical) ---

    def coalition_activation_prob(self, members: FrozenSet[int]) -> float:
        """Compute P(coalition C is connected) analytically.

        For |C| = 1: always 1.
        For |C| = 2: p_ij.
        For |C| = 3: uses inclusion-exclusion over spanning trees.
        For |C| > 3: falls back to Monte Carlo estimation.
        """
        members_list = sorted(members)
        k = len(members_list)
        if k == 1:
            return 1.0
        if k == 2:
            i, j = members_list
            return self.edge_probs[i, j]
        if k == 3:
            i, j, m = members_list
            p_ij = self.edge_probs[i, j]
            p_im = self.edge_probs[i, m]
            p_jm = self.edge_probs[j, m]
            # P(connected) = P(all 3 edges) + P(exactly 2 edges in a spanning tree)
            # = 1 - P(isolated node exists)
            # Using inclusion-exclusion:
            # P(disconnected) = P(i isolated) + P(j isolated) + P(m isolated)
            #                 - 2*P(all disconnected)  ... but let's use direct formula
            # P(connected) = p_ij*p_im + p_ij*p_jm + p_im*p_jm - 2*p_ij*p_im*p_jm
            return (p_ij * p_im + p_ij * p_jm + p_im * p_jm
                    - 2 * p_ij * p_im * p_jm)
        # Fallback: Monte Carlo for larger coalitions
        return self._mc_activation_prob(members, n_samples=10000)

    def _mc_activation_prob(self, members: FrozenSet[int], n_samples: int = 10000) -> float:
        """Monte Carlo estimate of P(coalition is connected)."""
        members_list = sorted(members)
        pairs = list(combinations(members_list, 2))
        count = 0
        for _ in range(n_samples):
            # Draw edges
            edges = {(i, j) for i, j in pairs if self.rng.random() < self.edge_probs[i, j]}
            # Check connectivity via BFS
            if self._is_connected_from_edges(members_list, edges):
                count += 1
        return count / n_samples

    @staticmethod
    def _is_connected_from_edges(nodes: List[int], edges: Set[Tuple[int, int]]) -> bool:
        if len(nodes) <= 1:
            return True
        adj = {n: set() for n in nodes}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        visited = {nodes[0]}
        queue = [nodes[0]]
        while queue:
            node = queue.pop(0)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return len(visited) == len(nodes)

    def __repr__(self) -> str:
        edges = int(self.adj.sum()) // 2
        return f"Network(n={self.n}, κ={self.kappa}, edges={edges})"
