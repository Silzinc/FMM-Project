from __future__ import annotations

from typing import Callable, Generator, List

import numpy as np

from .cell import FMMCell
from .sample import MassSample
from .tree import FMMTree
from .utils import Mat3x3, Vec3


class GenericSolver:
    """
    Generic solver template for particles systems physics.

    Fields:

        phi: function of r/epsilon representing the potential
            of one "particle" per unit mass (Callable[[Vec3], float])

        grad_phi: function of r/epsilon representing the gradient
            potential of one "particle" per unit mass (Callable[[Vec3], Vec3])

        dt: timestep (float)

        samples: list of mass samples (MassSample)

        epsilon: particle smoothing size (float)

        G: gravitational constant (float)
    """

    def __init__(
        self,
        dt: float,
        epsilon: float,
        samples: List[MassSample],
        phi: Callable[[Vec3], float],
        grad_phi: Callable[[Vec3], Vec3],
        G: float = 1e-1,
    ):
        self.phi = phi
        self.grad_phi = grad_phi
        self.dt = dt
        self.samples = samples
        self.epsilon = epsilon
        self.G = G

    def potential(self, diff: Vec3) -> float:
        """
        Helper function to compute the potential between two points, given their position difference.

        Args:
            diff: position difference between the two points (Vec3)

        Returns:
            potential (float)
        """
        return float(-self.G / np.linalg.norm(diff) * self.phi(diff / self.epsilon))

    def field_intensity(self, diff: Vec3) -> Vec3:
        """
        Helper function to compute the field intensity from one point to another, given their position difference.

        Args:
            diff: position difference between the two points (Vec3)

        Returns:
            field intensity (Vec3)
        """
        return self.G / (np.linalg.norm(diff) ** 2) * self.grad_phi(diff / self.epsilon)

    def update(self):
        """
        Updates the system.
        """
        pass

    # Utilities

    def average_pos(self) -> Vec3:
        """
        Average position of the samples.
        """
        avg: Vec3 = np.zeros(3)
        for s in self.samples:
            avg += s.pos
        return avg / len(self.samples)

    def std_pos(self) -> Vec3:
        """
        Standard deviation of the position of the samples.
        """
        avg_sq: Vec3 = np.zeros(3)
        avg = self.average_pos()
        for s in self.samples:
            avg_sq += (s.pos - avg) ** 2
        return np.sqrt(avg_sq / len(self.samples))

    def pos_divergence(self, lhs: GenericSolver) -> float:
        """
        Position divergence between two solvers.
        """
        div = 0
        for sample1, sample2 in zip(self.samples, lhs.samples):
            div += ((sample1.pos - sample2.pos) ** 2).sum()
        return div / len(self.samples)

    def total_energy(self) -> float:
        """
        Total energy of the system.
        """
        # Kinetic energy
        ke = sum(
            s.mass * np.linalg.norm(s.speed(self.dt)) ** 2 / 2 for s in self.samples
        )
        # Potential energy
        pe = 0
        for s1 in self.samples:
            for s2 in self.samples:
                if s1 is not s2:
                    pe += self.potential(s1.pos - s2.pos) * s1.mass * s2.mass
        return float(ke + pe)


class FMMSolver(GenericSolver):
    """
    Solver for particle systems using the Fast Multipole Multiplication method.
    If the depth of the space tree is chosen to be O(log8(len(samples)),
    the time complexity of the algorithm is expected to be O(len(samples)).

    Fields:

        size: size of the simulation cube (float)

        phi: function of r/epsilon representing the potential
            of one "particle" per unit mass (Callable[[Vec3], float])

        grad_phi: function of r/epsilon representing the gradient
            potential of one "particle" per unit mass (Callable[[Vec3], Vec3])

        hess_phi: function of r/epsilon representing the hessian
            potential of one "particle" per unit mass. Set to None to
            use an order 0 expansion of the inter-cell field. None by default.
            (Callable[[Vec3], Mat3x3] | None)

        dt: timestep (float)

        tree: octree of the volume cells (FMMTree)

        samples: list of mass samples (MassSample)

        epsilon: particle smoothing size (float)

        G: gravitational constant (float, defaults to 0.1)
    """

    def __init__(
        self,
        size: float,
        dt: float,
        samples: List[MassSample],
        depth: int,
        phi: Callable[[Vec3], float],
        grad_phi: Callable[[Vec3], Vec3],
        hess_phi: Callable[[Vec3], Mat3x3] | None = None,
        G: float = 1e-1,
    ):
        self.size = size
        self.phi = phi
        self.grad_phi = grad_phi
        self.hess_phi = hess_phi
        self.dt = dt
        self.tree = FMMTree(depth, size)
        self.samples = samples
        self.epsilon = 4 * size / np.sqrt(len(samples))
        self.G = G

    def field_jacobian(self, diff: Vec3) -> Mat3x3:
        """
        Helper function to compute the field jacobian from one point to another, given their position difference.

        Args:
            diff: position difference between the two points (Vec3)

        Returns:
            field jacobian (Mat3x3)
        """
        return (
            np.zeros((3, 3))
            if self.hess_phi is None
            else (
                self.G
                / (np.linalg.norm(diff) ** 3)
                * self.hess_phi(diff / self.epsilon)
            )
        )

    def update(self):
        """
        Updates the system with the FMM algorithm, with time step dt.

        The algorithm starts by updating the samples' distribution in the tree
        structure, then updates the multipoles, barycenters andfield tensors on each cell,
        and finally computes the field intensity at each point to update the positions
        using the Verlet integration scheme.

        Time complexity:
            O(8^self.tree.depth() + len(self.samples)), which becomes O(len(self.samples))
            if the depth of the tree is chosen to be O(log8(len(samples)).
        """
        self.tree.update(self.samples, self.field_intensity, self.field_jacobian)
        new_poss = np.zeros((len(self.samples), 3))
        index = 0
        depth = self.tree.depth()
        for i in range(2 ** (depth - 1)):
            for j in range(2 ** (depth - 1)):
                for k in range(2 ** (depth - 1)):
                    cell = self.tree[-1][i][j][k]
                    assert cell.samples is not None
                    if len(cell.samples) == 0:
                        continue
                    for sample in cell.samples:
                        grad_potential = np.zeros(3)
                        grad_potential += self.compute_close(cell, sample)
                        grad_potential += self.compute_far(cell, sample)
                        acc = grad_potential
                        new_poss[index] = (
                            2 * sample.pos - sample.prev_pos + acc * self.dt**2
                        )
                        index += 1

        index = 0
        for i in range(2 ** (depth - 1)):
            for j in range(2 ** (depth - 1)):
                for k in range(2 ** (depth - 1)):
                    cell = self.tree[-1][i][j][k]
                    assert cell.samples is not None
                    if len(cell.samples) == 0:
                        continue
                    for sample in cell.samples:
                        sample.prev_pos = sample.pos
                        sample.pos = new_poss[index]
                        index += 1

    def compute_close(self, cell: FMMCell, sample: MassSample) -> Vec3:
        total_field = np.zeros(3)
        for neighbor_cell in cell.direct_neighbors:
            assert neighbor_cell.samples is not None
            if len(neighbor_cell.samples) == 0:
                continue
            for close_sample in neighbor_cell.samples:
                diff = sample.pos - close_sample.pos
                if np.linalg.norm(diff) == 0:
                    continue
                field_intensity = self.field_intensity(diff)
                total_field += close_sample.mass * field_intensity
        return total_field

    def compute_far(self, cell: FMMCell, sample: MassSample) -> Vec3:
        return cell.field_tensor.field + cell.field_tensor.jacobian @ (
            sample.pos - cell.barycenter
        )

    def iter_cells(self, floor: int) -> Generator[FMMCell]:
        if floor >= self.tree.depth():
            raise ValueError(
                f"Asked floor {floor} in a tree of depth {self.tree.depth()}."
                "Floor must be less than the depth of the tree."
            )
        floor %= self.tree.depth()
        for i in range(2**floor):
            for j in range(2**floor):
                for k in range(2**floor):
                    yield self.tree[floor][i][j][k]


class NaiveSolver(GenericSolver):
    """
    Simple quadratic complexity solver for particle systems.
    Used for speed comparison with better algorithms.

    Fields:

        phi: function of r/epsilon representing the potential
            of one "particle" per unit mass (Callable[[Vec3], float])

        grad_phi: function of r/epsilon representing the gradient
            potential of one "particle" per unit mass (Callable[[Vec3], Vec3])

        dt: timestep (float)

        samples: list of mass samples (MassSample)

        epsilon: particle smoothing size (float)

        G: gravitational constant (float)
    """

    def update(self):
        """
        Updates the system with the basic O(len(self.samples)^2) algorithm.
        Used for speed comparison with the enhanced algorithm.
        """
        new_poss = np.zeros((len(self.samples), 3))
        for index, sample1 in enumerate(self.samples):
            acc = np.zeros(3)
            for sample2 in self.samples:
                diff = sample1.pos - sample2.pos
                if np.linalg.norm(diff) == 0:
                    continue
                field_intensity = self.field_intensity(diff)
                acc += sample2.mass * field_intensity
            new_poss[index] = 2 * sample1.pos - sample1.prev_pos + acc * self.dt**2

        for index, sample in enumerate(self.samples):
            sample.prev_pos = sample.pos
            sample.pos = new_poss[index]
