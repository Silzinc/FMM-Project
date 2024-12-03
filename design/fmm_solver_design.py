from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import numpy.random as rd
import numpy.typing as npt

np.set_printoptions(precision=3, suppress=True)

mu = 1.0
size = 10.0

type Vec3 = npt.NDArray[np.float64]
type Vec4 = npt.NDArray[np.float64]


class MassSample:
    """
    Fields:
      mass: mass of the particle (float)
      position: position of the particle (vec3)
      previous_position: previous position of the particle (for Verlet integration) (vec3)
    Methods:
      speed: speed of the particle (float -> vec3)
    """

    def __init__(self):
        self.mass: float = mu
        self.pos: Vec3 = rd.rand(3) * size - size / 2
        self.prev_pos: Vec3 = np.copy(self.pos)  # + rd.randn(3)

    def speed(self, dt: float) -> Vec3:
        return (self.pos - self.prev_pos) / dt


class FMMCell:
    """
    Fields:
      width: max distance between barycenter and contained particles (float)
      samples: list of mass samples in the cell, None if not leaf (list of MassSample or None)
      barycenter: centre of mass (vec3)
      mass: mass of the cell (float)
      field_tensor: when leaf cell sum of field tensor contributions of all other far cells (vec4)
      neighbors: list of neighboring cells (list of FMMCell)
    """

    def __init__(
        self, centroid: Vec3, size: float, samples: List[MassSample] | None = None
    ):
        self.samples = samples
        self.centroid: Vec3 = centroid
        self.size = size
        self.field_tensor: Vec4 = np.zeros(4)
        # NOTE: The following will be set when the tree is constructed
        self.mass: float = 0
        self.barycenter: Vec3 = np.zeros(3)
        self.interaction_list: List[FMMCell] = []
        self.direct_neighbors: List[FMMCell] = []

    def contains_sample(self, sample: MassSample):
        return np.abs(self.centroid - sample.pos).max() < self.size / 2

    def __str__(self):
        return (
            f"Cell(centroid: {self.centroid}, "
            f"size: {self.size}, "
            f"mass: {self.mass:.2f}, "
            f"field_tensor: {self.field_tensor})"
        )


class FMMTree:
    def __init__(self, depth: int, size: float) -> None:
        """
        Initializes the FMMTree with the given depth and size.
        Constructs the tree structure and initializes the direct_neighbors and interaction_list fields.
        These are the only fields that are initialized, as they do not depend on the samples but only on the structure.

        Args:
            depth: depth of the tree
            size: size of the simulation cube (the cells' centroids are placed so that the cube is centered at the origin)

        Time complexity:
            O(8^depth) given the depth of the tree, which becomes O(len(samples)) if depth = O(log8(len(samples)).
        """
        # Initialize the data
        self.data = [
            [
                [
                    [
                        FMMCell(
                            np.array(
                                [
                                    size * ((i + 0.5) / 2**l - 0.5),
                                    size * ((j + 0.5) / 2**l - 0.5),
                                    size * ((k + 0.5) / 2**l - 0.5),
                                ]
                            ),
                            size / 2**l,
                        )
                        for k in range(2**l)
                    ]
                    for j in range(2**l)
                ]
                for i in range(2**l)
            ]
            for l in range(depth)
        ]

        # Construct direct_neighbors and interaction_list
        for l in range(depth):
            width = 2**l
            for i in range(width):
                for j in range(width):
                    for k in range(width):
                        cell = self[l][i][j][k]
                        for ni in [i - 1, i, i + 1]:
                            for nj in [j - 1, j, j + 1]:
                                for nk in [k - 1, k, k + 1]:
                                    if (
                                        0 <= ni < width
                                        and 0 <= nj < width
                                        and 0 <= nk < width
                                    ):
                                        cell.direct_neighbors.append(
                                            self[l][ni][nj][nk]
                                        )
                        if l > 0:
                            # Parent indices
                            pi, pj, pk = i // 2, j // 2, k // 2
                            for ni in range(max(0, 2 * pi - 2), min(width, 2 * pi + 4)):
                                for nj in range(
                                    max(0, 2 * pj - 2), min(width, 2 * pj + 4)
                                ):
                                    for nk in range(
                                        max(0, 2 * pk - 2), min(width, 2 * pk + 4)
                                    ):
                                        near_child = self[l][ni][nj][nk]
                                        if (
                                            max(
                                                abs(ni - i),
                                                abs(nj - j),
                                                abs(nk - k),
                                            )
                                            > 1
                                        ):
                                            cell.interaction_list.append(near_child)

    def depth(self):
        return len(self.data)

    def tree_size(self) -> int:
        # There are 2^(3k) cells in the kth grid of the list
        # The total is sum(l = 0 -> depth - 1) of 8^k, i.e. (8^depth - 1) / 7
        return (8 ** self.depth() - 1) // 7

    def space_size(self) -> float:
        return self[0][0][0][0].size

    def __getitem__(self, i: int) -> List[List[List[FMMCell]]]:
        return self.data[i]

    def __setitem__(self, i: int, value: List[List[List[FMMCell]]]):
        self.data[i] = value

    def pretty_print(self) -> None:
        def aux(i: int, j: int, k: int, l: int) -> None:
            if l == self.depth():
                print("  " * l + str(self[i][j][k]))
            else:
                print(" " * l + "FMMTree{")
                i2 = 2 * i
                j2 = 2 * j
                k2 = 2 * k
                for incr1 in [0, 1]:
                    for incr2 in [0, 1]:
                        for incr3 in [0, 1]:
                            aux(i2 + incr1, j2 + incr2, k2 + incr3, l + 1)
                print(" " * l + "}")

        aux(0, 0, 0, 0)

    def get_leaf_from_pos(self, pos: Vec3) -> FMMCell:
        l = self.depth() - 1
        size = self.space_size()
        i = int((pos[0] / size + 0.5) * 2**l)
        j = int((pos[1] / size + 0.5) * 2**l)
        k = int((pos[2] / size + 0.5) * 2**l)
        return self[l][min(len(self[l]) - 1, max(0, i))][
            min(len(self[l]) - 1, max(0, j))
        ][min(len(self[l]) - 1, max(0, k))]

    def update(self, samples: List[MassSample], field: Callable[[Vec3], Vec3]) -> None:
        """
        Updates the tree's content with the given samples and field function.
        The structure itself does not change, but the mass, barycenter, field tensors
        and distribution of samples in the leaves are updated.

        Args:
            samples: list of mass samples
            field: function of the distance between two points representing the field between them per unit mass

        Returns:
            None

        Time complexity:
            O(8^depth) given the depth of the tree, which becomes O(len(samples)) if depth = O(log8(len(samples)).
        """

        # Clear the tree leaf cells
        for i in range(2 ** (self.depth() - 1)):
            for j in range(2 ** (self.depth() - 1)):
                for k in range(2 ** (self.depth() - 1)):
                    leaf = self[-1][i][j][k]
                    if leaf.samples is not None:
                        del leaf.samples[:]
                    else:
                        leaf.samples = []
                    leaf.mass = 0
                    leaf.barycenter = np.zeros(3)

        # Populate the leaves and compute their masses and barycenter
        for sample in samples:
            leaf = self.get_leaf_from_pos(sample.pos)
            assert leaf.samples is not None
            leaf.samples.append(sample)
            leaf.mass += sample.mass
            leaf.barycenter += sample.mass * sample.pos

        for i in range(2 ** (self.depth() - 1)):
            for j in range(2 ** (self.depth() - 1)):
                for k in range(2 ** (self.depth() - 1)):
                    cell = self[self.depth() - 1][i][j][k]
                    if cell.samples is not None and len(cell.samples) != 0:
                        cell.barycenter /= cell.mass

        # Propagate mass and barycenter upward.
        # Unintuitively with the 7 nested for loops, this is done
        # in O(len(samples)) if depth is chosen to have as many
        # leaves as there are samples.
        for l in range(self.depth() - 2, -1, -1):
            for i in range(2**l):
                for j in range(2**l):
                    for k in range(2**l):
                        cell = self[l][i][j][k]
                        cell.mass = 0
                        cell.barycenter = np.zeros(3)
                        for ci in [2 * i, 2 * i + 1]:
                            for cj in [2 * j, 2 * j + 1]:
                                for ck in [2 * k, 2 * k + 1]:
                                    child = self[l + 1][ci][cj][ck]
                                    cell.mass += child.mass
                                    cell.barycenter += child.mass * child.barycenter
                        if cell.mass != 0:
                            cell.barycenter /= cell.mass

        # Compute the field tensors. First, compute the contributions of the interaction lists.
        for l in range(self.depth()):
            for i in range(2**l):
                for j in range(2**l):
                    for k in range(2**l):
                        cell = self[l][i][j][k]
                        if cell.samples is not None and len(cell.samples) == 0:
                            continue
                        cell.field_tensor = np.zeros(4)
                        for neighbor in cell.interaction_list:
                            diff = cell.barycenter - neighbor.barycenter
                            field_intensity = field(diff)
                            cell.field_tensor[1:] += neighbor.mass * field_intensity
        # Then, propagate the field tensors downward.
        for l in range(self.depth() - 1):
            for i in range(2**l):
                for j in range(2**l):
                    for k in range(2**l):
                        cell = self[l][i][j][k]
                        for ci in [2 * i, 2 * i + 1]:
                            for cj in [2 * j, 2 * j + 1]:
                                for ck in [2 * k, 2 * k + 1]:
                                    child = self[l + 1][ci][cj][ck]
                                    child.field_tensor[1:] += cell.field_tensor[1:]


class FMMSolver:
    """
    Fields:
        size: size of the simulation cube (float)
        phi: function of r/epsilon representing the potential
            of one "particle" (Callable[[float], float])
        grad_phi: function of r/epsilon representing the gradient
            potential of one "particle" (Callable[[float], float])
        one "particle" per mass unit (Callable[[float], float])
        dt: timestep (float)
        tree: octree of the volume cells (OctTree)
        samples: list of mass samples (MassSample)
        epsilon: particle smoothing size (float)
        n_max: max number of particles per leaf cell (int)
    """

    def __init__(
        self,
        size: float,
        phi: Callable[[float], float],
        grad_phi: Callable[[float], float],
        dt: float,
        samples: List[MassSample],
        depth: int,
    ):
        self.size = size
        self.phi = phi
        self.grad_phi = grad_phi
        self.dt = dt
        self.tree = FMMTree(depth, size)
        self.samples = samples
        self.epsilon = 4 * size / np.sqrt(len(samples))
        self.G = 1e-1

    def potential(self, diff: Vec3) -> float:
        """
        Helper function to compute the potential between two points, given their position difference.

        Args:
            diff: position difference between the two points (Vec3)

        Returns:
            potential (float)
        """
        return (
            -self.G
            / np.linalg.norm(diff)
            * self.phi(np.linalg.norm(diff) / self.epsilon)
        )

    def field_intensity(self, diff: Vec3) -> Vec3:
        """
        Helper function to compute the field intensity from one point to another, given their position difference.

        Args:
            diff: position difference between the two points (Vec3)

        Returns:
            field intensity (Vec3)
        """
        return (
            -self.G
            * diff
            / (np.linalg.norm(diff) ** 3)
            * self.grad_phi(np.linalg.norm(diff) / self.epsilon)
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
        self.tree.update(self.samples, self.field_intensity)
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
                        grad_potential += self.compute_far(cell)
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

    def compute_far(self, cell: FMMCell) -> Vec3:
        return cell.field_tensor[1:]

    def naive_update(self):
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

    # Utilities

    def average_pos(self) -> Vec3:
        avg: Vec3 = np.zeros(3)
        for s in self.samples:
            avg += s.pos
        return avg / len(self.samples)

    def std_pos(self) -> Vec3:
        avg_sq: Vec3 = np.zeros(3)
        avg = self.average_pos()
        for s in self.samples:
            avg_sq += (s.pos - avg) ** 2
        return np.sqrt(avg_sq / len(self.samples))

    def pos_divergence(self, lhs: FMMSolver) -> float:
        div = 0
        for sample1, sample2 in zip(self.samples, lhs.samples):
            div += ((sample1.pos - sample2.pos) ** 2).sum()
        return div / len(self.samples)

    def total_energy(self) -> float:
        # Kinetic energy
        ke = sum(
            s.mass * np.linalg.norm(s.speed(self.dt)) ** 2 / 2 for s in self.samples
        )
        # Potential energy
        pe = 0
        for s1 in self.samples:
            for s2 in self.samples:
                if s1 is not s2:
                    pe += self.potential(s1.pos - s2.pos)
        return ke + pe
