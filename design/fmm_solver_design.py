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

    def __init__(self, samples: List[MassSample], centroid: Vec3, size: float):
        self.samples = samples
        self.centroid: Vec3 = centroid
        self.size = size
        self.field_tensor: Vec4 = np.zeros(4)
        # NOTE: The following will be set when the tree is constructed
        self.mass: float = 0
        self.barycenter: Vec3 = np.zeros(3)
        self.extension: float = 0
        self.neighbors: List[FMMCell] = []

    def contains_sample(self, sample: MassSample):
        return np.abs(self.centroid - sample.pos).max() < self.size / 2

    def __str__(self):
        return (
            f"Cell(centroid: {self.centroid}, "
            f"size: {self.size}, "
            f"mass: {self.mass:.2f}, "
            f"field_tensor: {self.field_tensor}, "
            f"extent: {self.extension:.2f})"
        )


T = TypeVar("T")


class OctTree(Generic[T]):
    """
    Fields:
        children: children of the node (list of 8 OctTree | None)
        value: value of the node
        parent: parent of the node (OctTree | None)
    """

    def __init__(self):
        self.value: Optional[T] = None
        self.children: Optional[List[OctTree[T]]] = None
        self.parent: Optional[OctTree[T]] = None

    def make_children(self) -> None:
        self.children = [OctTree[T]() for _ in range(8)]
        for child in self.children:
            child.parent = self

    def pretty_print(self, indent: int = 0) -> None:
        print("  " * indent + "OctTree{")
        if self.children is None:
            print("  " * (indent + 2) + str(self.value))
        else:
            for child in self.children:
                child.pretty_print(indent + 1)
        print("  " * indent + "}")


class FMMSolver:
    """
    Fields:
        size: size of the simulation cube (float)
        gradphi: function of r/epsilon representing the gradient
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
        grad_phi: Callable[[float], float],
        dt: float,
        samples: List[MassSample],
        n_max: int,
    ):
        self.size = size
        self.grad_phi = grad_phi
        self.dt = dt
        self.tree: OctTree[FMMCell] = OctTree()
        self.samples = samples
        self.epsilon = 4 * size / np.sqrt(len(samples))
        self.n_max = n_max
        self.G = 1e-2

        self.leaves_cells: List[OctTree[FMMCell]] = []

    def make_tree(self) -> None:
        self.tree.value = FMMCell(self.samples, np.zeros(3), self.size)
        tree_stack = [self.tree]
        while len(tree_stack) != 0:
            tree = tree_stack.pop()
            cell = tree.value
            assert cell is not None

            if len(cell.samples) <= self.n_max:
                self.leaves_cells.append(tree)
                continue

            new_size = cell.size / 2

            new_cells: List[FMMCell] = []
            for i1 in [-1, 1]:
                for i2 in [-1, 1]:
                    for i3 in [-1, 1]:
                        new_cell = FMMCell(
                            [],
                            cell.centroid + np.array([i1, i2, i3]) * new_size / 2,
                            new_size,
                        )
                        new_cells.append(new_cell)

            for sample in cell.samples:
                for new_cell in new_cells:
                    if new_cell.contains_sample(sample):
                        new_cell.samples.append(sample)

            tree.make_children()
            assert tree.children is not None
            for child, cell in zip(tree.children, new_cells):
                child.value = cell
                tree_stack.append(child)

    def compute_multipole_extension_barycenter(self) -> None:
        cell_stack = self.leaves_cells.copy()
        visited = set()
        index = 0
        while index < len(cell_stack):
            tree = cell_stack[index]
            index += 1
            cell = tree.value
            assert cell is not None
            if cell in visited:
                # In this case, the calculation has already been done
                continue
            visited.add(cell)

            if tree.children is None:
                if len(cell.samples) == 0:
                    continue
                # Then the cell is a non empty leaf
                for sample in cell.samples:
                    cell.mass += sample.mass
                    cell.barycenter += sample.mass * sample.pos
                cell.barycenter /= cell.mass
                for sample in cell.samples:
                    cell.extension = max(
                        cell.extension, np.linalg.norm(sample.pos - cell.barycenter)
                    )

            else:
                # Then the cell is a non empty parent
                for child in tree.children:
                    child_cell = child.value
                    assert child_cell is not None
                    cell.mass += child_cell.mass
                    cell.barycenter += child_cell.mass * child_cell.barycenter
                cell.barycenter /= cell.mass
                for child in tree.children:
                    child_cell = child.value
                    assert child_cell is not None
                    cell.extension = max(
                        cell.extension,
                        child_cell.extension
                        + np.linalg.norm(child_cell.barycenter - cell.barycenter),
                    )

            # Add parent to the working stack if any
            if tree.parent is not None:
                cell_stack.append(tree.parent)

    def clear_field_tensors_neighbors(self) -> None:
        def __clear(tree: OctTree[FMMCell]) -> None:
            cell = tree.value
            assert cell is not None
            cell.field_tensor = np.zeros(4)
            cell.neighbors = []
            if tree.children is not None:
                for child in tree.children:
                    __clear(child)

        __clear(self.tree)

    def compute_field_tensors_neighbors(self) -> None:
        self.clear_field_tensors_neighbors()
        # Deque containing pairs of cells to make interact with each other
        cell_pairs: Deque[Tuple[OctTree[FMMCell], OctTree[FMMCell]]] = deque()
        cell_pairs.append((self.tree, self.tree))

        while len(cell_pairs) != 0:
            (tree1, tree2) = cell_pairs.popleft()
            cell1 = tree1.value
            cell2 = tree2.value
            assert cell1 is not None
            assert cell2 is not None
            # If the extents of the cells overlap, i.e. if w1 + w2 >= |z1 - z2|,
            # subdivide the biggest cell into 8 children for this interaction
            if cell1.extension + cell2.extension >= np.linalg.norm(
                cell1.barycenter - cell2.barycenter
            ):
                if max(len(cell1.samples), len(cell2.samples)) <= self.n_max:
                    # If the cells close and are leaves, add them to each other's neighbors
                    cell1.neighbors.append(cell2)
                    if cell1 is not cell2:
                        cell2.neighbors.append(cell1)

                elif len(cell1.samples) >= len(cell2.samples):
                    assert tree1.children is not None
                    for child in tree1.children:
                        cell_pairs.append((child, tree2))

                else:
                    assert tree2.children is not None
                    for child in tree2.children:
                        cell_pairs.append((tree1, child))

            else:
                # Otherwise, compute the field tensor
                # Under a order-1 approximation of the potential, the contribution of cell1 to cell2's tensor is
                # [m1 * phi(z1 - z2), m1 * grad(phi)(z1 - z2)]
                # Actually, phi is not necessary and therefore not computed.

                diff = cell2.barycenter - cell1.barycenter

                field_intensity = (
                    -self.G
                    * diff
                    / (np.linalg.norm(diff) ** 3)
                    * self.grad_phi(np.linalg.norm(diff) / self.epsilon)
                )
                cell2.field_tensor[1:] += cell1.mass * field_intensity
                cell1.field_tensor[1:] += -cell2.mass * field_intensity

                # cell1.field_tensor[0] += cell2.mass * phi_val

        # Last downward pass to propagate the field tensors to the leaves
        cell_stack = [self.tree]
        while len(cell_stack) != 0:
            tree = cell_stack.pop()
            cell = tree.value
            assert cell is not None
            if tree.children is None:
                continue
            for child in tree.children:
                child_cell = child.value
                assert child_cell is not None
                # barycenter_diff = child_cell.barycenter - cell.barycenter
                # child_cell.field_tensor[0] += (
                #     cell.field_tensor[1:] @ barycenter_diff + cell.field_tensor[0]
                # )
                child_cell.field_tensor[1:] += cell.field_tensor[1:]
                cell_stack.append(child)

    def update(self):
        new_poss = np.zeros((len(self.samples), 3))
        index = 0
        for tree in self.leaves_cells:
            cell = tree.value
            assert cell is not None
            for sample in cell.samples:
                grad_potential = np.zeros(3)
                grad_potential += self.compute_close(cell, sample)
                grad_potential += self.compute_far(cell)
                acc = grad_potential
                new_poss[index] = 2 * sample.pos - sample.prev_pos + acc * self.dt**2
                index += 1
        index = 0
        for tree in self.leaves_cells:
            cell = tree.value
            assert cell is not None
            for sample in cell.samples:
                sample.prev_pos = sample.pos
                sample.pos = new_poss[index]
                index += 1

    def naive_update(self):
        """
        Updates the system with the basic O(n^2) algorithm.
        Used for speed comparison with the enhanced algorithm.
        """
        new_poss = np.zeros((len(self.samples), 3))
        for index, sample1 in enumerate(self.samples):
            acc = np.zeros(3)
            for sample2 in self.samples:
                diff = sample1.pos - sample2.pos
                if np.linalg.norm(diff) == 0:
                    continue
                acc += sample2.mass * (
                    -self.G
                    * diff
                    / (np.linalg.norm(diff) ** 3)
                    * self.grad_phi(np.linalg.norm(diff) / self.epsilon)
                )
            new_poss[index] = 2 * sample1.pos - sample1.prev_pos + acc * self.dt**2

        for index, sample in enumerate(self.samples):
            sample.prev_pos = sample.pos
            sample.pos = new_poss[index]

    def compute_close(self, cell: FMMCell, sample: MassSample) -> Vec3:
        total_field = np.zeros(3)
        for neighbor_cell in cell.neighbors:
            for close_sample in neighbor_cell.samples:
                diff = sample.pos - close_sample.pos
                if np.linalg.norm(diff) == 0:
                    continue

                field_intensity = (
                    -self.G
                    * diff
                    / (np.linalg.norm(diff) ** 3)
                    * self.grad_phi(np.linalg.norm(diff) / self.epsilon)
                )
                total_field += close_sample.mass * field_intensity
        return total_field

    def compute_far(self, cell: FMMCell) -> Vec3:
        return cell.field_tensor[1:]

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
