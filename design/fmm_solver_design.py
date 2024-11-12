from typing import Callable, Generic, List, Optional, TypeVar

import numpy as np
import numpy.random as rd
import numpy.typing as npt

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
        self.index: int = 0

    def speed(self, dt: float) -> Vec3:
        return (self.pos - self.prev_pos) / dt

    def update_id(self, index: int) -> None:
        self.index = index


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
        self.neighbors: List[int] = []

    def contains_sample(self, sample: MassSample):
        return np.abs(self.centroid - sample.pos).max() < self.size / 2

    def __str__(self):
        return f"Cell(centroid: {self.centroid}, size: {self.size}, mass: {self.mass}, nparticles: {len(self.samples)})"


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
       phi: function representing the potential of one "particle" (Callable[[float], float])
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
        dt: float,
        samples: List[MassSample],
        n_max: int,
    ):
        self.size = size
        self.phi = phi
        self.dt = dt
        self.tree: OctTree[FMMCell] = OctTree()
        self.samples = samples
        self.epsilon = 4 * size / np.sqrt(len(samples))
        self.n_max = n_max

        self.leaves_cells: List[FMMCell] = []

    def make_tree(self):
        self.tree.value = FMMCell(self.samples, np.zeros(3), self.size)
        tree_stack = [self.tree]
        while len(tree_stack) != 0:
            tree = tree_stack.pop()
            cell = tree.value
            assert cell is not None

            if len(cell.samples) <= self.n_max:
                self.leaves_cells.append(cell)
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

    # def compute_multipole_extension_barycenter(self):
    #     for cell in self.leaves_cells:
    #         cell.mass = 0
    #         cell.barycenter = np.zeros(0)
    #         cell.extension = 0
    #         for sample in cell.samples:
    #             cell.mass += sample.mass
    #             cell.barycenter += sample.mass * sample.pos
    #             cell.extension = max(cell.extension, np.linalg.norm(sample.pos - cell.centroid))
