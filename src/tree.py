from typing import Callable, List

import numpy as np

from .cell import FMMCell
from .sample import MassSample
from .utils import Mat3x3, Vec3


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
        """
        Returns the leaf cell containing the given position.
        """
        l = self.depth() - 1
        size = self.space_size()
        i = int((pos[0] / size + 0.5) * 2**l)
        j = int((pos[1] / size + 0.5) * 2**l)
        k = int((pos[2] / size + 0.5) * 2**l)
        return self[l][min(len(self[l]) - 1, max(0, i))][
            min(len(self[l]) - 1, max(0, j))
        ][min(len(self[l]) - 1, max(0, k))]

    def update(
        self,
        samples: List[MassSample],
        field: Callable[[Vec3], Vec3],
        field_jacobian: Callable[[Vec3], Mat3x3],
    ) -> None:
        """
        Updates the tree's content with the given samples and field function.
        The structure itself does not change, but the mass, barycenter, field tensors
        and distribution of samples in the leaves are updated.

        Args:
            samples: list of mass samples
            field: function of the distance between two points representing the field between them per unit mass
            field_jacobian: function of the distance between two points representing the jacobian of the field between them per unit mass

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
                        cell.field_tensor.clear()
                        for neighbor in cell.interaction_list:
                            diff = cell.barycenter - neighbor.barycenter
                            field_intensity = field(diff)
                            jacobian = field_jacobian(diff)
                            cell.field_tensor.field += neighbor.mass * field_intensity
                            cell.field_tensor.jacobian += neighbor.mass * jacobian
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
                                    child.field_tensor += cell.field_tensor
                                    child.field_tensor.field += (
                                        cell.field_tensor.jacobian
                                        @ (child.barycenter - cell.barycenter)
                                    )
