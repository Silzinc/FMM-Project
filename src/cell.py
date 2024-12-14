from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .sample import MassSample
from .utils import Mat3x3, Vec3


@dataclass
class FieldTensor:
    field: Vec3
    jacobian: Mat3x3

    def clear(self):
        self.field.fill(0)
        self.jacobian.fill(0)

    def __add__(self, other: FieldTensor) -> FieldTensor:
        return FieldTensor(
            field=self.field + other.field, jacobian=self.jacobian + other.jacobian
        )

    def __iadd__(self, other: FieldTensor) -> FieldTensor:
        self.field += other.field
        self.jacobian += other.jacobian
        return self

    def __str__(self):
        return f"FieldTensor(\nfield: {self.field},\njacobian: {self.jacobian}\n)"


class FMMCell:
    """
    Fields:
      centroid: position of the cell (vec3)

      size: size of the cell (float)

      samples: list of mass samples in the cell, None if not leaf (list of MassSample or None)

      barycenter: centre of mass (vec3)

      mass: mass of the cell (float)

      field_tensor: when leaf cell sum of field tensor contributions of all other far cells (vec4)

      direct_neighbors: list of neighboring cells (list of FMMCell)

      interaction_list: list of cells that are in direct interaction with the cell (list of FMMCell)
    """

    def __init__(
        self, centroid: Vec3, size: float, samples: List[MassSample] | None = None
    ):
        self.samples = samples
        self.centroid: Vec3 = centroid
        self.size = size
        # NOTE: The following will be set when the tree is constructed
        self.interaction_list: List[FMMCell] = []
        self.direct_neighbors: List[FMMCell] = []
        # NOTE: The following will be updated with the tree
        self.field_tensor: FieldTensor = FieldTensor(
            field=np.zeros(3), jacobian=np.zeros((3, 3))
        )
        self.mass: float = 0
        self.barycenter: Vec3 = np.zeros(3)

    def contains_sample(self, sample: MassSample):
        return np.abs(self.centroid - sample.pos).max() < self.size / 2

    def __str__(self):
        return (
            f"Cell(centroid: {self.centroid}, "
            f"size: {self.size}, "
            f"mass: {self.mass:.2f}, "
            f"field_tensor: {self.field_tensor})"
        )
