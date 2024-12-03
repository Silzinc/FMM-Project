from typing import List

import numpy as np

from .sample import MassSample
from .utils import Vec3


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
        # NOTE: The following will be set when the tree is constructed
        self.interaction_list: List[FMMCell] = []
        self.direct_neighbors: List[FMMCell] = []
        # NOTE: The following will be updated with the tree
        self.field_tensor: Vec3 = np.zeros(3)
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
