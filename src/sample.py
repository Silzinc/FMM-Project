import numpy as np

from .utils import Vec3


class MassSample:
    """
    Fields:
      mass: mass of the particle (float)
      position: position of the particle (vec3)
      previous_position: previous position of the particle (for Verlet integration) (vec3)
    Methods:
      speed: speed of the particle (float -> vec3)
    """

    def __init__(self, pos: Vec3, prev_pos: Vec3 | None = None, mass: float = 1.0):
        """
        Initializes the MassSample with the given mass and position.
        If prev_pos is None, it is set equal to pos.

        Args:
            pos: position of the particle
            prev_pos: previous position of the particle (for Verlet integration)
            mass: mass of the particle
        """
        self.mass = mass
        self.pos: Vec3 = pos
        self.prev_pos: Vec3 = prev_pos if prev_pos is not None else np.copy(self.pos)

    def speed(self, dt: float) -> Vec3:
        """
        Returns the speed of the particle, computed as the difference between the current and previous positions, divided by dt.
        """
        return (self.pos - self.prev_pos) / dt
