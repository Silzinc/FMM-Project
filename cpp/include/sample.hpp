#pragma once
#include "utils.hpp"
#include <optional>

namespace fmm
{
/**
 * @brief A sample of a mass point
 *
 * This class represents a physical point mass with position and previous
 * position for Verlet integration calculations.
 *
 * @details The class stores three main properties:
 *          - Mass of the particle
 *          - Current position in 3D space
 *          - Previous position
 *
 * @param mass Mass of the particle (double)
 * @param position Position vector of the particle (Vec3)
 * @param prev_pos Previous position vector (Vec3)
 *
 * @see MassSample() Constructor allows optional mass (default 1.0) and
 * previous position
 * @see speed() Method to calculate current velocity
 */
struct MassSample
{
  double mass;
  Vec3 position;
  Vec3 prev_pos;

  /**
   * @brief Construct a new MassSample object
   *
   * @param pos Position of the mass sample
   * @param prev_pos Previous position of the mass sample, default is the same
   * as pos
   * @param mass Mass of the mass sample, default is 1.0
   */
  explicit MassSample(
    Vec3 pos,
    double mass = 1.0,
    std::optional<Vec3> prev_pos = std::nullopt
  );

  /**
   * @brief Calculate the speed of the mass sample
   *
   * @param dt Time step
   * @return Vec3 Velocity vector
   */
  [[nodiscard]] Vec3 speed(double dt) const;
};
}
