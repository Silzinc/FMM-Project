#pragma once

#include "cell.hpp"
#include "sample.hpp"
#include "tree.hpp"
#include "utils.hpp"
#include <functional>
#include <optional>
#include <vector>

/**
 * @brief Generic solver for particle systems.
 *
 * @field dt Time step for the solver (double)
 * @field epsilon Particles smoothing range (double)
 * @field samples List of mass samples (std::vector<MassSample>)
 * @field phi Function of r/epsilon representing the potential of one "particle"
 * per unit mass (const Vec3& -> double)
 * @field grad_phi Function of r/epsilon representing the gradient of the
 * potential of one "particle" per unit mass (const Vec3& -> Vec3)
 * @field G Gravitational constant (double)
 */
struct GenericSolver
{
  virtual ~GenericSolver() = default;

  double dt;
  double epsilon;
  std::vector<MassSample> samples;
  std::function<double(const Vec3&)> phi;
  std::function<Vec3(const Vec3&)> grad_phi;
  double G;

  GenericSolver(
    double dt,
    double epsilon,
    const std::vector<MassSample>& samples,
    const std::function<double(const Vec3&)>& phi,
    const std::function<Vec3(const Vec3&)>& grad_phi,
    double G = 0.1f);

  /**
   * Helper function to compute the potential between two points, given their
   * position difference.
   *
   * @param diff Position difference between two points (Vec3)
   *
   * @return Potential between two points (double)
   */
  [[nodiscard]] double potential(const Vec3& diff) const;

  /**
   * Helper function to compute the field intensity between two points, given
   * their position difference.
   *
   * @param diff Position difference between two points (Vec3)
   *
   * @return Field intensity from the origin of diff to its end (Vec3)
   */
  [[nodiscard]] Vec3 field_intensity(const Vec3& diff) const;

  // Utilities

  [[nodiscard]] Vec3 average_position() const;
  [[nodiscard]] Vec3 std_pos() const;
  [[nodiscard]] double pos_divergence(const GenericSolver& other) const;
  [[nodiscard]] double total_energy() const;
};

/**
 * @brief Solver for particle systems using the Fast Multipole Multiplication
 * method.
 *
 * If the depth of the space tree is chosen to be O(log8(len(samples)),
 * the time complexity of the algorithm is expected to be O(len(samples)).
 *
 * @details
 * @param size      Size of the simulation cube
 * @param phi       Function of r/epsilon representing the potential of one
 * "particle" per unit mass
 * @param grad_phi  Function of r/epsilon representing the gradient potential of
 * one "particle" per unit mass
 * @param hess_phi  Function of r/epsilon representing the hessian potential of
 * one "particle" per unit mass. Set to nullopt to use an order 0 expansion of
 * the inter-cell field. nullopt by default.
 * @param dt        Timestep
 * @param tree      Octree of the volume cells
 * @param samples   List of mass samples
 * @param epsilon   Particle smoothing size
 * @param G         Gravitational constant (defaults to 0.1)
 */
struct FMMSolver : GenericSolver
{
  FMMTree tree;
  double size;
  std::optional<std::function<Mat3x3(const Vec3&)>> hess_phi;

  FMMSolver(
    double size,
    double dt,
    const std::vector<MassSample>& samples,
    index_t depth,
    const std::function<double(const Vec3&)>& phi,
    const std::function<Vec3(const Vec3&)>& grad_phi,
    const std::optional<std::function<Mat3x3(const Vec3&)>>& hess_phi =
      std::nullopt,
    double G = 0.1f);

  /**
   * Helper function to compute the field jacobian between two points, given
   * their position difference.
   *
   * @param diff Position difference between two points (Vec3)
   *
   * @return Field jacobian from the origin of diff to its end (Mat3x3)
   */
  [[nodiscard]] Mat3x3 field_jacobian(const Vec3& diff) const;
  [[nodiscard]] Vec3 compute_close(
    const FMMCell& cell,
    const MassSample& sample) const;
  [[nodiscard]] static Vec3 compute_far(
    const FMMCell& cell,
    const MassSample& sample);
  void update(size_t threads = 1);
};

/**
 * @brief Simple quadratic complexity solver for particle systems.
 * Used for speed comparison with better algorithms.
 *
 * If the depth of the space tree is chosen to be O(log8(len(samples)),
 * the time complexity of the algorithm is expected to be O(len(samples)).
 *
 * @details
 * @param phi       Function of r/epsilon representing the potential of one
 * "particle" per unit mass
 * @param grad_phi  Function of r/epsilon representing the gradient potential of
 * one "particle" per unit mass
 * @param dt        Timestep
 * @param samples   List of mass samples
 * @param epsilon   Particle smoothing size
 * @param G         Gravitational constant (defaults to 0.1)
 */
struct NaiveSolver : GenericSolver
{
  // Inherit constructor
  using GenericSolver::GenericSolver;

  void update(size_t threads = 1);
};
