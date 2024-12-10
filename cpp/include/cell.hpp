#pragma once
#include "sample.hpp"
#include "utils.hpp"
#include <functional>
#include <vector>

namespace fmm
{
struct FieldTensor
{
  Vec3 field;
  Mat3x3 jacobian;

  FieldTensor(const Vec3& field, const Mat3x3& jacobian);
  void clear();
  void operator+=(const FieldTensor& rhs);
  FieldTensor operator+(const FieldTensor& rhs) const;
  void operator-=(const FieldTensor& rhs);
  FieldTensor operator-(const FieldTensor& rhs) const;
};

/**
 * @brief Fast Multipole Method (FMM) cell representation
 *
 * @details Represents a spatial cell in the FMM hierarchy containing mass
 * samples and their interactions. The cell maintains both geometric and
 * physical properties needed for the FMM algorithm.
 *
 * @struct FMMCell
 * @field samples List of mass samples contained in this cell (empty if not a
 * leaf)
 * @field centroid Geometric center of the cell
 * @field size Width/size of the cell (max distance from center to contained
 * particles)
 * @field interaction_list List of cells for far-field interactions
 * @field direct_neighbors List of adjacent cells for near-field calculations
 * @field field_tensor Accumulated field tensor from far-field interactions
 * @field mass Total mass of all contained particles
 * @field barycenter Center of mass of the cell
 *
 * @note interaction_list and direct_neighbors are populated during tree
 * construction
 * @note field_tensor, mass, and barycenter are updated during tree traversal
 */
struct FMMCell
{
  std::vector<std::reference_wrapper<MassSample>> samples;
  Vec3 centroid;
  double size;
  // NOTE: The following will be set when the tree is constructed
  std::vector<std::reference_wrapper<const FMMCell>> interaction_list;
  std::vector<std::reference_wrapper<const FMMCell>> direct_neighbors;
  // NOTE: The following will be updated with the tree
  FieldTensor field_tensor;
  double mass;
  Vec3 barycenter;

  explicit FMMCell(
    const Vec3& centroid = Vec3{ 0.0, 0.0, 0.0 },
    double size = 0.0);
  [[nodiscard]] bool contains_sample(const MassSample& sample) const;
};
}
