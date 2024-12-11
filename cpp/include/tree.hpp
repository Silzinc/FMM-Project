#pragma once
#include "cell.hpp"
#include "utils.hpp"
#include <boost/multi_array.hpp>
#include <functional>
#include <vector>

namespace fmm
{
template<typename T>
using Array3D = boost::multi_array<T, 3>;
using index_t = Array3D<FMMCell>::index;

struct FMMTree
{
  std::vector<Array3D<FMMCell>> data;

  /**
   * @brief Constructs a FMMTree with the specified depth and size.
   *
   * Constructs the tree structure and initializes the direct_neighbors and
   * interaction_list fields. These are the only fields that are initialized, as
   * they do not depend on the samples but only on the structure.
   *
   * @param depth The depth of the tree
   * @param size The size of the simulation cube (the cells' centroids are
   * placed so that the cube is centered at the origin)
   *
   * @note Time complexity: O(8^depth) given the depth of the tree, which
   * becomes O(len(samples)) if depth = O(log8(len(samples)))
   */
  FMMTree(index_t depth, double size);

  [[nodiscard]] index_t depth() const;
  [[nodiscard]] index_t tree_size() const;
  [[nodiscard]] double space_size() const;
  FMMCell& get_leaf_from_pos(const Vec3& pos);
  void update(
    std::vector<MassSample>& samples,
    const std::function<Vec3(const Vec3&)>& field,
    const std::function<Mat3x3(const Vec3&)>& field_jacobian);

  Array3D<FMMCell>& operator[](index_t i);
};
}
