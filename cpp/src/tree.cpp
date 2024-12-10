#include <algorithm>
#include <cstdlib>
#include <fmt/base.h>
#include <functional>
#include <ranges>

#include "sample.hpp"
#include "tree.hpp"

namespace fmm
{
FMMTree::FMMTree(const index_t depth, const double size)
{
  const auto& range = std::ranges::views::iota;

  data.clear();
  data.reserve(depth);

  for (size_t l = 0; l < depth; l++) {
    const index_t n = 1 << l;
    const auto fn = static_cast<double>(n);
    data.emplace_back(boost::extents[n][n][n]);

    for (index_t i = 0; i < n; i++) {
      for (index_t j = 0; j < n; j++) {
        for (index_t k = 0; k < n; k++) {
          data[l][i][j][k] = FMMCell(
            Vec3{ size * ((static_cast<double>(i) + 0.5) / fn - 0.5),
                  size * ((static_cast<double>(j) + 0.5) / fn - 0.5),
                  size * ((static_cast<double>(k) + 0.5) / fn - 0.5) },
            size / fn);
        }
      }
    }
  }

  for (index_t l = 0; l < depth; l++) {
    index_t width = 1 << l;
    for (index_t i = 0; i < width; i++) {
      for (index_t j = 0; j < width; j++) {
        for (index_t k = 0; k < width; k++) {
          FMMCell& cell = data[l][i][j][k];
          for (const index_t ni :
               range(std::max(i, 1l) - 1l, std::min(i + 2l, width)))
            for (const index_t nj :
                 range(std::max(j, 1l) - 1l, std::min(j + 2l, width)))
              for (const index_t nk :
                   range(std::max(k, 1l) - 1l, std::min(k + 2l, width)))
                cell.direct_neighbors.emplace_back(data[l][ni][nj][nk]);
          if (l > 0) {
            index_t pi = i / 2, pj = j / 2, pk = k / 2;
            for (const index_t ni :
                 range(2 * std::max(pi, 1l) - 2, std::min(width, 2 * pi + 4))) {
              for (const index_t nj : range(
                     2 * std::max(pj, 1l) - 2, std::min(width, 2 * pj + 4))) {
                for (const index_t nk : range(
                       2 * std::max(pk, 1l) - 2, std::min(width, 2 * pk + 4))) {
                  const FMMCell& near_child = data[l][ni][nj][nk];
                  if (
                    std::max({ std::max(ni, i) - std::min(ni, i),
                               std::max(nj, j) - std::min(nj, j),
                               std::max(nk, k) - std::min(nk, k) }) > 1)
                    cell.interaction_list.emplace_back(near_child);
                }
              }
            }
          }
        }
      }
    }
  }
}

index_t
FMMTree::depth() const
{
  return static_cast<index_t>(data.size());
}

index_t
FMMTree::tree_size() const
{
  return static_cast<index_t>(std::round(std::pow(8, depth()) - 1)) / 7;
}

double
FMMTree::space_size() const
{
  return data[0][0][0][0].size;
}

FMMCell&
FMMTree::get_leaf_from_pos(const Vec3& pos)
{
  index_t l = depth() - 1;
  index_t width = 1 << l;
  auto wf = static_cast<double>(width);
  double size = space_size();
  Vec3 local_pos = pos / size + Vec3{ 0.5, 0.5, 0.5 };
  index_t i = std::clamp(
    static_cast<index_t>(std::floor(X(local_pos) * wf)), 0l, width - 1);
  index_t j = std::clamp(
    static_cast<index_t>(std::floor(Y(local_pos) * wf)), 0l, width - 1);
  index_t k = std::clamp(
    static_cast<index_t>(std::floor(Z(local_pos) * wf)), 0l, width - 1);
  return data[l][i][j][k];
}

void
FMMTree::update(
  std::vector<MassSample>& samples,
  const std::function<Vec3(const Vec3&)>& field,
  const std::function<Mat3x3(const Vec3&)>& field_jacobian)
{
  using fmt::println;

  const auto& range = std::ranges::views::iota;

  index_t d = depth();
  index_t l = d - 1;
  index_t width = 1 << l;

  // Clear the tree leaf cells
  for (const index_t i : range(0, width)) {
    for (const index_t j : range(0, width)) {
      for (const index_t k : range(0, width)) {
        FMMCell& leaf = data[l][i][j][k];
        leaf.samples.clear();
        leaf.mass = 0.0;
        leaf.barycenter = { 0.0, 0.0, 0.0 };
      }
    }
  }

  // Populate the leaves and compute their masses and barycenter
  for (MassSample& sample : samples) {
    FMMCell& leaf = get_leaf_from_pos(sample.position);
    leaf.samples.emplace_back(sample);
    leaf.mass += sample.mass;
    leaf.barycenter += sample.mass * sample.position;
  }

  for (const index_t i : range(0, width)) {
    for (const index_t j : range(0, width)) {
      for (const index_t k : range(0, width)) {
        if (FMMCell& leaf = data[l][i][j][k]; !leaf.samples.empty())
          leaf.barycenter /= leaf.mass;
      }
    }
  }

  // Propagate mass and barycenter upward.
  // Unintuitively with the 7 nested for loops, this is done
  // in O(len(samples)) if depth is chosen to have as many
  // leaves as there are samples.
  for (const index_t ml : range(0, d - 1)) {
    l = d - 2 - ml;
    width = 1 << l;
    for (const index_t i : range(0, width)) {
      for (const index_t j : range(0, width)) {
        for (const index_t k : range(0, width)) {
          FMMCell& cell = data[l][i][j][k];
          cell.mass = 0.0;
          cell.barycenter = Vec3{ 0.0, 0.0, 0.0 };
          for (const index_t ci : { 2 * i, 2 * i + 1 }) {
            for (const index_t cj : { 2 * j, 2 * j + 1 }) {
              for (const index_t ck : { 2 * k, 2 * k + 1 }) {
                FMMCell& child = data[l + 1][ci][cj][ck];
                cell.mass += child.mass;
                cell.barycenter += child.mass * child.barycenter;
              }
            }
          }
          if (cell.mass > 0.0)
            cell.barycenter /= cell.mass;
        }
      }
    }
  }

  // Compute the field tensors. First, compute the contributions of the
  // interaction lists.
  for (l = 0; l < d; l++) {
    width = 1 << l;
    for (const index_t i : range(0, width)) {
      for (const index_t j : range(0, width)) {
        for (const index_t k : range(0, width)) {
          FMMCell& cell = data[l][i][j][k];
          if (cell.samples.empty())
            continue;
          cell.field_tensor.clear();
          for (auto& neighbor_ref : cell.interaction_list) {
            const FMMCell& neighbor = neighbor_ref.get();
            const Vec3 diff = cell.barycenter - neighbor.barycenter;
            const Vec3 field_intensity = field(diff);
            const Mat3x3 jacobian = field_jacobian(diff);
            cell.field_tensor.field += neighbor.mass * field_intensity;
            cell.field_tensor.jacobian += neighbor.mass * jacobian;
          }
        }
      }
    }
  }
  // Then, propagate the field tensors downward.
  for (l = 0; l < d - 1; l++) {
    width = 1 << l;
    for (const index_t i : range(0, width)) {
      for (const index_t j : range(0, width)) {
        for (const index_t k : range(0, width)) {
          const FMMCell& cell = data[l][i][j][k];
          for (const index_t ci : { 2 * i, 2 * i + 1 }) {
            for (const index_t cj : { 2 * j, 2 * j + 1 }) {
              for (const index_t ck : { 2 * k, 2 * k + 1 }) {
                FMMCell& child = data[l + 1][ci][cj][ck];
                child.field_tensor += cell.field_tensor;
              }
            }
          }
        }
      }
    }
  }
}

Array3D<FMMCell>&
FMMTree::operator[](index_t i)
{
  return data[i];
}
}
