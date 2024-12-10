#include "solver.hpp"
#include "cell.hpp"
#include "tree.hpp"
#include "utils.hpp"
#include <algorithm>
#include <boost/qvm.hpp>
#include <boost/qvm/math.hpp>
#include <fmt/base.h>
#include <ranges>
#include <thread>

namespace qvm = boost::qvm;

GenericSolver::GenericSolver(
  const double dt,
  const double epsilon,
  const std::vector<MassSample>& samples,
  const std::function<double(const Vec3&)>& phi,
  const std::function<Vec3(const Vec3&)>& grad_phi,
  const double G)
  : dt(dt)
  , epsilon(epsilon)
  , samples(samples)
  , phi(phi)
  , grad_phi(grad_phi)
  , G(G)
{}

double
GenericSolver::potential(const Vec3& diff) const
{
  return -G / qvm::mag(diff) * phi(diff / epsilon);
}

Vec3
GenericSolver::field_intensity(const Vec3& diff) const
{
  return G / qvm::mag_sqr(diff) * grad_phi(diff / epsilon);
}

Vec3
GenericSolver::average_position() const
{
  Vec3 avg{ 0.0f };
  for (const MassSample& sample : samples)
    avg += sample.position;
  return avg / static_cast<double>(samples.size());
}

Vec3
GenericSolver::std_pos() const
{
  using qvm::X, qvm::Y, qvm::Z;

  const Vec3 avg = average_position();
  Vec3 std{ 0.0f };
  for (const MassSample& sample : samples) {
    const Vec3 diff = sample.position - avg;
    std += Vec3{ X(diff) * X(diff), Y(diff) * Y(diff), Z(diff) * Z(diff) };
  }
  std /= static_cast<double>(samples.size());
  X(std) = std::sqrt(X(std));
  Y(std) = std::sqrt(Y(std));
  Z(std) = std::sqrt(Z(std));
  return std;
}

double
GenericSolver::pos_divergence(const GenericSolver& other) const
{
  const auto& range = std::ranges::views::iota;

  double divergence = 0.0f;
  for (size_t i : range(0ul, samples.size())) {
    const Vec3 diff = samples[i].position - other.samples[i].position;
    divergence += qvm::mag_sqr(diff);
  }
  divergence /= static_cast<double>(samples.size());
  return divergence;
}

double
GenericSolver::total_energy() const
{
  // Kinetic energy
  double ke = 0.0f;
  for (const MassSample& sample : samples) {
    ke += 0.5f * sample.mass * qvm::mag_sqr(sample.speed(dt));
  }
  // Potential energy
  double pe = 0.0f;
  for (const MassSample& s1 : samples)
    for (const MassSample& s2 : samples)
      if (&s1 != &s2)
        pe += potential(s1.position - s2.position) * s1.mass * s2.mass;
  return ke + pe;
}

FMMSolver::FMMSolver(
  const double size,
  const double dt,
  const std::vector<MassSample>& samples,
  const index_t depth,
  const std::function<double(const Vec3&)>& phi,
  const std::function<Vec3(const Vec3&)>& grad_phi,
  const std::optional<std::function<Mat3x3(const Vec3&)>>& hess_phi,
  const double G)
  : GenericSolver(
      dt,
      4.0f * size / std::sqrt(static_cast<double>(samples.size())),
      samples,
      phi,
      grad_phi,
      G)
  , tree(FMMTree(depth, size))
  , size(size)
  , hess_phi(hess_phi)
{}

Mat3x3
FMMSolver::field_jacobian(const Vec3& diff) const
{
  if (hess_phi.has_value())
    return G / std::pow(qvm::mag(diff), 3.0f) *
           hess_phi.value()(diff / epsilon);
  else
    return Mat3x3{ 0.0f };
}

Vec3
FMMSolver::compute_close(const FMMCell& cell, const MassSample& sample) const
{
  Vec3 total_field = { 0.0f };
  for (const auto& neighbor_cell : cell.direct_neighbors) {
    if (neighbor_cell.get().samples.empty())
      continue;
    for (const auto& neighbor_sample : neighbor_cell.get().samples) {
      const Vec3 diff = sample.position - neighbor_sample.get().position;
      if (qvm::mag(diff) == 0.0f)
        continue;
      total_field += neighbor_sample.get().mass * field_intensity(diff);
    }
  }
  return total_field;
}

Vec3
FMMSolver::compute_far(const FMMCell& cell, const MassSample& sample)
{
  return cell.field_tensor.field +
         cell.field_tensor.jacobian * (sample.position - cell.barycenter);
}

void
FMMSolver::update(const size_t threads)
{
  const auto& range = std::ranges::views::iota;

  tree.update(
    samples,
    [this](const Vec3& diff) { return field_intensity(diff); },
    [this](const Vec3& diff) { return field_jacobian(diff); });

  std::vector<Vec3> new_poss(samples.size());

  index_t index = 0;
  const index_t l = tree.depth() - 1, width = 1 << l;
  for (const index_t i : range(0, width)) {
    for (const index_t j : range(0, width)) {
      for (const index_t k : range(0, width)) {
        const FMMCell& cell = tree[l][i][j][k];
        if (cell.samples.empty())
          continue;
        for (const auto& sample : cell.samples) {
          Vec3 acc = { 0.0f };
          acc += compute_close(cell, sample);
          acc += compute_far(cell, sample);
          new_poss[index] =
            2 * sample.get().position - sample.get().prev_pos + acc * dt * dt;
          index++;
        }
      }
    }
  }

  index = 0;
  for (const index_t i : range(0, width)) {
    for (const index_t j : range(0, width)) {
      for (const index_t k : range(0, width)) {
        const FMMCell& cell = tree[l][i][j][k];
        if (cell.samples.empty())
          continue;
        for (auto& sample : cell.samples) {
          sample.get().prev_pos = sample.get().position;
          sample.get().position = new_poss[index];
          index++;
        }
      }
    }
  }
}

void
NaiveSolver::update(const size_t threads)
{
  std::vector<Vec3> new_poss;
  new_poss.resize(samples.size());
  std::ranges::fill(new_poss, Vec3{ 0.0f, 0.0f, 0.0f });

  for (index_t index = 0; index < samples.size(); ++index) {
    Vec3 acc = { 0.0f };
    const MassSample& sample1 = samples[index];
    for (const MassSample& sample2 : samples) {
      const Vec3& diff = sample1.position - sample2.position;
      if (qvm::mag_sqr(diff) != 0.0f)
        acc += sample2.mass * field_intensity(diff);
    }
    new_poss[index] = 2 * sample1.position - sample1.prev_pos + acc * dt * dt;
  }

  for (index_t index = 0; index < samples.size(); index++) {
    MassSample& sample = samples[index];
    sample.prev_pos = sample.position;
    sample.position = new_poss[index];
  }
}
