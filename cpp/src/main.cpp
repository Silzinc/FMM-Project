#include "sample.hpp"
#include "solver.hpp"
#include "utils.hpp"
#include <boost/qvm.hpp>
#include <ctime>
#include <fmt/core.h>
#include <random>

namespace qvm = boost::qvm;

double
phi(const Vec3& xi)
{
  const double xin = qvm::mag(xi);
  return xin / std::sqrt(1.0f + xin * xin);
}

Vec3
grad_phi(const Vec3& xi)
{
  const double xin2 = qvm::mag_sqr(xi);
  return -xi * xin2 / std::pow(1.0f + xin2, 1.5f);
}

Mat3x3
hess_phi(const Vec3& xi)
{
  using qvm::X, qvm::Y, qvm::Z;

  const Mat3x3 outer = { X(xi) * X(xi), X(xi) * Y(xi), X(xi) * Z(xi),
                         Y(xi) * X(xi), Y(xi) * Y(xi), Y(xi) * Z(xi),
                         Z(xi) * X(xi), Z(xi) * Y(xi), Z(xi) * Z(xi) };

  const Mat3x3 identity = {
    1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f
  };

  const double xin2 = qvm::mag_sqr(xi);

  return 3 * outer / std::pow(1.0f + xin2, 2.5f) -
         identity / std::pow(1.0f + xin2, 1.5f);
}

int
main()
{
  using fmt::println;

  const double size = 10.0f;
  const double mu = 1.0f;

  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution dist(-size * 0.05f, size * 0.05f);

  const int nsamples = 1000;
  const int updates = 1000;
  const double dt = 0.01f;

  int threads = 10;

  std::vector<MassSample> samples;
  samples.reserve(nsamples);
  for (int i = 0; i < nsamples; ++i)
    samples.emplace_back(
      Vec3{ dist(rng), dist(rng), dist(rng) }, std::nullopt, mu);
  std::vector<MassSample> samples_copy_1 = samples;
  std::vector<MassSample> samples_copy_2 = samples;
  std::vector<MassSample> samples_copy_3 = samples;

  FMMSolver solver_o0(size, dt, samples_copy_1, 5, &phi, &grad_phi);
  FMMSolver solver_o1(size, dt, samples_copy_2, 5, &phi, &grad_phi, &hess_phi);
  NaiveSolver naive_solver(
    dt, solver_o0.epsilon, samples_copy_3, &phi, &grad_phi);
  NaiveSolver solver0(dt, solver_o0.epsilon, samples, &phi, &grad_phi);

  println(
    "Parameters: {} updates (dt = {}) with {} samples.",
    updates,
    dt,
    samples.size());
  println("Average position starts at {}.", solver_o0.average_position());
  println("Standard deviation starts at {}.", solver_o0.std_pos());
  println("Total energy starts at {:.3f}.\n", solver_o0.total_energy());

  println("Use naive solver...");
  clock_t start = clock();
  for (int i = 0; i < updates; i++)
    naive_solver.update();
  clock_t end = clock();
  println(
    "Took {:.3f} seconds.",
    (double)threads * (double)(end - start) / (double)CLOCKS_PER_SEC);
  println("Average position is now {}.", naive_solver.average_position());
  println("Standard deviation is now {}.", naive_solver.std_pos());
  println("Total energy is now {:.3f}.", naive_solver.total_energy());
  println(
    "Square divergence with the start: {}.\n",
    naive_solver.pos_divergence(solver0));

  println("Use order 0 fmm solver...");
  start = clock();
  for (int i = 0; i < updates; i++)
    solver_o0.update();
  end = clock();
  println(
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC);
  println("Average position is now {}.", solver_o0.average_position());
  println("Standard deviation is now {}.", solver_o0.std_pos());
  println("Total energy is now {:.3f}.", solver_o0.total_energy());
  println(
    "Square divergence with the naive solver : {}.",
    solver_o0.pos_divergence(naive_solver));
  println(
    "Square divergence with the start: {}.\n",
    solver_o0.pos_divergence(solver0));

  println("Use order 1 fmm solver...");
  start = clock();
  for (int i = 0; i < updates; i++)
    solver_o1.update();
  end = clock();
  println(
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC);
  println("Average position is now {}.", solver_o1.average_position());
  println("Standard deviation is now {}.", solver_o1.std_pos());
  println("Total energy is now {}.\n", solver_o1.total_energy());
  println(
    "Square divergence with the naive solver : {}.",
    solver_o1.pos_divergence(naive_solver));
  println(
    "Square divergence with the start: {}.", solver_o1.pos_divergence(solver0));

  return 0;
}
