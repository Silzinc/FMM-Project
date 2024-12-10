#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "sample.hpp"
#include "solver.hpp"
#include "utils.hpp"
#include <boost/qvm.hpp>
#include <ctime>
#include <fmt/core.h>
#include <random>

TEST_CASE("FMM solver with uniformly randomly distributed samples")
{
  using fmt::println;

  const double size = 10.0;
  const double mu = 1.0;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-size * 0.5, size * 0.5);

  const int nsamples = 100;
  const int updates = 100;
  const double dt = 0.1;

  int threads = 10;

  std::vector<fmm::MassSample> samples;
  samples.reserve(nsamples);
  for (int i = 0; i < nsamples; ++i)
    samples.emplace_back(fmm::Vec3{ dist(rng), dist(rng), dist(rng) }, mu);
  std::vector<fmm::MassSample> samples_copy_1 = samples;
  std::vector<fmm::MassSample> samples_copy_2 = samples;
  std::vector<fmm::MassSample> samples_copy_3 = samples;

  fmm::FMMSolver solver_o0(
    size, dt, samples_copy_1, 3, &fmm::plummer::phi, &fmm::plummer::grad_phi);
  fmm::FMMSolver solver_o1(
    size,
    dt,
    samples_copy_2,
    3,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi,
    &fmm::plummer::hess_phi);
  fmm::NaiveSolver naive_solver(
    dt,
    solver_o0.epsilon,
    samples_copy_3,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi);
  fmm::NaiveSolver solver0(
    dt,
    solver_o0.epsilon,
    samples,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi);

  println(
    "Parameters: {} updates (dt = {}) with {} samples.",
    updates,
    dt,
    samples.size());
  println("Average position starts at {}.", solver_o0.average_position());
  println("Standard deviation starts at {}.", solver_o0.std_pos());
  println("Total energy starts at {:.3f}.", solver_o0.total_energy());
  println("");

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
    "Square divergence with the start: {}.",
    naive_solver.pos_divergence(solver0));
  println("");

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
    "Square divergence with the start: {}.", solver_o0.pos_divergence(solver0));
  println("");

  println("Use order 1 fmm solver...");
  start = clock();
  for (int i = 0; i < updates; i++)
    solver_o1.update();
  end = clock();
  println(
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC);
  println("Average position is now {}.", solver_o1.average_position());
  println("Standard deviation is now {}.", solver_o1.std_pos());
  println("Total energy is now {}.", solver_o1.total_energy());
  println(
    "Square divergence with the naive solver : {}.",
    solver_o1.pos_divergence(naive_solver));
  println(
    "Square divergence with the start: {}.", solver_o1.pos_divergence(solver0));
  println("");
}
