#include "sample.hpp"
#include "solver.hpp"
#include "utils.hpp"
#include <boost/qvm.hpp>
#include <ctime>
#include <fmt/core.h>
#include <random>

int
main()
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

  return 0;
}
