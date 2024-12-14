#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "fmm.hpp"
#include <boost/qvm.hpp>
#include <ctime>
#include <fmt/core.h>
#include <matplot/matplot.h>
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

  std::vector<fmm::MassSample> samples;
  samples.reserve(nsamples);
  for (int i = 0; i < nsamples; ++i)
    samples.emplace_back(fmm::Vec3{ dist(rng), dist(rng), dist(rng) }, mu);
  std::vector<fmm::MassSample> samples_copy_1 = samples;
  std::vector<fmm::MassSample> samples_copy_2 = samples;
  std::vector<fmm::MassSample> samples_copy_3 = samples;

  fmm::FMMSolver solver_o0(
    size, dt, samples_copy_1, 3, &fmm::plummer::phi, &fmm::plummer::grad_phi
  );
  fmm::FMMSolver solver_o1(
    size,
    dt,
    samples_copy_2,
    3,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi,
    &fmm::plummer::hess_phi
  );
  fmm::NaiveSolver naive_solver(
    size,
    dt,
    solver_o0.epsilon,
    samples_copy_3,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi
  );
  fmm::NaiveSolver solver0(
    size,
    dt,
    solver_o0.epsilon,
    samples,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi
  );

  println(
    "Parameters: {} updates (dt = {}) with {} samples.",
    updates,
    dt,
    samples.size()
  );
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
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC
  );
  println("Average position is now {}.", naive_solver.average_position());
  println("Standard deviation is now {}.", naive_solver.std_pos());
  println("Total energy is now {:.3f}.", naive_solver.total_energy());
  println(
    "Square divergence with the start: {}.",
    naive_solver.pos_divergence(solver0)
  );
  println("");

  println("Use order 0 fmm solver...");
  start = clock();
  for (int i = 0; i < updates; i++)
    solver_o0.update();
  end = clock();
  println(
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC
  );
  println("Average position is now {}.", solver_o0.average_position());
  println("Standard deviation is now {}.", solver_o0.std_pos());
  println("Total energy is now {:.3f}.", solver_o0.total_energy());
  println(
    "Square divergence with the naive solver : {}.",
    solver_o0.pos_divergence(naive_solver)
  );
  println(
    "Square divergence with the start: {}.", solver_o0.pos_divergence(solver0)
  );
  println("");

  println("Use order 1 fmm solver...");
  start = clock();
  for (int i = 0; i < updates; i++)
    solver_o1.update();
  end = clock();
  println(
    "Took {:.3f} seconds.", (double)(end - start) / (double)CLOCKS_PER_SEC
  );
  println("Average position is now {}.", solver_o1.average_position());
  println("Standard deviation is now {}.", solver_o1.std_pos());
  println("Total energy is now {}.", solver_o1.total_energy());
  println(
    "Square divergence with the naive solver : {}.",
    solver_o1.pos_divergence(naive_solver)
  );
  println(
    "Square divergence with the start: {}.", solver_o1.pos_divergence(solver0)
  );
  println("");
}

TEST_CASE("Dependency of solving time against number of particles")
{
  using fmt::println;
  namespace plt = matplot;

  const double size = 10.0;
  const double mu = 1.0;
  const double dt = 0.1;
  const int updates = 50;

  const int nsamples_start = 25;
  const int nsamples_end = 2000;
  const float nsamples_factor = 1.2;

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> dist(-size * 0.5, size * 0.5);
  std::vector<fmm::MassSample> samples;

  fmm::FMMSolver solver_o0_d3(
    size, dt, {}, 3, &fmm::plummer::phi, &fmm::plummer::grad_phi
  );
  fmm::FMMSolver solver_o1_d3(
    size,
    dt,
    {},
    3,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi,
    &fmm::plummer::hess_phi
  );
  fmm::FMMSolver solver_o0_d4(
    size, dt, {}, 4, &fmm::plummer::phi, &fmm::plummer::grad_phi
  );
  fmm::FMMSolver solver_o1_d4(
    size,
    dt,
    {},
    4,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi,
    &fmm::plummer::hess_phi
  );
  fmm::FMMSolver solver_o0_d5(
    size, dt, {}, 5, &fmm::plummer::phi, &fmm::plummer::grad_phi
  );
  fmm::FMMSolver solver_o1_d5(
    size,
    dt,
    {},
    5,
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi,
    &fmm::plummer::hess_phi
  );
  fmm::NaiveSolver naive_solver(
    size,
    dt,
    solver_o0_d3.epsilon,
    {},
    &fmm::plummer::phi,
    &fmm::plummer::grad_phi
  );

  std::vector<double> times_o0_d3;
  std::vector<double> times_o1_d3;
  std::vector<double> times_o0_d4;
  std::vector<double> times_o1_d4;
  std::vector<double> times_o0_d5;
  std::vector<double> times_o1_d5;
  std::vector<double> times_naive;
  std::vector<int> samples_count;

  // Fixing epsilon for all simulations
  solver_o0_d3.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  solver_o1_d3.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  solver_o0_d4.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  solver_o1_d4.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  solver_o0_d5.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  solver_o1_d5.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));
  naive_solver.epsilon =
    4.0 * size / std::sqrt(static_cast<double>(nsamples_end));

  std::vector<std::thread> threads;

  samples.reserve(nsamples_end);

  for (int nsamples = nsamples_start; nsamples < nsamples_end;
       nsamples =
         static_cast<int>(static_cast<float>(nsamples) * nsamples_factor)) {

    println("Number of samples: {}", nsamples);
    samples_count.push_back(nsamples);

    while (samples.size() < nsamples)
      samples.emplace_back(fmm::Vec3{ dist(rng), dist(rng), dist(rng) }, mu);

    solver_o0_d3.samples.clear();
    solver_o0_d3.samples.insert(
      solver_o0_d3.samples.end(), samples.begin(), samples.end()
    );

    solver_o1_d3.samples.clear();
    solver_o1_d3.samples.insert(
      solver_o1_d3.samples.end(), samples.begin(), samples.end()
    );

    solver_o0_d4.samples.clear();
    solver_o0_d4.samples.insert(
      solver_o0_d4.samples.end(), samples.begin(), samples.end()
    );

    solver_o1_d4.samples.clear();
    solver_o1_d4.samples.insert(
      solver_o1_d4.samples.end(), samples.begin(), samples.end()
    );

    solver_o0_d5.samples.clear();
    solver_o0_d5.samples.insert(
      solver_o0_d5.samples.end(), samples.begin(), samples.end()
    );

    solver_o1_d5.samples.clear();
    solver_o1_d5.samples.insert(
      solver_o1_d5.samples.end(), samples.begin(), samples.end()
    );

    naive_solver.samples.clear();
    naive_solver.samples.insert(
      naive_solver.samples.end(), samples.begin(), samples.end()
    );

    threads.clear();

    threads.emplace_back([&solver_o0_d3, updates, &times_o0_d3]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o0_d3.update();
      clock_t end = clock();
      times_o0_d3.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&solver_o1_d3, updates, &times_o1_d3]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o1_d3.update();
      clock_t end = clock();
      times_o1_d3.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&solver_o0_d4, updates, &times_o0_d4]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o0_d4.update();
      clock_t end = clock();
      times_o0_d4.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&solver_o1_d4, updates, &times_o1_d4]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o1_d4.update();
      clock_t end = clock();
      times_o1_d4.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&solver_o0_d5, updates, &times_o0_d5]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o0_d5.update();
      clock_t end = clock();
      times_o0_d5.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&solver_o1_d5, updates, &times_o1_d5]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        solver_o1_d5.update();
      clock_t end = clock();
      times_o1_d5.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    threads.emplace_back([&naive_solver, updates, &times_naive]() {
      clock_t start = clock();
      for (int i = 0; i < updates; i++)
        naive_solver.update();
      clock_t end = clock();
      times_naive.push_back((double)(end - start) / (double)CLOCKS_PER_SEC);
    });

    for (auto& thread : threads)
      thread.join();
  }

  plt::hold(plt::on);

  plt::plot(samples_count, times_o0_d3, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 3");
  plt::plot(samples_count, times_o1_d3, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 3");
  plt::plot(samples_count, times_o0_d4, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 4");
  plt::plot(samples_count, times_o1_d4, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 4");
  plt::plot(samples_count, times_o0_d5, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 5");
  plt::plot(samples_count, times_o1_d5, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 5");
  plt::plot(samples_count, times_naive, "-x")
    ->line_width(2)
    .display_name("Naive");

  plt::hold(plt::off);

  plt::xlabel("Number of samples");
  plt::ylabel("Time for " + std::to_string(updates) + " updates (s)");
  plt::legend()->location(plt::legend::general_alignment::topleft);
  plt::show();

  plt::cla();
  plt::hold(plt::on);

  plt::loglog(samples_count, times_o0_d3, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 3");
  plt::loglog(samples_count, times_o1_d3, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 3");
  plt::loglog(samples_count, times_o0_d4, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 4");
  plt::loglog(samples_count, times_o1_d4, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 4");
  plt::loglog(samples_count, times_o0_d5, "-x")
    ->line_width(2)
    .display_name("FMM 0-order, depth = 5");
  plt::loglog(samples_count, times_o1_d5, "-x")
    ->line_width(2)
    .display_name("FMM 1st-order depth = 5");
  plt::loglog(samples_count, times_naive, "-x")
    ->line_width(2)
    .display_name("Naive");

  plt::hold(plt::off);

  plt::xlabel("Number of samples");
  plt::ylabel("Time for " + std::to_string(updates) + " updates (s)");
  plt::legend()->location(plt::legend::general_alignment::topleft);
  plt::show();
}
