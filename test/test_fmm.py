from copy import deepcopy
from time import time

import numpy as np

import src as fmm

np.set_printoptions(precision=3, suppress=True)

size = 10.0
mu = 1.0


def phi(xi: float) -> float:
    return xi / (1 + xi * xi) ** (1 / 2)


def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)


def test_random_samples_fmm():
    def gen_random_samples(n: int):
        return [
            fmm.MassSample((np.random.rand(3) - 0.5) * size, mass=mu) for _ in range(n)
        ]

    nsamples = 140
    updates = 150
    dt = 0.01

    samples = gen_random_samples(nsamples)
    solver = fmm.FMMSolver(size, phi, grad_phi, dt, deepcopy(samples), 3)
    naive_solver = fmm.NaiveSolver(phi, grad_phi, dt, solver.epsilon, deepcopy(samples))
    solver0 = fmm.NaiveSolver(phi, grad_phi, dt, solver.epsilon, deepcopy(samples))

    print(
        f"Parameters: {updates} updates with {len(samples)} samples.\n"
        f"Average position starts at {solver.average_pos()}.\n"
        f"Standard deviation starts at {solver.std_pos()}.\n"
        f"Total energy starts at {solver.total_energy()}."
    )
    print()

    print("Use fmm solver...")
    t = time()
    for k in range(updates):
        solver.update()
    print(f"Took {time() - t:.3f} seconds.")
    print(f"Average position is now {solver.average_pos()}.")
    print(f"Standard deviation is now {solver.std_pos()}.")
    print(f"Total energy is now {solver.total_energy()}.")
    print(f"Square divergence from the start: {solver0.pos_divergence(solver)}")
    print()

    print("Use naive solver...")
    t = time()
    for _ in range(updates):
        naive_solver.update()
    print(f"Took {time() - t:.3f} seconds.")
    print(f"Average position is now {naive_solver.average_pos()}")
    print(f"Standard deviation is now {naive_solver.std_pos()}")
    print(f"Total energy is now {naive_solver.total_energy()}.")
    print(f"Square divergence from the start: {solver0.pos_divergence(naive_solver)}")
    print()

    print(
        f"Square divergence between the two predictions: {solver.pos_divergence(naive_solver)}"
    )
