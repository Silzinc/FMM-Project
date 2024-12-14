from copy import deepcopy
from time import time

import numpy as np

import src as fmm

np.set_printoptions(precision=3, suppress=True)

size = 10.0
mu = 1.0


def phi(xi: fmm.Vec3) -> float:
    xin = np.linalg.norm(xi)
    return float(xin / (1 + xin * xin) ** (1 / 2))


def grad_phi(xi: fmm.Vec3) -> fmm.Vec3:
    xin = np.linalg.norm(xi)
    return -(xin**2) / (1 + xin * xin) ** (3 / 2) * xi


def hess_phi(xi: fmm.Vec3) -> fmm.Mat3x3:
    xin = np.linalg.norm(xi)
    return xin**3 * (
        3 * np.outer(xi, xi) / (1 + xin**2) ** (5 / 2)
        - np.eye(3) / (1 + xin**2) ** (3 / 2)
    )


def test_random_samples_fmm():
    nsamples = 100
    updates = 100
    dt = 0.1

    # Use Mersenne Twister 19937 to match the C++ implementation
    # Well after trying it, it seems the implementations are not the same
    rng = np.random.Generator(np.random.MT19937(42))
    samples = [
        fmm.MassSample(rng.uniform(low=-size * 0.5, high=size * 0.5, size=3), mass=mu)
        for _ in range(nsamples)
    ]

    solver_o0 = fmm.FMMSolver(size, dt, deepcopy(samples), 4, phi, grad_phi)
    solver_o1 = fmm.FMMSolver(
        size, dt, deepcopy(samples), 4, phi, grad_phi, hess_phi=hess_phi
    )
    naive_solver = fmm.NaiveSolver(
        size, dt, solver_o0.epsilon, deepcopy(samples), phi, grad_phi
    )
    solver_init = fmm.NaiveSolver(
        size, dt, solver_o0.epsilon, deepcopy(samples), phi, grad_phi
    )

    print(
        f"Parameters: {updates} updates (dt = {dt}) with {len(samples)} samples.\n"
        f"Average position starts at {solver_init.average_pos()}.\n"
        f"Standard deviation starts at {solver_init.std_pos()}.\n"
        f"Total energy starts at {solver_init.total_energy()}."
    )
    print()

    print("Use naive solver...")
    t = time()
    for _ in range(updates):
        naive_solver.update()
    print(f"Took {time() - t:.3f} seconds.")
    print(f"Average position is now {naive_solver.average_pos()}")
    print(f"Standard deviation is now {naive_solver.std_pos()}")
    print(f"Total energy is now {naive_solver.total_energy()}.")
    print(
        f"Square divergence from the start: {solver_init.pos_divergence(naive_solver)}"
    )
    print()

    print("Use order 0 fmm solver...")
    t = time()
    for _ in range(updates):
        solver_o0.update()
    print(f"Took {time() - t:.3f} seconds.")
    print(f"Average position is now {solver_o0.average_pos()}.")
    print(f"Standard deviation is now {solver_o0.std_pos()}.")
    print(f"Total energy is now {solver_o0.total_energy()}.")
    print(f"Square divergence from the start: {solver_init.pos_divergence(solver_o0)}")
    print(
        f"Square divergence with the naive solver : {solver_o0.pos_divergence(naive_solver)}"
    )
    print()

    print("Use order 1 fmm solver...")
    t = time()
    for _ in range(updates):
        solver_o1.update()
    print(f"Took {time() - t:.3f} seconds.")
    print(f"Average position is now {solver_o1.average_pos()}.")
    print(f"Standard deviation is now {solver_o1.std_pos()}.")
    print(f"Total energy is now {solver_o1.total_energy()}.")
    print(f"Square divergence from the start: {solver_init.pos_divergence(solver_o1)}")
    print(
        f"Square divergence with the naive solver : {solver_o1.pos_divergence(naive_solver)}"
    )
    print()
