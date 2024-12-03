from copy import deepcopy
from time import time

from fmm_solver_design import FMMSolver, MassSample, size


def gen_random_samples(n: int):
    return [MassSample() for _ in range(n)]


def phi(xi: float) -> float:
    return xi / (1 + xi * xi) ** (1 / 2)


def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)


samples = gen_random_samples(240)
solver = FMMSolver(size, phi, grad_phi, 0.1, deepcopy(samples), 3)
naive_solver = FMMSolver(size, phi, grad_phi, 0.1, deepcopy(samples), 3)
solver0 = FMMSolver(size, phi, grad_phi, 0.1, deepcopy(samples), 3)

ups = 50
print(
    f"Parameters: {ups} updates with {len(samples)} samples.\n"
    f"Average position starts at {solver.average_pos()}.\n"
    f"Standard deviation starts at {solver.std_pos()}.\n"
    f"Total energy starts at {solver.total_energy()}."
)
print()

print("Use fmm solver...")
t = time()
for k in range(ups):
    solver.update()
print(f"Took {time() - t:.3f} seconds.")
print(f"Average position is now {solver.average_pos()}.")
print(f"Standard deviation is now {solver.std_pos()}.")
print(f"Total energy is now {solver.total_energy()}.")
print(f"Square divergence from the start: {solver0.pos_divergence(solver)}")
print()

print("Use naive solver...")
t = time()
for _ in range(ups):
    naive_solver.naive_update()
print(f"Took {time() - t:.3f} seconds.")
print(f"Average position is now {naive_solver.average_pos()}")
print(f"Standard deviation is now {naive_solver.std_pos()}")
print(f"Total energy is now {naive_solver.total_energy()}.")
print(f"Square divergence from the start: {solver0.pos_divergence(naive_solver)}")
print()

print(
    f"Square divergence between the two predictions: {solver.pos_divergence(naive_solver)}"
)
