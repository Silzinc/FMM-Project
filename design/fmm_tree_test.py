from copy import deepcopy
from time import time

from fmm_solver_design import FMMSolver, MassSample, size


def gen_random_samples(n: int):
    return [MassSample() for _ in range(n)]


def grad_phi(xi: float) -> float:
    return -(xi**3) / (1 + xi * xi) ** (3 / 2)


samples = gen_random_samples(100)
solver = FMMSolver(size, grad_phi, 0.1, deepcopy(samples), 5)
naive_solver = FMMSolver(size, grad_phi, 0.1, deepcopy(samples), 5)

solver.make_tree()
solver.compute_multipole_extension_barycenter()
solver.compute_field_tensors_neighbors()

ups = 100
print(
    f"Parameters: {ups} updates with {len(samples)} samples.\n"
    f"Average position starts at {solver.average_pos()}.\n"
    f"Standard deviation starts at {solver.std_pos()}."
)
print()

print("Use fmm solver...")
t = time()
for k in range(ups):
    solver.update()
print(f"Took {time() - t:.3f} seconds.")
print(f"Average position is now {solver.average_pos()}")
print(f"Standard deviation is now {solver.std_pos()}")
print()

print("Use naive solver...")
t = time()
for _ in range(ups):
    naive_solver.naive_update()
print(f"Took {time() - t:.3f} seconds.")
print(f"Average position is now {naive_solver.average_pos()}")
print(f"Standard deviation is now {naive_solver.std_pos()}")
print()

print(
    f"Square divergence between the two predictions: {solver.pos_divergence(naive_solver)}"
)
