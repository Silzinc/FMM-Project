from fmm_solver_design import FMMSolver, MassSample, size


def gen_random_samples(n: int):
    return [MassSample() for _ in range(n)]


samples = gen_random_samples(100)
solver = FMMSolver(size, lambda x: x, 0.1, samples, 10)

solver.make_tree()

solver.tree.pretty_print()
