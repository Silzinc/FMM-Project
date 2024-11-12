import numpy as np
from fmm_solver_design import FMMCell, MassSample

test_sample = MassSample()
test_cell = FMMCell([test_sample], np.array([5, 5, 5]), 5)

print(test_sample.pos)
print(test_cell.contains_sample(test_sample))
