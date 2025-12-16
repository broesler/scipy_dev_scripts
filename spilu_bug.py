#!/usr/bin/env python3
# =============================================================================
#     File: spilu_bug.py
#  Created: 2025-12-16 12:59
#   Author: Bernie Roesler
#
"""Test incomplete LU decomposition with SciPy spilu."""
# =============================================================================

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy import linalg as la
from scipy.sparse import csc_array
from scipy.sparse import linalg as spla


def allclose(a, b, atol=1e-15):
    """Check if two arrays are close to each other."""
    return assert_allclose(a, b, atol=atol)


# See: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems,
# p 74 (Figure 5.1)
N = 8
rows = np.array(
    [0, 1, 2, 3, 4, 5, 6, 3, 6, 1, 6, 0, 2, 5, 7, 4, 7, 0, 1, 3, 7, 5, 6],
    dtype=np.int32,
)
cols = np.array(
    [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7],
    dtype=np.int32,
)

rng = np.random.default_rng(565656)
vals = rng.random(len(rows), dtype=np.float64)

# vals = np.ones(len(rows), dtype=np.float64)
vals[:7] = np.arange(1, 8, dtype=np.float64)  # make diagonal entries non-unit
A = csc_array((vals, (rows, cols)), shape=(N, N))

# TODO permute rows
# # ---------- Permute the matrix rows arbitrarily
# p_input = np.r_[5, 1, 7, 0, 2, 6, 4, 3]
# # p_input_inv = np.argsort(p_input)  # [3, 1, 4, 7, 6, 0, 5, 2]
# A = A[p_input, :]

# -----------------------------------------------------------------------------
#         Test Incomplete LU decomposition
# -----------------------------------------------------------------------------
# Get dense LU for comparison
Ad = A.toarray()
pd, Ld, Ud = la.lu(Ad, p_indices=True)

allclose(Ld[pd] @ Ud, Ad)

# NOTE spilu does not drop L entries that are < drop_tol?
#  * In SuperLU, 0 <= tol <= 1, because the drop tolerance is a fraction of the
#    maximum entry in each column.

# drop_tol = 0.0  # keep everything
# drop_tol = 0.08
drop_tol = 1.0  # drop everything off-diagonal -> FIXME does nothing?
# drop_tol = np.inf  # drop everything off-diagonal -> FIXME does nothing?

lu = spla.splu(A, permc_spec="NATURAL")
ilu = spla.spilu(A, drop_tol=drop_tol, permc_spec="NATURAL")

p, L, U = lu.perm_r, lu.L, lu.U
pi, Li, Ui = ilu.perm_r, ilu.L, ilu.U

assert_array_equal(p, np.arange(N))
assert_array_equal(p, pi)

print(f"---------- ilu ({drop_tol=}):")
print("L:")
print(L.toarray())
print("Li:")
print(Li.toarray())
print("U:")
print(U.toarray())
print("Ui:")
print(Ui.toarray())

if drop_tol == 0:
    allclose(Li.toarray(), L.toarray())
    allclose(Ui.toarray(), U.toarray())
    allclose(Li.toarray(), Ld)
    allclose(Ui.toarray(), Ud)
    allclose((Li[pi] @ Ui).toarray(), Ad)
elif drop_tol >= 1:  # only diagonals
    allclose(Li.toarray(), np.eye(Li.shape[0]))  # FIXME both fail!
    allclose(Ui.diagonal(), A.diagonal())

# =============================================================================
# =============================================================================
