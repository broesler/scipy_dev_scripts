#!/usr/bin/env python3
# =============================================================================
#     File: spilu_bug.py
#  Created: 2025-12-16 12:59
#   Author: Bernie Roesler
#
"""Test incomplete LU decomposition with SciPy spilu."""
# =============================================================================

import numpy as np
from numpy.testing import assert_allclose as _assert_allclose
from numpy.testing import assert_array_equal
from scipy import linalg as la
from scipy.sparse import csc_array
from scipy.sparse import linalg as spla


def assert_allclose(a, b, atol=1e-15):
    """Check if two arrays are close to each other."""
    return _assert_allclose(a, b, atol=atol)


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

# rng = np.random.default_rng(565656)
# vals = rng.random(len(rows), dtype=np.float64)
vals = np.ones(len(rows), dtype=np.float64)

# make diagonal entries non-unit to avoid pivoting
vals[:7] = np.arange(1, 8, dtype=np.float64)

A = csc_array((vals, (rows, cols)), shape=(N, N))

# -----------------------------------------------------------------------------
#         Test Incomplete LU decomposition
# -----------------------------------------------------------------------------
# Get dense LU for comparison
Ad = A.toarray()
pd, Ld, Ud = la.lu(Ad, p_indices=True)

assert_allclose(Ld[pd] @ Ud, Ad)

# Get (complete) sparse LU for comparison
lu = spla.splu(A, permc_spec="NATURAL")
p, L, U = lu.perm_r, lu.L, lu.U

assert_array_equal(p, np.arange(N))  # no column permutation

print("A:")
print(Ad)
print("L:")
print(L.toarray())
print("U:")
print(U.toarray())

# NOTE spilu does not drop L entries that are < drop_tol?
#  * In SuperLU, 0 <= tol <= 1, because the drop tolerance is a fraction of the
#    maximum entry in each column.
# drop_tol = 0.0     # keep everything
# drop_tol = 0.08    # arbitrary, should drop 1 entry L[3, 6]
# drop_tol = 1.0     # drop everything off-diagonal -> FIXME does nothing?
# drop_tol = np.inf  # drop everything off-diagonal -> FIXME does nothing?

for drop_tol in [0.0, 1.0, np.inf]:
    # Compute the incomplete LU decomposition
    ilu = spla.spilu(A, drop_tol=drop_tol, permc_spec="NATURAL")
    pi, Li, Ui = ilu.perm_r, ilu.L, ilu.U

    assert_array_equal(p, pi)

    if drop_tol == 0:
        assert_allclose(Li.toarray(), L.toarray())
        assert_allclose(Ui.toarray(), U.toarray())
        assert_allclose(Li.toarray(), Ld)
        assert_allclose(Ui.toarray(), Ud)
        assert_allclose((Li[pi] @ Ui).toarray(), Ad)
    elif drop_tol >= 1:  # only diagonals
        print(f"---------- ilu ({drop_tol=})")
        print("Li:")
        print(Li.toarray())
        print("Ui:")
        print(Ui.toarray())
        # assert_allclose(Li.toarray(), np.eye(Li.shape[0]))  # FIXME both fail!
        # assert_allclose(Ui.diagonal(), A.diagonal())

# =============================================================================
# =============================================================================
