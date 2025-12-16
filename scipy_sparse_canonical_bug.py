#!/usr/bin/env python3
# =============================================================================
#     File: scipy_sparse_canonical_bug.py
#  Created: 2025-09-18 13:23
#   Author: Bernie Roesler
#
"""Reproduce a bug in SciPy's sparse matrix canonicalization."""
# =============================================================================

import numpy as np
from scipy import sparse
from numpy.testing import assert_array_equal, assert_allclose

# See: Davis, Timothy A. (2006). Direct Methods for Sparse Linear Systems,
# pp 708 (Equation 2.1).
N = 11
rows = np.array([5, 6, 2, 7, 9, 10, 5, 9, 7, 10, 8, 9, 10, 9, 10, 10])
cols = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 9])

# vals = np.ones(len(rows), dtype=np.float64)
rng = np.random.default_rng(565656)
vals = rng.random(len(rows), dtype=np.float64)

L = sparse.coo_array((vals, (rows, cols)), shape=(N, N))

# -----------------------------------------------------------------------------
#         Create a canonical matrix A
# -----------------------------------------------------------------------------
A = L + L.T   # make it symmetric
A.setdiag(N)  # make it strongly positive definite
A = A.tocsc()

assert A.has_canonical_format
assert A.has_sorted_indices

print(f"{A.indices=}")

# -----------------------------------------------------------------------------
#         Create a non-canonical matrix B
# -----------------------------------------------------------------------------
B = (L + L.T).tocsc()

assert B.has_canonical_format
assert B.has_sorted_indices

Bp = B.indptr
Bi = B.indices

for p in range(N):
    col_idx = Bi[Bp[p]:Bp[p+1]]
    if np.any(np.diff(col_idx) < 0):
        raise RuntimeError('B does not have sorted indices.')

# import ipdb; ipdb.set_trace()
B.setdiag(N)  # XXX this line creates duplicate/unsorted entries!

assert B.has_canonical_format
assert B.has_sorted_indices

# # Show that B does *not* have sorted indices
# assert_array_equal(A.indptr, B.indptr, strict=True)
# assert not np.array_equal(A.indices, B.indices)
# assert not np.allclose(A.data, B.data)

print(f"{B.indices=}")  # XXX NOT SORTED despite B.has_sorted_indices=True!

# Actual test of B.indices
expect_unsorted = [0, 1, 2, 3, 4, 5, 6, 7, 9]

Bp = B.indptr
Bi = B.indices

for p in range(N):
    col_idx = Bi[Bp[p]:Bp[p+1]]
    if np.all(np.diff(col_idx) >= 0):
        # assert p not in expect_unsorted
        print(f"Column {p} is sorted: {col_idx}")
    else:
        # assert p in expect_unsorted
        print(f"Column {p} is not sorted: {col_idx}")

# # -----------------------------------------------------------------------------
# #         Create a copy of B and canonicalize it
# # -----------------------------------------------------------------------------
# C = B.copy()

# # NOTE the copy operation resets the flags, so when we check them, a fresh
# # check is performed internally and sets the flags correctly.
# assert not C.has_canonical_format
# assert not C.has_sorted_indices

# C.sort_indices()
# assert C.has_sorted_indices

# C.sum_duplicates()
# assert C.has_canonical_format

# # Check that A and C are identical
# assert_array_equal(A.indptr, C.indptr, strict=True)
# assert_array_equal(A.indices, C.indices, strict=True)
# assert_allclose(A.data, C.data, strict=True)

# print(f"{C.indices=}")

# # -----------------------------------------------------------------------------
# #         Create a "copy" of B and canonicalize it in place
# # -----------------------------------------------------------------------------
# # D = sparse.csc_array(B)  # XXX DOES NOT MAKE A DEEP COPY
# D = sparse.csc_array(B, copy=True)  # makes a deep copy
# assert D is not B  # ensure it's a different object

# assert_array_equal(B.indptr, D.indptr, strict=True)
# assert_array_equal(B.indices, D.indices, strict=True)
# assert_allclose(B.data, D.data, strict=True)

# assert not D.has_canonical_format
# assert not D.has_sorted_indices

# D.sum_duplicates()  # sort indices and sum duplicates

# assert D.has_canonical_format
# assert D.has_sorted_indices

# # Check that A and D are identical
# assert_array_equal(A.indptr, D.indptr, strict=True)
# assert_array_equal(A.indices, D.indices, strict=True)
# assert_allclose(A.data, D.data, strict=True)

# print(f"{A.indices=}")
# print(f"{B.indices=}")
# print(f"{C.indices=}")
# print(f"{D.indices=}")

# # =============================================================================
# # =============================================================================
