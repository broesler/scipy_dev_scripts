# Part of the scikit-sparse project.
# Copyright (C) 2025 Bernard Roesler. All rights reserved.
# See pyproject.toml for full author list and LICENSE.txt for license details.
# SPDX-License-Identifier: BSD-2-Clause
#
# =============================================================================
#     File: compare_umfpack.py
#  Created: 2025-10-30 14:18
# =============================================================================

"""
Compare the scikit-sparse UMFPACK interface with the existing scikit-umfpack
interface.
"""

import gc
import timeit
import tracemalloc
import warnings
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import LaplacianNd, splu
from scipy.sparse.linalg._dsolve import _superlu
from tqdm import tqdm

SEED = 565656

SAVE_FIGS = False

DATA_PATH = Path(__file__).absolute().parent


def measure_perf(func, N_repeats=5, N_samples=None):
    """Measure time and memory usage of a function.

    Parameters
    ----------
    func : callable
        The function to measure.

    Returns
    -------
    time : float
        The minimum execution time in seconds.
    peak_mb : float
        The peak memory usage in megabytes.
    """
    # Measure timing (multiple runs)
    timer = timeit.Timer(func)
    if N_samples is None:
        N_samples, _ = timer.autorange()
    ts = timer.repeat(repeat=N_repeats, number=N_samples)
    ts = np.array(ts) / N_samples
    time = np.min(ts)

    # Measure memory usage (single pass)
    gc.collect()  # force garbage collection before measuring
    tracemalloc.start()

    try:
        func()
    except Exception:
        tracemalloc.stop()
        raise

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024**2)  # convert to MB

    return time, peak_mb


def gssv_solve(A, b, permc_spec=None):
    """Solve Ax = b using SuperLU gssv interface."""
    if A.format == "csc":
        flag = 1  # CSC format
    else:
        flag = 0  # CSR format

    N = A.shape[0]
    indices = A.indices.astype(np.intc, copy=False)
    indptr = A.indptr.astype(np.intc, copy=False)
    options = {"ColPerm": permc_spec}
    x, info = _superlu.gssv(N, A.nnz, A.data, indices, indptr, b, flag, options=options)
    return x


def superlu_solve(A, b, permc_spec=None):
    """Solve Ax = b using SuperLU spsolve interface."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SparseEfficiencyWarning)
        return splu(A, permc_spec=permc_spec).solve(b)


def run_comparison(df_file, force_update=False):
    """Run performance comparison between scikit-sparse and scikits-umfpack."""
    if not force_update and df_file.exists():
        print(f"Loaded existing results from: {df_file}")
        return pd.read_pickle(df_file)

    assert df_file.parent.exists(), f"Data path does not exist: {df_file.parent}"
    print(f"Running performance tests for {df_file}...")
    Ns = np.unique(np.logspace(1, 4, num=10, dtype=int))
    sqrtNs = np.unique([int(np.sqrt(N)) for N in Ns])

    results = []

    funcs = {
        "superlu": superlu_solve,
        "gssv": gssv_solve,
    }

    permc_specs = [None, "COLAMD"]

    # Test performance of multiple solves
    for sqrtN in tqdm(sqrtNs):
        A = -LaplacianNd((sqrtN, sqrtN), dtype=float).tosparse().tocsc()
        A[-1, -1] += 1.0  # make sure A is non-singular
        N = A.shape[0]
        x_col = np.arange(1, N + 1, dtype=float)

        for K in tqdm([1, 10, 100, 1000], leave=False):
            expect_x = np.outer(x_col, np.arange(1, K))  # many RHS columns
            B = A @ expect_x  # C order

            for key, func in tqdm(funcs.items(), leave=False):
                for fmt in tqdm(["csc", "csr"], leave=False):
                    for permc_spec in tqdm(permc_specs, leave=False):
                        A_fmt = A.asformat(fmt)
                        func_fmt = partial(func, A_fmt, B, permc_spec=permc_spec)
                        time, mem = measure_perf(func_fmt)
                        results.append(
                            {
                                "function": key,
                                "format": fmt,
                                "permc_spec": str(permc_spec),
                                "N": N,
                                "K": K,
                                "time": time,
                                "memory": mem,
                            }
                        )

    # Build the results DataFrame
    df = (
        pd.DataFrame(results)
        .set_index(["function", "format", "permc_spec", "N", "K"])
        .sort_index()
    )
    df.columns.name = "metric"

    df.to_pickle(df_file)
    return df


# -----------------------------------------------------------------------------
#         Run the Tests
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    df = run_comparison(DATA_PATH / "superlu_solve_results.pkl", force_update=False)

    # df = pd.concat(
    #     [
    #         df.xs(1, level="K", drop_level=False),
    #         df.xs(10, level="K", drop_level=False)
    #     ]
    # ).sort_index()

    for fignum, fmt in enumerate(["CSC", "CSR"]):
        fig, axs = plt.subplots(num=fignum, nrows=2, sharex=True, clear=True)
        fig.suptitle(
            "SuperLU.solve vs. _superlu.gssv\n"
            f"A (N, N) 2D Laplacian in {fmt} format, b (N, K)",
        )
        fig.set_size_inches((6.4, 8), forward=True)

        for i, col in enumerate(["time", "memory"]):
            sns.lineplot(
                ax=axs[i],
                data=df.xs(("COLAMD", fmt.lower()), level=("permc_spec", "format")),
                x="N",
                y=col,
                hue="K",
                style="function",
                markers=True,
                palette="flare",
                hue_norm=mpl.colors.LogNorm(),
                legend=(i == 1),
            )
            axs[i].grid(True, which="both")
            axs[i].set(yscale="log")

        axs[0].set(
            ylabel="time [s]",
        )

        axs[1].legend()
        axs[1].set(
            xscale="log",
            xlabel="Number of Rows/Columns (N)",
            ylabel="peak memory [MB]",
        )

        plt.show()

        if SAVE_FIGS:
            fig_file = DATA_PATH / f"superlu_solve_perf_{fmt}.pdf"
            fig.savefig(fig_file)
            print(f"Saved figure to: {fig_file}")

# =============================================================================
# =============================================================================
