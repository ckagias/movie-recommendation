import resource
import time

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from als import CONFIDENCE, N_FACTORS, REGULARIZATION
from split_32m import TRAIN, split

N_ITERATIONS = 3


def peak_memory_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def train_matrix(user, movie, pile):
    keep = pile == TRAIN
    user_ids, rows = np.unique(user[keep], return_inverse=True)
    movie_ids, cols = np.unique(movie[keep], return_inverse=True)
    ones = np.ones(keep.sum(), dtype=np.float32)
    matrix = csr_matrix((ones, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    return matrix, user_ids, movie_ids


if __name__ == "__main__":
    start = time.time()
    user, movie, rating, pile = split()
    print(f"load and split: {time.time() - start:.1f} s")

    start = time.time()
    matrix, user_ids, movie_ids = train_matrix(user, movie, pile)
    print(f"sparse matrix: {time.time() - start:.1f} s")
    print("shape:", matrix.shape, "| stored ratings:", matrix.nnz)
    print(f"filled share of the table: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.3%}")
    print(f"matrix size in memory: {(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / 1024**3:.2f} GB")
    print(f"peak memory so far: {peak_memory_gb():.1f} GB")

    model = AlternatingLeastSquares(
        factors=N_FACTORS,
        regularization=REGULARIZATION,
        alpha=CONFIDENCE,
        iterations=N_ITERATIONS,
        random_state=0,
    )
    print(f"\nfitting ALS: {N_FACTORS} factors, {N_ITERATIONS} iterations, threads {model.num_threads}")
    stamps = [time.time()]
    model.fit(matrix, show_progress=False, callback=lambda *args: stamps.append(time.time()))
    for i in range(1, len(stamps)):
        print(f"iteration {i}: {stamps[i] - stamps[i - 1]:.1f} s")
    print(f"peak memory after fit: {peak_memory_gb():.1f} GB")
    print("factor tables:", model.user_factors.shape, model.item_factors.shape, model.user_factors.dtype)
