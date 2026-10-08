import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from als import fit, interaction_matrix, loss, solve_side
from baselines import summarise
from explain_content import LIKED_FROM
from hit_rate import N_SEEDS
from library_hit_rate import prepare, hit_rate
from mf import index_ids
from split import load_ratings, split_ratings

N_WORKERS = 10
N_FACTORS = 32
REGULARIZATION = 0.1
CONFIDENCE = 10.0
N_ITERATIONS = 15

VARIANTS = {"rated at all": None, "liked only": LIKED_FROM}
MODELS = ["my ALS", "library ALS, exact solves", "library ALS, default solver"]
NAMES = [f"{model}, {variant}" for variant in VARIANTS for model in MODELS]


def checks(seed=0):
    rng = np.random.default_rng(seed)
    dense = (rng.random((6, 9)) < 0.3).astype(float)
    users, items = rng.normal(size=(6, 3)), rng.normal(size=(9, 3))
    weights = np.where(dense == 1, CONFIDENCE, 1.0)
    brute = np.sum(weights * (dense - users @ items.T) ** 2) + REGULARIZATION * (np.sum(users**2) + np.sum(items**2))
    fast = loss(users, items, csr_matrix(dense), CONFIDENCE, REGULARIZATION)
    print(f"loss formula on a 6 x 9 matrix: {fast:.6f}, brute force: {brute:.6f}")

    train, _ = split_ratings(load_ratings(), seed)
    user_index, movie_index = index_ids(train["userId"]), index_ids(train["movieId"])
    user_items = interaction_matrix(train, user_index, movie_index)
    _, _, history = fit(user_items, N_FACTORS, REGULARIZATION, CONFIDENCE, N_ITERATIONS, seed, track_loss=True)
    rises = sum(b > a + 1e-9 for a, b in zip(history, history[1:]))
    print(f"loss on shuffle {seed}: {history[0]:.0f} after iteration 1, {history[-1]:.0f} after iteration {N_ITERATIONS}, rises: {rises}")

    user_items32 = user_items.astype(np.float32).tocsr()
    model = AlternatingLeastSquares(
        factors=N_FACTORS, regularization=REGULARIZATION, alpha=CONFIDENCE, iterations=N_ITERATIONS,
        use_cg=False, num_threads=1, random_state=seed,
    )
    model.fit(user_items32, show_progress=False)
    rows = range(0, user_items.shape[0], 61)
    differences = [
        np.abs(
            model.recalculate_user(u, user_items32[u])
            - solve_side(model.item_factors.astype(np.float64), user_items[u : u + 1], CONFIDENCE, REGULARIZATION)[0]
        ).max()
        for u in rows
    ]
    print(f"largest difference from the library's exact solve for {len(rows)} users: {max(differences):.1e}\n")


def run_seed(seed):
    train, test = split_ratings(load_ratings(), seed)
    prepared = prepare(train, test)
    user_index, movie_index = index_ids(train["userId"]), index_ids(train["movieId"])
    pool_cols = np.array([movie_index[m] for m in prepared["pool_ids"]])

    results, seconds = {}, {}
    for variant, liked_from in VARIANTS.items():
        user_items = interaction_matrix(train, user_index, movie_index, liked_from)

        start = time.perf_counter()
        users, items, _ = fit(user_items, N_FACTORS, REGULARIZATION, CONFIDENCE, N_ITERATIONS, seed)
        seconds[f"my ALS, {variant}"] = time.perf_counter() - start
        pool_items = items[pool_cols]
        results[f"my ALS, {variant}"] = hit_rate(prepared, lambda user_id: pool_items @ users[user_index[user_id]])

        for name, use_cg in [("library ALS, exact solves", False), ("library ALS, default solver", True)]:
            start = time.perf_counter()
            model = AlternatingLeastSquares(
                factors=N_FACTORS,
                regularization=REGULARIZATION,
                alpha=CONFIDENCE,
                iterations=N_ITERATIONS,
                use_cg=use_cg,
                num_threads=1,
                random_state=seed,
            )
            model.fit(user_items.astype(np.float32).tocsr(), show_progress=False)
            seconds[f"{name}, {variant}"] = time.perf_counter() - start
            pool_items = model.item_factors[pool_cols]
            results[f"{name}, {variant}"] = hit_rate(
                prepared, lambda user_id: pool_items @ model.user_factors[user_index[user_id]]
            )
    return seed, results, seconds


if __name__ == "__main__":
    checks()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        runs = sorted(pool.map(run_seed, range(N_SEEDS)))

    print(f"Hit rate at 10, {N_FACTORS} hidden numbers, regularization {REGULARIZATION}, confidence {CONFIDENCE:g}, {N_ITERATIONS} iterations, no tuning")
    print(f"{N_SEEDS} shuffles, mean (lowest to highest)\n")
    print(f"{'model':<46}{'hit rate at 10':<26}{'recall at 10':<26}{'fit seconds'}")
    for name in NAMES:
        hits = [run[1][name][0] for run in runs]
        recalls = [run[1][name][1] for run in runs]
        print(f"{name:<46}{summarise(hits):<26}{summarise(recalls):<26}{np.mean([run[2][name] for run in runs]):.1f}")

    print("\nhit rate of my ALS minus the library, per shuffle (mean, lowest to highest, shuffles where mine is higher):")
    for variant in VARIANTS:
        mine = np.array([run[1][f"my ALS, {variant}"][0] for run in runs])
        for other in MODELS[1:]:
            theirs = np.array([run[1][f"{other}, {variant}"][0] for run in runs])
            print(f"  {variant:<14} vs {other:<30}{summarise(mine - theirs)}  {int((mine > theirs).sum())}/{N_SEEDS}")
