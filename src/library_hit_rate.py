import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix

from baselines import summarise
from explain_content import LIKED_FROM
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from hit_rate import MIN_TRAIN_RATINGS, N_SEEDS, top_n
from mf import LEARNING_RATE, fit, index_ids
from split import load_ratings, split_ratings, split_validation

N_WORKERS = 10

ALS_GRID = [
    {"factors": f, "regularization": r, "alpha": a}
    for f, r, a in itertools.product([32, 64], [0.01, 0.1, 1.0], [1, 10, 40])
]
BPR_GRID = [
    {"factors": f, "learning_rate": lr, "regularization": r, "iterations": it}
    for f, lr, r, it in itertools.product([32, 64, 128], [0.01, 0.05], [0.01, 0.1], [100, 300])
]
KNN_GRID = [{"K": k} for k in [20, 50, 100, 200]]

LIBRARY_MODELS = {"ALS": ALS_GRID, "BPR": BPR_GRID, "item-item cosine": KNN_GRID}
VARIANTS = {"rated at all": False, "liked only": True}

MOST_RATED = "most rated"
MINE = "my matrix factorization"
NAMES = [MOST_RATED, MINE] + [f"{m}, {v}" for m in LIBRARY_MODELS for v in VARIANTS]


def interaction_matrix(frame, user_index, movie_index, liked_only):
    if liked_only:
        frame = frame[frame["rating"] >= LIKED_FROM]
    rows = frame["userId"].map(user_index).to_numpy()
    cols = frame["movieId"].map(movie_index).to_numpy()
    shape = (len(user_index), len(movie_index))
    return csr_matrix((np.ones(len(frame), dtype=np.float32), (rows, cols)), shape=shape)


def prepare(fit_pile, held_out):
    counts = fit_pile.groupby("movieId").size()
    counts = counts[counts >= MIN_TRAIN_RATINGS]
    pool_ids = counts.index.to_numpy()
    position = {movie_id: p for p, movie_id in enumerate(pool_ids)}

    liked = held_out[held_out["rating"] >= LIKED_FROM]
    relevant = (
        liked[liked["movieId"].isin(position)]
        .groupby("userId")["movieId"]
        .agg(lambda ids: {position[m] for m in ids})
    )
    seen = {}
    for user_id, movies in fit_pile.groupby("userId")["movieId"]:
        if user_id in relevant.index:
            mask = np.zeros(len(pool_ids), dtype=bool)
            mask[[position[m] for m in movies if m in position]] = True
            seen[user_id] = mask
    return {"pool_ids": pool_ids, "popularity": counts.to_numpy(), "relevant": relevant, "seen": seen}


def hit_rate(prepared, score_for_user):
    flags, recalls, pick_popularity = [], [], []
    for user_id, wanted in prepared["relevant"].items():
        shown = top_n(score_for_user(user_id), prepared["seen"][user_id], prepared["popularity"])
        hits = len(set(shown) & wanted)
        flags.append(hits > 0)
        recalls.append(hits / len(wanted))
        pick_popularity.append(np.median(prepared["popularity"][shown]))
    return np.mean(flags), np.mean(recalls), np.median(pick_popularity)


def library_scorer(kind, params, fit_pile, prepared, liked_only, seed):
    user_index, movie_index = index_ids(fit_pile["userId"]), index_ids(fit_pile["movieId"])
    user_items = interaction_matrix(fit_pile, user_index, movie_index, liked_only)
    pool_cols = np.array([movie_index[m] for m in prepared["pool_ids"]])

    if kind == "item-item cosine":
        model = CosineRecommender(num_threads=1, **params)
        model.fit(user_items, show_progress=False)
        return lambda user_id: np.asarray((user_items[user_index[user_id]] @ model.similarity).todense()).ravel()[pool_cols]

    cls = AlternatingLeastSquares if kind == "ALS" else BayesianPersonalizedRanking
    model = cls(num_threads=1, random_state=seed, **params)
    model.fit(user_items, show_progress=False)
    pool_items = model.item_factors[pool_cols]
    return lambda user_id: pool_items @ model.user_factors[user_index[user_id]]


def run_seed(seed):
    train, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train, seed)
    validation_prepared = prepare(inner, validation)
    test_prepared = prepare(train, test)

    results, chosen, seconds = {}, {}, {}
    popularity = test_prepared["popularity"].astype(float)
    results[MOST_RATED] = hit_rate(test_prepared, lambda user_id: popularity)

    start = time.perf_counter()
    model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    in_model = np.array([model["movie_index"][m] for m in test_prepared["pool_ids"]])
    pool_bias, pool_Q = model["movie_bias"][in_model], model["Q"][in_model]

    def mf_scores(user_id):
        u = model["user_index"][user_id]
        return model["mu"] + model["user_bias"][u] + pool_bias + pool_Q @ model["P"][u]

    results[MINE] = hit_rate(test_prepared, mf_scores)
    seconds[MINE] = time.perf_counter() - start

    for kind, grid in LIBRARY_MODELS.items():
        for variant, liked_only in VARIANTS.items():
            name = f"{kind}, {variant}"
            validation_scores = [
                hit_rate(validation_prepared, library_scorer(kind, params, inner, validation_prepared, liked_only, seed))[0]
                for params in grid
            ]
            best = grid[int(np.argmax(validation_scores))]
            start = time.perf_counter()
            scorer = library_scorer(kind, best, train, test_prepared, liked_only, seed)
            seconds[name] = time.perf_counter() - start
            results[name] = hit_rate(test_prepared, scorer)
            chosen[name] = best
    return seed, results, chosen, seconds


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        runs = sorted(pool.map(run_seed, range(N_SEEDS)))

    print(f"Top 10 from movies with at least {MIN_TRAIN_RATINGS} training ratings, {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"{'method':<34}{'hit rate at 10':<26}{'recall at 10':<26}{'median ratings of picks'}")
    for name in NAMES:
        hits = [run[1][name][0] for run in runs]
        recalls = [run[1][name][1] for run in runs]
        popularity = [run[1][name][2] for run in runs]
        print(f"{name:<34}{summarise(hits):<26}{summarise(recalls):<26}{np.mean(popularity):.0f}")

    mf_hits = np.array([run[1][MINE][0] for run in runs])
    most_rated = np.array([run[1][MOST_RATED][0] for run in runs])
    print("\nhit rate difference per shuffle, library model minus reference, shuffles where it is higher:")
    print(f"{'method':<34}{'minus my matrix factorization':<34}{'minus most rated'}")
    for name in NAMES[2:]:
        hits = np.array([run[1][name][0] for run in runs])
        print(
            f"{name:<34}{summarise(hits - mf_hits)} {int((hits > mf_hits).sum())}/10".ljust(68)
            + f"{summarise(hits - most_rated)} {int((hits > most_rated).sum())}/10"
        )

    print("\nsettings chosen on validation, per shuffle:")
    for name in NAMES[2:]:
        print(f"  {name}:")
        for run in runs:
            print(f"    {run[0]}: {run[2][name]}")

    print("\nfit seconds on the full training pile (10 jobs at once):")
    for name in [MINE] + NAMES[2:]:
        print(f"  {name:<34}{np.mean([run[3][name] for run in runs]):.1f}")
