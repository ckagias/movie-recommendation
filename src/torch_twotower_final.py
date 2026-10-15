import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from baselines import summarise
from library_hit_rate import MOST_RATED, hit_rate, prepare
from split import load_ratings, split_ratings, split_validation
from torch_data import index_ids
from torch_twotower import EVAL_EVERY, N_FACTORS, scorer, train

N_WORKERS = 10
N_SEEDS = 10
MAX_EPOCHS = 200
LEARNING_RATE = 0.003
REGULARIZATION = 0.003
N_NEGATIVES = 5

VARIANTS = {"rated at all": False, "liked only": True}
TOWERS = {"embeddings": 0, "MLP towers": 128}
NAMES = [MOST_RATED] + [f"two-tower {t}, {v}" for t in TOWERS for v in VARIANTS]


def run_seed(seed):
    torch.set_num_threads(1)
    train_all, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train_all, seed)
    validation_prepared, test_prepared = prepare(inner, validation), prepare(train_all, test)
    inner_users, inner_movies = index_ids(inner["userId"]), index_ids(inner["movieId"])
    full_users, full_movies = index_ids(train_all["userId"]), index_ids(train_all["movieId"])

    popularity = test_prepared["popularity"].astype(float)
    results, chosen = {MOST_RATED: hit_rate(test_prepared, lambda user_id: popularity)}, {}

    for tower_name, hidden in TOWERS.items():
        for variant, liked_only in VARIANTS.items():
            name = f"two-tower {tower_name}, {variant}"
            curve = []

            def record(epoch, model):
                curve.append(hit_rate(validation_prepared, scorer(model, inner_users, inner_movies, validation_prepared))[0])

            train(inner, inner_users, inner_movies, liked_only, N_FACTORS, hidden, LEARNING_RATE, REGULARIZATION,
                  MAX_EPOCHS, seed, n_negatives=N_NEGATIVES, callback=record)
            best_epoch = (int(np.argmax(curve)) + 1) * EVAL_EVERY

            model = train(train_all, full_users, full_movies, liked_only, N_FACTORS, hidden, LEARNING_RATE,
                          REGULARIZATION, best_epoch, seed, n_negatives=N_NEGATIVES)
            results[name] = hit_rate(test_prepared, scorer(model, full_users, full_movies, test_prepared))
            chosen[name] = best_epoch
    return seed, results, chosen


if __name__ == "__main__":
    start = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        runs = sorted(pool.map(run_seed, range(N_SEEDS)))
    print(f"{time.time() - start:.0f} s")

    print(f"\nTop 10 from movies with at least 20 training ratings, {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"{'method':<40}{'hit rate at 10':<26}{'recall at 10':<26}{'median ratings of picks'}")
    for name in NAMES:
        hits = [run[1][name][0] for run in runs]
        recalls = [run[1][name][1] for run in runs]
        popularity = [run[1][name][2] for run in runs]
        print(f"{name:<40}{summarise(hits):<26}{summarise(recalls):<26}{np.mean(popularity):.0f}")

    print("\nchosen number of epochs per shuffle:")
    for name in NAMES[1:]:
        print(f"{name:<40}{[run[2][name] for run in runs]}")
