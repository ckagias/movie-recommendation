import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from surprise import BaselineOnly, Dataset, Reader, SVD

from baselines import MODELS, rmse, summarise
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit, predict
from split import load_ratings, split_ratings, split_validation

N_SEEDS = 10
N_WORKERS = 10

GRID = {
    "n_factors": [20, 50, 100],
    "reg_all": [0.02, 0.05, 0.1, 0.15],
    "lr_all": [0.005, 0.01, 0.02],
    "n_epochs": [20, 40, 60],
}
GRID_CONFIGS = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
MY_SETTINGS = {
    "n_factors": N_FACTORS,
    "reg_all": REGULARIZATION,
    "lr_all": LEARNING_RATE,
    "n_epochs": N_EPOCHS,
}

BASELINE_3 = list(MODELS)[-1]
MINE = "my matrix factorization"
NAMES = [
    BASELINE_3,
    "Surprise BaselineOnly",
    MINE,
    "Surprise SVD, library defaults",
    "Surprise SVD, my settings",
    "Surprise SVD, tuned on validation",
]


def to_trainset(frame):
    reader = Reader(rating_scale=(0.5, 5.0))
    return Dataset.load_from_df(frame[["userId", "movieId", "rating"]], reader).build_full_trainset()


def surprise_predict(algo, trainset, frame):
    algo.fit(trainset)
    pairs = zip(frame["userId"], frame["movieId"], frame["rating"])
    return np.array([p.est for p in algo.test(pairs)])


def run_seed(seed):
    train, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train, seed)
    actual = test["rating"].to_numpy()
    trainset = to_trainset(train)
    scores, seconds = {}, {}

    scores[BASELINE_3] = rmse(MODELS[BASELINE_3](train, test), actual)

    start = time.perf_counter()
    scores["Surprise BaselineOnly"] = rmse(surprise_predict(BaselineOnly(verbose=False), trainset, test), actual)
    seconds["Surprise BaselineOnly"] = time.perf_counter() - start

    start = time.perf_counter()
    model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    scores[MINE] = rmse(predict(model, test), actual)
    seconds[MINE] = time.perf_counter() - start

    start = time.perf_counter()
    algo = SVD(random_state=seed, verbose=False)
    scores["Surprise SVD, library defaults"] = rmse(surprise_predict(algo, trainset, test), actual)
    seconds["Surprise SVD, library defaults"] = time.perf_counter() - start

    start = time.perf_counter()
    algo = SVD(random_state=seed, verbose=False, **MY_SETTINGS)
    scores["Surprise SVD, my settings"] = rmse(surprise_predict(algo, trainset, test), actual)
    seconds["Surprise SVD, my settings"] = time.perf_counter() - start

    inner_set = to_trainset(inner)
    validation_actual = validation["rating"].to_numpy()
    validation_scores = [
        rmse(surprise_predict(SVD(random_state=seed, verbose=False, **cfg), inner_set, validation), validation_actual)
        for cfg in GRID_CONFIGS
    ]
    best = GRID_CONFIGS[int(np.argmin(validation_scores))]
    algo = SVD(random_state=seed, verbose=False, **best)
    scores["Surprise SVD, tuned on validation"] = rmse(surprise_predict(algo, trainset, test), actual)
    return seed, scores, seconds, best, float(np.min(validation_scores))


if __name__ == "__main__":
    runs = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(run_seed, seed) for seed in range(N_SEEDS)]
        for done, future in enumerate(as_completed(futures), start=1):
            seed, scores, seconds, best, _ = future.result()
            runs[seed] = (scores, seconds, best)
            print(f"finished {done} of {N_SEEDS}", flush=True)

    print(f"\nRMSE on the test pile, {N_SEEDS} shuffles, mean (lowest to highest)")
    print(f"{'model':<38}{'rmse':<26}{'mine is lower in':<20}{'fit seconds (10 jobs at once)'}")
    for name in NAMES:
        values = np.array([runs[s][0][name] for s in sorted(runs)])
        mine = np.array([runs[s][0][MINE] for s in sorted(runs)])
        wins = "" if name == MINE else f"{int((mine < values).sum())} of {N_SEEDS}"
        times = [runs[s][1][name] for s in runs if name in runs[s][1]]
        seconds = f"{np.mean(times):.1f}" if times else ""
        print(f"{name:<38}{summarise(values):<26}{wins:<20}{seconds}")

    print("\nmy matrix factorization minus each library model, per shuffle (negative = mine is better):")
    mine = np.array([runs[s][0][MINE] for s in sorted(runs)])
    for name in NAMES:
        if name == MINE:
            continue
        diff = mine - np.array([runs[s][0][name] for s in sorted(runs)])
        print(f"  {name:<36}{summarise(diff)}")

    print("\nsettings chosen for the tuned SVD, per shuffle:")
    for seed in sorted(runs):
        print(" ", seed, runs[seed][2])
