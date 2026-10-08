import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from baselines import MAX_RATING, MIN_RATING, MODELS, rmse, summarise
from final_score import N_EPOCHS, N_FACTORS, REGULARIZATION
from mf import LEARNING_RATE, fit
from split import load_ratings, split_ratings

N_SEEDS = 10
N_COLD_USERS = 100
KEPT_RATINGS = [0, 1, 2, 5, 10, 20]
N_WORKERS = 10
MOVIE_BUCKETS = [(0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, 49), (50, 10**9)]

BASELINE_2, BASELINE_3 = list(MODELS)[1:]
MF_NAME = "matrix factorization"
NAMES = [BASELINE_2, BASELINE_3, MF_NAME]


def bucket_label(lo, hi):
    if lo == hi:
        return str(lo)
    return f"{lo}+" if hi >= 10**9 else f"{lo} to {hi}"


def predict_known_or_not(model, data):
    u = data["userId"].map(model["user_index"])
    m = data["movieId"].map(model["movie_index"])
    has_user, has_movie = u.notna().to_numpy(), m.notna().to_numpy()
    u_idx, m_idx = u.fillna(0).astype(int).to_numpy(), m.fillna(0).astype(int).to_numpy()

    out = np.full(len(data), model["mu"])
    out[has_user] += model["user_bias"][u_idx[has_user]]
    out[has_movie] += model["movie_bias"][m_idx[has_movie]]
    both = has_user & has_movie
    out[both] += np.sum(model["P"][u_idx[both]] * model["Q"][m_idx[both]], axis=1)
    return np.clip(out, MIN_RATING, MAX_RATING)


def all_predictions(model, train, test):
    predictions = {name: MODELS[name](train, test) for name in [BASELINE_2, BASELINE_3]}
    predictions[MF_NAME] = predict_known_or_not(model, test)
    return predictions


def fit_final(train, seed):
    model, _ = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION)
    return model


def cold_users_for(train, seed):
    rng = np.random.default_rng(seed + 1000)
    return rng.choice(np.sort(train["userId"].unique()), size=N_COLD_USERS, replace=False)


def keep_only(train, cold_users, kept, seed):
    is_cold = train["userId"].isin(cold_users)
    shuffled = train[is_cold].sample(frac=1, random_state=seed)
    return pd.concat([train[~is_cold], shuffled.groupby("userId").head(kept)])


def movie_job(seed):
    train, test = split_ratings(load_ratings(), seed)
    model = fit_final(train, seed)
    predictions = all_predictions(model, train, test)
    actual = test["rating"].to_numpy()
    movie_n = test["movieId"].map(train.groupby("movieId").size()).fillna(0).to_numpy()

    buckets = {}
    for lo, hi in MOVIE_BUCKETS:
        mask = (movie_n >= lo) & (movie_n <= hi)
        buckets[(lo, hi)] = {"share": mask.mean()}
        buckets[(lo, hi)].update({name: rmse(predictions[name][mask], actual[mask]) for name in NAMES})

    cold_mask = test["userId"].isin(cold_users_for(train, seed)).to_numpy()
    reference = {name: rmse(predictions[name][cold_mask], actual[cold_mask]) for name in NAMES}
    reference["n test"] = int(cold_mask.sum())
    return ("movies", seed, buckets, reference)


def user_job(seed, kept):
    train, test = split_ratings(load_ratings(), seed)
    cold_users = cold_users_for(train, seed)
    reduced = keep_only(train, cold_users, kept, seed)
    model = fit_final(reduced, seed)

    test_cold = test[test["userId"].isin(cold_users)]
    predictions = all_predictions(model, reduced, test_cold)
    actual = test_cold["rating"].to_numpy()
    return ("users", seed, kept, {name: rmse(predictions[name], actual) for name in NAMES})


if __name__ == "__main__":
    movie_runs, user_runs, reference = {}, {}, {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = [pool.submit(movie_job, seed) for seed in range(N_SEEDS)]
        futures += [pool.submit(user_job, seed, kept) for seed in range(N_SEEDS) for kept in KEPT_RATINGS]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result[0] == "movies":
                movie_runs[result[1]], reference[result[1]] = result[2], result[3]
            else:
                user_runs[(result[1], result[2])] = result[3]
            print(f"finished {done} of {len(futures)}", flush=True)

    print(f"\nNew movies: test ratings grouped by how many training ratings the movie has, {N_SEEDS} shuffles")
    print(f"{'training ratings':<20}{'share of test':<16}" + "".join(f"{name:<34}" for name in NAMES))
    for bucket in MOVIE_BUCKETS:
        share = np.mean([run[bucket]["share"] for run in movie_runs.values()])
        cells = "".join(
            f"{summarise([run[bucket][name] for run in movie_runs.values()]):<34}" for name in NAMES
        )
        print(f"{bucket_label(*bucket):<20}{share:<16.1%}{cells}")

    print(f"\nNew users: {N_COLD_USERS} users per shuffle keep only k training ratings, scored on their hidden ratings")
    print(f"{'ratings kept':<20}" + "".join(f"{name:<34}" for name in NAMES))
    for kept in KEPT_RATINGS:
        cells = "".join(
            f"{summarise([user_runs[(seed, kept)][name] for seed in range(N_SEEDS)]):<34}" for name in NAMES
        )
        print(f"{kept:<20}{cells}")
    cells = "".join(f"{summarise([reference[seed][name] for seed in range(N_SEEDS)]):<34}" for name in NAMES)
    print(f"{'all (full history)':<20}{cells}")
    print("test ratings per shuffle for these users:", summarise([reference[seed]["n test"] for seed in range(N_SEEDS)]))
