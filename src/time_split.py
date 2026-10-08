import numpy as np
import pandas as pd

from baselines import MODELS, rmse, summarise
from mf import LEARNING_RATE, fit, predict
from split import TEST_FRACTION, load_ratings

N_FACTORS = 20
REGULARIZATION = 0.1
N_EPOCHS = 55
N_SEEDS = 5

MF_NAME = "matrix factorization"


def split_global(ratings, held_out_fraction=TEST_FRACTION):
    ordered = ratings.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n_train = len(ordered) - int(round(held_out_fraction * len(ordered)))
    return ordered.iloc[:n_train].reset_index(drop=True), ordered.iloc[n_train:].reset_index(drop=True)


def split_per_user(ratings, held_out_fraction=TEST_FRACTION):
    ordered = ratings.sort_values(["userId", "timestamp"], kind="stable").reset_index(drop=True)
    rank_from_end = ordered.groupby("userId").cumcount(ascending=False)
    n_ratings = ordered.groupby("userId")["rating"].transform("size")
    is_test = rank_from_end < np.round(held_out_fraction * n_ratings)
    return ordered[~is_test].reset_index(drop=True), ordered[is_test].reset_index(drop=True)


SPLITS = {
    "global cut (newest 20% of all ratings)": split_global,
    "per-user cut (each user's newest 20%)": split_per_user,
}


def date(timestamp):
    return pd.to_datetime(timestamp, unit="s").strftime("%Y-%m-%d")


if __name__ == "__main__":
    ratings = load_ratings()
    names = list(MODELS) + [MF_NAME]

    for split_name, split in SPLITS.items():
        train, test = split(ratings)
        actual = test["rating"].to_numpy()
        new_user = ~test["userId"].isin(train["userId"]).to_numpy()
        new_movie = ~test["movieId"].isin(train["movieId"]).to_numpy()
        both_known = ~new_user & ~new_movie

        print(f"\n=== {split_name} ===")
        print(f"train {len(train)} ratings, {date(train['timestamp'].min())} to {date(train['timestamp'].max())}")
        print(f"test  {len(test)} ratings, {date(test['timestamp'].min())} to {date(test['timestamp'].max())}")
        print(f"test ratings from users with no train rating: {new_user.sum()} ({new_user.mean():.1%})")
        print(f"test ratings of movies with no train rating:  {new_movie.sum()} ({new_movie.mean():.1%})")
        print(f"test ratings with a known user and movie:     {both_known.sum()} ({both_known.mean():.1%})")
        print(f"average rating, train / test: {train['rating'].mean():.3f} / {test['rating'].mean():.3f}")

        scores = {name: [] for name in names}
        known_scores = {name: [] for name in names}
        for name, predict_fn in MODELS.items():
            predictions = predict_fn(train, test)
            scores[name].append(rmse(predictions, actual))
            known_scores[name].append(rmse(predictions[both_known], actual[both_known]))

        for seed in range(N_SEEDS):
            model, _ = fit(
                train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION
            )
            predictions = predict(model, test)
            scores[MF_NAME].append(rmse(predictions, actual))
            known_scores[MF_NAME].append(rmse(predictions[both_known], actual[both_known]))
            print("finished seed", seed, flush=True)

        print(f"\n{'model':<36}{'rmse, all test':<26}{'rmse, known user and movie'}")
        for name in names:
            print(f"{name:<36}{summarise(scores[name]):<26}{summarise(known_scores[name])}")
