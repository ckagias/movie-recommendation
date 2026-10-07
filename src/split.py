from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.125


def load_ratings():
    return pd.read_csv(DATA / "ratings.csv")


def split_ratings(ratings, seed, held_out_fraction=TEST_FRACTION):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ratings))
    n_held_out = int(round(held_out_fraction * len(ratings)))
    held_out = ratings.iloc[order[:n_held_out]]
    kept = ratings.iloc[order[n_held_out:]]
    return kept.reset_index(drop=True), held_out.reset_index(drop=True)


def split_validation(train, seed):
    return split_ratings(train, seed, VALIDATION_FRACTION)


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = split_ratings(ratings, seed=0)

    print("all ratings:", len(ratings))
    print("train:", len(train), f"({len(train) / len(ratings):.1%})")
    print("test:", len(test), f"({len(test) / len(ratings):.1%})")

    print("\naverage rating, all / train / test:")
    print(round(ratings["rating"].mean(), 3), round(train["rating"].mean(), 3), round(test["rating"].mean(), 3))

    overlap = train.merge(test, on=["userId", "movieId"])
    print("\nuser-movie pairs in both piles (should be 0):", len(overlap))

    unseen_users = ~test["userId"].isin(train["userId"])
    unseen_movies = ~test["movieId"].isin(train["movieId"])
    print("test ratings from users with no train rating:", unseen_users.sum())
    print("test ratings of movies with no train rating:", unseen_movies.sum(), f"({unseen_movies.mean():.1%})")

    same = split_ratings(ratings, seed=0)[1]
    other = split_ratings(ratings, seed=1)[1]
    print("\nsame seed gives same test pile:", test.equals(same))
    print("different seed gives different test pile:", not test.equals(other))
