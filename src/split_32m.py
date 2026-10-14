from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "ml-32m"
CACHE = DATA / "ratings.npz"

TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15
SEED = 0

TRAIN, VALIDATION, TEST = 0, 1, 2
PILE_NAMES = ["train", "validation", "test"]


def load_ratings():
    if CACHE.exists():
        arrays = np.load(CACHE)
        return arrays["user"], arrays["movie"], arrays["rating"]
    table = pd.read_csv(
        DATA / "ratings.csv",
        usecols=["userId", "movieId", "rating"],
        dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
    )
    user = table["userId"].to_numpy()
    movie = table["movieId"].to_numpy()
    rating = table["rating"].to_numpy()
    np.savez(CACHE, user=user, movie=movie, rating=rating)
    return user, movie, rating


def assign_piles(user, seed=SEED):
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(user)), user))
    sorted_user = user[order]

    first_of_user = np.flatnonzero(np.r_[True, sorted_user[1:] != sorted_user[:-1]])
    counts = np.diff(np.r_[first_of_user, len(user)])
    rank = np.arange(len(user)) - np.repeat(first_of_user, counts)

    n_test = np.round(TEST_FRACTION * counts).astype(np.int64)
    n_validation = np.round(VALIDATION_FRACTION * counts).astype(np.int64)
    n_train = counts - n_test - n_validation

    pile_sorted = np.full(len(user), TEST, dtype=np.uint8)
    pile_sorted[rank < np.repeat(n_train + n_validation, counts)] = VALIDATION
    pile_sorted[rank < np.repeat(n_train, counts)] = TRAIN

    pile = np.empty(len(user), dtype=np.uint8)
    pile[order] = pile_sorted
    return pile


def split(seed=SEED):
    user, movie, rating = load_ratings()
    pile = assign_piles(user, seed)
    return user, movie, rating, pile


if __name__ == "__main__":
    user, movie, rating, pile = split()

    print("all ratings:", len(user))
    for code, name in enumerate(PILE_NAMES):
        share = (pile == code).mean()
        print(f"{name}: {(pile == code).sum()} ({share:.1%}), average rating {rating[pile == code].mean():.3f}")

    pair = user.astype(np.int64) * 2**20 + movie
    print("\nduplicate user-movie pairs in the raw file (0 means piles cannot overlap):", len(pair) - len(np.unique(pair)))

    per_user = pd.DataFrame({"user": user, "pile": pile}).groupby(["user", "pile"]).size().unstack(fill_value=0)
    per_user.columns = PILE_NAMES
    print("\nratings per user in each pile (min / median / max):")
    for name in PILE_NAMES:
        column = per_user[name]
        print(f"{name}: {column.min()} / {int(column.median())} / {column.max()}")
    print("users:", len(per_user), "| users with an empty pile:", int((per_user == 0).any(axis=1).sum()))

    train_movies = np.unique(movie[pile == TRAIN])
    print("\nmovies in the data:", len(np.unique(movie)), "| movies in train:", len(train_movies))
    for code in (VALIDATION, TEST):
        unseen = ~np.isin(movie[pile == code], train_movies)
        print(f"{PILE_NAMES[code]} ratings of movies with no train rating: {unseen.sum()} ({unseen.mean():.2%})")

    again = assign_piles(user, SEED)
    other = assign_piles(user, SEED + 1)
    print("\nsame seed gives same split:", np.array_equal(pile, again))
    print("different seed gives different split:", not np.array_equal(pile, other))
