import numpy as np
import pandas as pd
import pytest

from baselines import rmse
from split import DATA, load_ratings, split_ratings
from time_split import split_global, split_per_user

needs_data = pytest.mark.skipif(not (DATA / "ratings.csv").exists(), reason="data/ratings.csv is not downloaded")


def toy_ratings():
    rows = [(user, movie, 3.0, 100 * user + movie) for user in (1, 2, 3) for movie in range(1, 11)]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


def leaks_per_user(train, test):
    newest_train = train.groupby("userId")["timestamp"].max()
    oldest_test = test.groupby("userId")["timestamp"].min()
    return int((newest_train.reindex(oldest_test.index) > oldest_test).sum())


def test_per_user_split_toy():
    train, test = split_per_user(toy_ratings())
    assert len(train) == 24 and len(test) == 6
    assert leaks_per_user(train, test) == 0
    assert sorted(test.loc[test["userId"] == 2, "movieId"]) == [9, 10]


def test_global_split_toy():
    train, test = split_global(toy_ratings())
    assert len(train) == 24 and len(test) == 6
    assert train["timestamp"].max() <= test["timestamp"].min()


def test_leak_check_flags_a_random_split():
    train, test = split_ratings(toy_ratings(), seed=0, held_out_fraction=0.5)
    assert leaks_per_user(train, test) > 0


@needs_data
def test_time_splits_on_the_real_data():
    ratings = load_ratings()
    train, test = split_per_user(ratings)
    assert leaks_per_user(train, test) == 0
    assert len(train) + len(test) == len(ratings)
    assert not train.merge(test, on=["userId", "movieId"]).shape[0]

    train, test = split_global(ratings)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(ratings)


def test_rmse_by_hand():
    predictions = np.array([3.0, 4.0, 2.0])
    actual = np.array([5.0, 4.0, 1.0])
    assert rmse(predictions, actual) == pytest.approx(np.sqrt((4 + 0 + 1) / 3))
