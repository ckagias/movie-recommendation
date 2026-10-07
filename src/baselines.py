import numpy as np

from split import load_ratings, split_ratings

N_SEEDS = 10
MIN_RATING = 0.5
MAX_RATING = 5.0


def rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))


def summarise(values):
    return f"{np.mean(values):.3f} ({np.min(values):.3f} to {np.max(values):.3f})"


def predict_global_average(train, test):
    return np.full(len(test), train["rating"].mean())


def predict_movie_average(train, test):
    movie_means = train.groupby("movieId")["rating"].mean()
    fallback = train["rating"].mean()
    return test["movieId"].map(movie_means).fillna(fallback).to_numpy()


def predict_movie_average_plus_user_bias(train, test):
    movie_means = train.groupby("movieId")["rating"].mean()
    fallback = train["rating"].mean()
    train_expected = train["movieId"].map(movie_means)
    user_bias = (train["rating"] - train_expected).groupby(train["userId"]).mean()
    expected = test["movieId"].map(movie_means).fillna(fallback)
    predictions = expected + test["userId"].map(user_bias).fillna(0)
    return predictions.clip(MIN_RATING, MAX_RATING).to_numpy()


MODELS = {
    "baseline 1, overall average": predict_global_average,
    "baseline 2, movie average": predict_movie_average,
    "baseline 3, movie avg + user bias": predict_movie_average_plus_user_bias,
}


if __name__ == "__main__":
    ratings = load_ratings()

    scores = {name: [] for name in MODELS}
    unseen_scores = {name: [] for name in MODELS}
    unseen_share = []

    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        actual = test["rating"].to_numpy()
        unseen = ~test["movieId"].isin(train["movieId"]).to_numpy()
        unseen_share.append(unseen.mean())

        for name, predict in MODELS.items():
            predictions = predict(train, test)
            scores[name].append(rmse(predictions, actual))
            unseen_scores[name].append(rmse(predictions[unseen], actual[unseen]))

    print(f"{'model':<36}{'rmse, all test':<26}{'rmse, unseen movies only'}")
    for name in scores:
        print(f"{name:<36}{summarise(scores[name]):<26}{summarise(unseen_scores[name])}")
    print(f"\nshare of test ratings on movies with no train rating: {summarise(unseen_share)}")
