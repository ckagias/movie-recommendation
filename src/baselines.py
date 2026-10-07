import numpy as np

from split import load_ratings, split_ratings

N_SEEDS = 10


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


if __name__ == "__main__":
    ratings = load_ratings()

    scores = {"baseline 1, overall average": [], "baseline 2, movie average": []}
    unseen_scores = {"baseline 1, overall average": [], "baseline 2, movie average": []}
    unseen_share = []

    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        actual = test["rating"].to_numpy()
        unseen = ~test["movieId"].isin(train["movieId"]).to_numpy()
        unseen_share.append(unseen.mean())

        for name, predict in [
            ("baseline 1, overall average", predict_global_average),
            ("baseline 2, movie average", predict_movie_average),
        ]:
            predictions = predict(train, test)
            scores[name].append(rmse(predictions, actual))
            unseen_scores[name].append(rmse(predictions[unseen], actual[unseen]))

    print(f"{'model':<32}{'rmse, all test':<26}{'rmse, unseen movies only'}")
    for name in scores:
        print(f"{name:<32}{summarise(scores[name]):<26}{summarise(unseen_scores[name])}")
    print(f"\nshare of test ratings on movies with no train rating: {summarise(unseen_share)}")
