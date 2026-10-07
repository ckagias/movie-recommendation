import numpy as np

from split import load_ratings, split_ratings

N_SEEDS = 10


def rmse(predictions, actual):
    return np.sqrt(np.mean((predictions - actual) ** 2))


def summarise(values):
    return f"{np.mean(values):.3f} ({np.min(values):.3f} to {np.max(values):.3f})"


def predict_global_average(train, test):
    return np.full(len(test), train["rating"].mean())


if __name__ == "__main__":
    ratings = load_ratings()

    scores = []
    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        predictions = predict_global_average(train, test)
        scores.append(rmse(predictions, test["rating"].to_numpy()))

    print("baseline 1, guess the overall average")
    print("rmse over", N_SEEDS, "shuffles:", summarise(scores))
