import numpy as np

from baselines import MODELS, rmse, summarise
from mf import LEARNING_RATE, fit, predict
from split import load_ratings, split_ratings

N_FACTORS = 20
REGULARIZATION = 0.1
N_EPOCHS = 55
N_SEEDS = 10

MF_NAME = "matrix factorization"


if __name__ == "__main__":
    ratings = load_ratings()

    names = list(MODELS) + [MF_NAME]
    scores = {name: [] for name in names}
    unseen_scores = {name: [] for name in names}

    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        actual = test["rating"].to_numpy()
        unseen = ~test["movieId"].isin(train["movieId"]).to_numpy()

        all_predictions = {name: predict_fn(train, test) for name, predict_fn in MODELS.items()}
        model, _ = fit(
            train, N_FACTORS, LEARNING_RATE, N_EPOCHS, seed, regularization=REGULARIZATION
        )
        all_predictions[MF_NAME] = predict(model, test)

        for name, predictions in all_predictions.items():
            scores[name].append(rmse(predictions, actual))
            unseen_scores[name].append(rmse(predictions[unseen], actual[unseen]))
        print("finished seed", seed, flush=True)

    print(f"\n{'model':<36}{'rmse, all test':<26}{'rmse, unseen movies only'}")
    for name in names:
        print(f"{name:<36}{summarise(scores[name]):<26}{summarise(unseen_scores[name])}")

    best_baseline = list(MODELS)[-1]
    gains = np.array(scores[best_baseline]) - np.array(scores[MF_NAME])
    print(f"\nrmse gain of matrix factorization over {best_baseline}, per shuffle:")
    print(" ".join(f"{g:.3f}" for g in gains))
    print("shuffles where it wins:", int((gains > 0).sum()), "of", N_SEEDS)
    print("average gain:", summarise(gains))
