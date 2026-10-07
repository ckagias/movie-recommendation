import numpy as np

from baselines import summarise
from mf import LEARNING_RATE, N_EPOCHS, fit
from split import load_ratings, split_ratings, split_validation

FACTOR_OPTIONS = [5, 10, 20, 50]
N_SEEDS = 5


if __name__ == "__main__":
    ratings = load_ratings()

    best_scores = {k: [] for k in FACTOR_OPTIONS}
    best_epochs = {k: [] for k in FACTOR_OPTIONS}

    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        fit_pile, validation = split_validation(train, seed)

        for k in FACTOR_OPTIONS:
            _, history = fit(fit_pile, k, LEARNING_RATE, N_EPOCHS, seed, validation)
            validation_scores = np.array([h[2] for h in history])
            best_scores[k].append(validation_scores.min())
            best_epochs[k].append(validation_scores.argmin() + 1)
        print("finished seed", seed, flush=True)

    print(f"\n{'hidden numbers':<18}{'best validation rmse':<26}{'best epoch (min to max)'}")
    for k in FACTOR_OPTIONS:
        print(
            f"{k:<18}{summarise(best_scores[k]):<26}"
            f"{np.mean(best_epochs[k]):.1f} ({min(best_epochs[k])} to {max(best_epochs[k])})"
        )

    best = min(FACTOR_OPTIONS, key=lambda k: np.mean(best_scores[k]))
    print("\nbest on the validation pile:", best, "hidden numbers")
