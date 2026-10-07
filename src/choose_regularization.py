import argparse

import numpy as np

from baselines import summarise
from mf import LEARNING_RATE, fit
from split import load_ratings, split_ratings, split_validation

REGULARIZATION_OPTIONS = [0.0, 0.01, 0.02, 0.05, 0.1]
FACTOR_OPTIONS = [5, 20, 50]
N_EPOCHS = 40
N_SEEDS = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", type=float, nargs="+", default=REGULARIZATION_OPTIONS)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    args = parser.parse_args()

    ratings = load_ratings()

    settings = [(k, reg) for k in FACTOR_OPTIONS for reg in args.strengths]
    best_scores = {s: [] for s in settings}
    best_epochs = {s: [] for s in settings}
    last_train = {s: [] for s in settings}
    last_validation = {s: [] for s in settings}

    for seed in range(N_SEEDS):
        train, test = split_ratings(ratings, seed)
        fit_pile, validation = split_validation(train, seed)

        for k, reg in settings:
            _, history = fit(
                fit_pile, k, LEARNING_RATE, args.epochs, seed, validation, regularization=reg
            )
            validation_scores = np.array([h[2] for h in history])
            best_scores[(k, reg)].append(validation_scores.min())
            best_epochs[(k, reg)].append(validation_scores.argmin() + 1)
            last_train[(k, reg)].append(history[-1][1])
            last_validation[(k, reg)].append(history[-1][2])
        print("finished seed", seed, flush=True)

    print(
        f"\n{'hidden':<8}{'reg':<7}{'best validation rmse':<26}{'best epoch':<12}"
        f"{'train rmse at end':<20}{'validation rmse at end'}"
    )
    for k, reg in settings:
        s = (k, reg)
        print(
            f"{k:<8}{reg:<7}{summarise(best_scores[s]):<26}{np.mean(best_epochs[s]):<12.1f}"
            f"{np.mean(last_train[s]):<20.3f}{np.mean(last_validation[s]):.3f}"
        )

    best = min(settings, key=lambda s: np.mean(best_scores[s]))
    print("\nbest on the validation pile: hidden numbers", best[0], "and regularization", best[1])
