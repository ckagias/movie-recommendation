import argparse

import numpy as np

from baselines import summarise
from split import load_ratings, split_ratings, split_validation
from torch_data import make_datasets
from torch_mf import LEARNING_RATE, fit

REGULARIZATION_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.8]
FACTOR_OPTIONS = [20, 50]
N_SEEDS = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strengths", type=float, nargs="+", default=REGULARIZATION_OPTIONS)
    parser.add_argument("--factors", type=int, nargs="+", default=FACTOR_OPTIONS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    ratings = load_ratings()

    settings = [(k, reg) for k in args.factors for reg in args.strengths]
    scores = {s: [] for s in settings}
    epochs = {s: [] for s in settings}

    for seed in range(N_SEEDS):
        train_all, test = split_ratings(ratings, seed)
        train, validation = split_validation(train_all, seed)
        train_set, validation_set, _, user_index, movie_index = make_datasets(train, validation, test)

        for k, reg in settings:
            model, history, best_epoch = fit(
                train_set, validation_set, len(user_index), len(movie_index), k, args.learning_rate, reg, seed
            )
            scores[(k, reg)].append(history[best_epoch - 1][2])
            epochs[(k, reg)].append(best_epoch)
        print("finished seed", seed, flush=True)

    print(f"\nAdam learning rate {args.learning_rate}")
    print(f"{'hidden':<8}{'reg':<7}{'best validation rmse':<26}{'best epoch'}")
    for k, reg in settings:
        s = (k, reg)
        print(f"{k:<8}{reg:<7}{summarise(scores[s]):<26}{np.mean(epochs[s]):.1f}")

    best = min(settings, key=lambda s: np.mean(scores[s]))
    print("\nbest on the validation pile: hidden numbers", best[0], "and regularization", best[1])
