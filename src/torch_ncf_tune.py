import argparse
import itertools

import numpy as np

from baselines import summarise
from split import load_ratings, split_ratings, split_validation
from torch_data import make_datasets
from torch_ncf import LEARNING_RATE, N_FACTORS, fit

HIDDEN_OPTIONS = [(64, 32), (128, 64)]
DROPOUT_OPTIONS = [0.2, 0.4, 0.6]
REGULARIZATION_OPTIONS = [0.15, 0.3]
WEIGHT_DECAY_OPTIONS = [0.0]
N_SEEDS = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropouts", type=float, nargs="+", default=DROPOUT_OPTIONS)
    parser.add_argument("--strengths", type=float, nargs="+", default=REGULARIZATION_OPTIONS)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=WEIGHT_DECAY_OPTIONS)
    parser.add_argument("--factors", type=int, default=N_FACTORS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    ratings = load_ratings()

    settings = list(itertools.product(HIDDEN_OPTIONS, args.dropouts, args.strengths, args.weight_decays))
    scores = {s: [] for s in settings}
    epochs = {s: [] for s in settings}

    for seed in range(N_SEEDS):
        train_all, test = split_ratings(ratings, seed)
        train, validation = split_validation(train_all, seed)
        train_set, validation_set, _, user_index, movie_index = make_datasets(train, validation, test)

        for hidden, dropout, reg, decay in settings:
            _, history, best_epoch = fit(
                train_set, validation_set, len(user_index), len(movie_index), args.factors, hidden, dropout,
                args.learning_rate, reg, seed, weight_decay=decay,
            )
            scores[(hidden, dropout, reg, decay)].append(history[best_epoch - 1][2])
            epochs[(hidden, dropout, reg, decay)].append(best_epoch)
        print("finished seed", seed, flush=True)

    print(f"\n{args.factors} hidden numbers per user and movie, Adam learning rate {args.learning_rate}")
    print(f"{'MLP':<12}{'dropout':<9}{'reg':<7}{'decay':<9}{'best validation rmse':<26}{'best epoch'}")
    for s in settings:
        hidden, dropout, reg, decay = s
        print(f"{str(hidden):<12}{dropout:<9}{reg:<7}{decay:<9}{summarise(scores[s]):<26}{np.mean(epochs[s]):.1f}")

    best = min(settings, key=lambda s: np.mean(scores[s]))
    print("\nbest on the validation pile:", best)
