import numpy as np

import mf
from baselines import summarise
from split import load_ratings, split_ratings, split_validation
from torch_data import make_datasets
from torch_mf import LEARNING_RATE, N_FACTORS, REGULARIZATION, fit, rmse

N_SHUFFLES = 10


if __name__ == "__main__":
    ratings = load_ratings()

    torch_scores, numpy_scores, best_epochs = [], [], []
    for seed in range(N_SHUFFLES):
        train_all, test = split_ratings(ratings, seed)
        train, validation = split_validation(train_all, seed)
        train_set, validation_set, test_set, user_index, movie_index = make_datasets(train, validation, test)

        model, _, best_epoch = fit(
            train_set, validation_set, len(user_index), len(movie_index),
            N_FACTORS, LEARNING_RATE, REGULARIZATION, seed,
        )
        torch_scores.append(rmse(model, test_set))
        best_epochs.append(best_epoch)

        _, history = mf.fit(train_all, 20, mf.LEARNING_RATE, 55, seed, test, regularization=0.1)
        numpy_scores.append(history[-1][2])
        print(f"shuffle {seed}: PyTorch {torch_scores[-1]:.3f}, NumPy {numpy_scores[-1]:.3f}", flush=True)

    difference = np.array(torch_scores) - np.array(numpy_scores)
    print(f"\nPyTorch MF ({N_FACTORS} hidden numbers, regularization {REGULARIZATION}, Adam lr {LEARNING_RATE}, early stopping):", summarise(torch_scores))
    print("NumPy MF (20 hidden numbers, regularization 0.1, 55 epochs, SGD):", summarise(numpy_scores))
    print(f"PyTorch minus NumPy: {difference.mean():+.3f} on average, PyTorch lower in {int((difference < 0).sum())} of {N_SHUFFLES} shuffles")
    print("best epochs:", best_epochs)
