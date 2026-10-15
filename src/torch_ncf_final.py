import numpy as np

import torch_mf
from baselines import summarise
from split import load_ratings, split_ratings, split_validation
from torch_data import make_datasets
from torch_ncf import DROPOUT, HIDDEN_SIZES, LEARNING_RATE, N_FACTORS, REGULARIZATION, WEIGHT_DECAY, fit

N_SHUFFLES = 10


if __name__ == "__main__":
    ratings = load_ratings()

    ncf_scores, mf_scores, best_epochs, train_scores = [], [], [], []
    for seed in range(N_SHUFFLES):
        train_all, test = split_ratings(ratings, seed)
        train, validation = split_validation(train_all, seed)
        train_set, validation_set, test_set, user_index, movie_index = make_datasets(train, validation, test)
        n_users, n_movies = len(user_index), len(movie_index)

        model, history, best_epoch = fit(
            train_set, validation_set, n_users, n_movies,
            N_FACTORS, HIDDEN_SIZES, DROPOUT, LEARNING_RATE, REGULARIZATION, seed, WEIGHT_DECAY,
        )
        ncf_scores.append(torch_mf.rmse(model, test_set))
        best_epochs.append(best_epoch)
        train_scores.append(history[best_epoch - 1][1])

        mf_model, _, _ = torch_mf.fit(
            train_set, validation_set, n_users, n_movies,
            torch_mf.N_FACTORS, torch_mf.LEARNING_RATE, torch_mf.REGULARIZATION, seed,
        )
        mf_scores.append(torch_mf.rmse(mf_model, test_set))
        print(f"shuffle {seed}: NCF {ncf_scores[-1]:.3f}, PyTorch MF {mf_scores[-1]:.3f}", flush=True)

    difference = np.array(ncf_scores) - np.array(mf_scores)
    print(f"\nNCF (MLP {HIDDEN_SIZES}, {N_FACTORS} hidden numbers, dropout {DROPOUT}, regularization {REGULARIZATION}, weight decay {WEIGHT_DECAY}):", summarise(ncf_scores))
    print("PyTorch MF (50 hidden numbers, regularization 0.15):", summarise(mf_scores))
    print(f"NCF minus MF: {difference.mean():+.3f} on average, NCF lower in {int((difference < 0).sum())} of {N_SHUFFLES} shuffles")
    print("NCF best epochs:", best_epochs)
    print(f"NCF train rmse at the best epoch: {np.mean(train_scores):.3f}")
