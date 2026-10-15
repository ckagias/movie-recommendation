import torch
from torch import nn

from split import load_ratings, split_ratings, split_validation
from torch_data import SEED, make_datasets
from torch_mf import MatrixFactorization, rmse, train_model

N_FACTORS = 20
HIDDEN_SIZES = (128, 64)
DROPOUT = 0.2
LEARNING_RATE = 0.003
REGULARIZATION = 0.15
WEIGHT_DECAY = 0.003


class NeuralCF(MatrixFactorization):
    def __init__(self, n_users, n_movies, n_factors, mu, hidden_sizes, dropout):
        super().__init__(n_users, n_movies, n_factors, mu)
        layers, width = [], 2 * n_factors
        for size in hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, users, movies):
        p, q = self.user_factors(users), self.movie_factors(movies)
        bu, bm = self.user_bias(users).squeeze(1), self.movie_bias(movies).squeeze(1)
        interaction = self.mlp(torch.cat([p, q], dim=1)).squeeze(1)
        prediction = self.mu + bu + bm + interaction
        penalty = (p**2).sum(dim=1) + (q**2).sum(dim=1) + bu**2 + bm**2
        return prediction, penalty


def fit(train_set, validation_set, n_users, n_movies, n_factors, hidden_sizes, dropout, learning_rate,
        regularization, seed, weight_decay=0.0):
    torch.manual_seed(seed)
    model = NeuralCF(n_users, n_movies, n_factors, train_set.ratings.mean().item(), hidden_sizes, dropout)
    return train_model(model, train_set, validation_set, learning_rate, regularization, seed, weight_decay=weight_decay)


if __name__ == "__main__":
    ratings = load_ratings()
    train_all, test = split_ratings(ratings, SEED)
    train, validation = split_validation(train_all, SEED)
    train_set, validation_set, test_set, user_index, movie_index = make_datasets(train, validation, test)

    model, history, best_epoch = fit(
        train_set, validation_set, len(user_index), len(movie_index),
        N_FACTORS, HIDDEN_SIZES, DROPOUT, LEARNING_RATE, REGULARIZATION, SEED, WEIGHT_DECAY,
    )

    print(f"neural collaborative filtering, {N_FACTORS} hidden numbers, MLP {HIDDEN_SIZES}, dropout {DROPOUT}, regularization {REGULARIZATION}, weight decay {WEIGHT_DECAY}, Adam lr {LEARNING_RATE}, seed {SEED}")
    print(f"{'epoch':<8}{'train rmse':<14}{'validation rmse'}")
    for epoch, train_score, validation_score in history:
        print(f"{epoch:<8}{train_score:<14.3f}{validation_score:.3f}")
    print("best epoch:", best_epoch)
