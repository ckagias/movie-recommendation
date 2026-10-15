import copy

import torch
from torch import nn

from baselines import MAX_RATING, MIN_RATING
from split import load_ratings, split_ratings, split_validation
from torch_data import SEED, UNKNOWN, make_datasets, make_loader

N_FACTORS = 50
LEARNING_RATE = 0.003
REGULARIZATION = 0.15
MAX_EPOCHS = 100
PATIENCE = 5
INIT_SCALE = 0.1


class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, mu):
        super().__init__()
        self.user_factors = nn.Embedding(n_users + 1, n_factors, padding_idx=UNKNOWN)
        self.movie_factors = nn.Embedding(n_movies + 1, n_factors, padding_idx=UNKNOWN)
        self.user_bias = nn.Embedding(n_users + 1, 1, padding_idx=UNKNOWN)
        self.movie_bias = nn.Embedding(n_movies + 1, 1, padding_idx=UNKNOWN)
        self.register_buffer("mu", torch.tensor(float(mu)))

        for factors in (self.user_factors, self.movie_factors):
            nn.init.normal_(factors.weight, 0.0, INIT_SCALE)
            with torch.no_grad():
                factors.weight[UNKNOWN].zero_()
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, users, movies):
        p, q = self.user_factors(users), self.movie_factors(movies)
        bu, bm = self.user_bias(users).squeeze(1), self.movie_bias(movies).squeeze(1)
        prediction = self.mu + bu + bm + (p * q).sum(dim=1)
        penalty = (p**2).sum(dim=1) + (q**2).sum(dim=1) + bu**2 + bm**2
        return prediction, penalty


def rmse(model, dataset):
    model.eval()
    with torch.no_grad():
        prediction, _ = model(dataset.users, dataset.movies)
        prediction = prediction.clamp(MIN_RATING, MAX_RATING)
        return torch.sqrt(((prediction - dataset.ratings) ** 2).mean()).item()


def train_model(model, train_set, validation_set, learning_rate, regularization, seed,
                max_epochs=MAX_EPOCHS, patience=PATIENCE, weight_decay=0.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loader = make_loader(train_set, shuffle=True, seed=seed)

    history = []
    best_score, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, max_epochs + 1):
        model.train()
        for users, movies, ratings in loader:
            prediction, penalty = model(users, movies)
            loss = 0.5 * ((ratings - prediction) ** 2 + regularization * penalty).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_score, validation_score = rmse(model, train_set), rmse(model, validation_set)
        history.append((epoch, train_score, validation_score))
        if validation_score < best_score:
            best_score, best_epoch, best_state = validation_score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def fit(train_set, validation_set, n_users, n_movies, n_factors, learning_rate, regularization, seed,
        max_epochs=MAX_EPOCHS, patience=PATIENCE):
    torch.manual_seed(seed)
    model = MatrixFactorization(n_users, n_movies, n_factors, train_set.ratings.mean().item())
    return train_model(model, train_set, validation_set, learning_rate, regularization, seed, max_epochs, patience)


if __name__ == "__main__":
    ratings = load_ratings()
    train_all, test = split_ratings(ratings, SEED)
    train, validation = split_validation(train_all, SEED)
    train_set, validation_set, test_set, user_index, movie_index = make_datasets(train, validation, test)

    model, history, best_epoch = fit(
        train_set, validation_set, len(user_index), len(movie_index),
        N_FACTORS, LEARNING_RATE, REGULARIZATION, SEED,
    )

    print(f"matrix factorization in PyTorch, {N_FACTORS} hidden numbers, regularization {REGULARIZATION}, Adam lr {LEARNING_RATE}, seed {SEED}")
    print(f"{'epoch':<8}{'train rmse':<14}{'validation rmse'}")
    for epoch, train_score, validation_score in history:
        print(f"{epoch:<8}{train_score:<14.3f}{validation_score:.3f}")
    print("best epoch:", best_epoch)
