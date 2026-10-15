import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from explain_content import LIKED_FROM
from library_hit_rate import hit_rate, prepare
from split import load_ratings, split_ratings, split_validation
from torch_data import UNKNOWN, index_ids

SEED = 0
BATCH_SIZE = 1024
INIT_SCALE = 0.1
N_FACTORS = 64
LEARNING_RATE = 0.01
REGULARIZATION = 0.01
N_NEGATIVES = 1
HIDDEN = 0
MAX_EPOCHS = 100
EVAL_EVERY = 5


class TwoTower(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, hidden):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, n_factors, padding_idx=UNKNOWN)
        self.movie_embedding = nn.Embedding(n_movies + 1, n_factors, padding_idx=UNKNOWN)
        self.movie_bias = nn.Embedding(n_movies + 1, 1, padding_idx=UNKNOWN)
        for embedding in (self.user_embedding, self.movie_embedding):
            nn.init.normal_(embedding.weight, 0.0, INIT_SCALE)
            with torch.no_grad():
                embedding.weight[UNKNOWN].zero_()
        nn.init.zeros_(self.movie_bias.weight)

        def tower():
            if not hidden:
                return nn.Identity()
            return nn.Sequential(nn.Linear(n_factors, hidden), nn.ReLU(), nn.Linear(hidden, n_factors))

        self.user_tower, self.movie_tower = tower(), tower()

    def user_vectors(self, users):
        return self.user_tower(self.user_embedding(users))

    def movie_vectors(self, movies):
        return self.movie_tower(self.movie_embedding(movies))

    def score(self, user_vectors, movies):
        return (user_vectors * self.movie_vectors(movies)).sum(dim=1) + self.movie_bias(movies).squeeze(1)


def positives(pile, user_index, movie_index, liked_only):
    if liked_only:
        pile = pile[pile["rating"] >= LIKED_FROM]
    users = torch.tensor(pile["userId"].map(user_index).to_numpy(), dtype=torch.long)
    movies = torch.tensor(pile["movieId"].map(movie_index).to_numpy(), dtype=torch.long)
    return users, movies


def scorer(model, user_index, movie_index, prepared):
    model.eval()
    with torch.no_grad():
        pool = torch.tensor([movie_index[m] for m in prepared["pool_ids"]], dtype=torch.long)
        pool_vectors = model.movie_vectors(pool).numpy()
        pool_bias = model.movie_bias(pool).squeeze(1).numpy()
        users = torch.arange(len(user_index) + 1)
        user_vectors = model.user_vectors(users).numpy()
    return lambda user_id: pool_vectors @ user_vectors[user_index[user_id]] + pool_bias


def train(pile, user_index, movie_index, liked_only, n_factors, hidden, learning_rate, regularization, n_epochs,
          seed, n_negatives=N_NEGATIVES, callback=None, eval_every=EVAL_EVERY):
    torch.manual_seed(seed)
    n_users, n_movies = len(user_index), len(movie_index)
    users, movies = positives(pile, user_index, movie_index, liked_only)
    is_positive = torch.zeros(n_users + 1, n_movies + 1, dtype=torch.bool)
    is_positive[users, movies] = True

    model = TwoTower(n_users, n_movies, n_factors, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(users), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            u, i = users[batch].repeat_interleave(n_negatives), movies[batch].repeat_interleave(n_negatives)
            j = torch.randint(1, n_movies + 1, (len(u),), generator=generator)
            valid = ~is_positive[u, j]

            user_raw, movie_raw, negative_raw = model.user_embedding(u), model.movie_embedding(i), model.movie_embedding(j)
            user_vectors = model.user_tower(user_raw)
            difference = model.score(user_vectors, i) - model.score(user_vectors, j)
            penalty = (
                (user_raw**2).sum(dim=1) + (movie_raw**2).sum(dim=1) + (negative_raw**2).sum(dim=1)
                + model.movie_bias(i).squeeze(1) ** 2 + model.movie_bias(j).squeeze(1) ** 2
            )
            loss = (valid * (F.softplus(-difference) + regularization * penalty)).sum() / valid.sum().clamp(min=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if callback is not None and (epoch % eval_every == 0 or epoch == n_epochs):
            callback(epoch, model)
    return model


if __name__ == "__main__":
    torch.set_num_threads(1)
    ratings = load_ratings()
    train_all, test = split_ratings(ratings, SEED)
    inner, validation = split_validation(train_all, SEED)
    prepared = prepare(inner, validation)
    user_index, movie_index = index_ids(inner["userId"]), index_ids(inner["movieId"])

    for liked_only in (False, True):
        for hidden in (0, 128):
            curve = []
            start = time.time()

            def record(epoch, model):
                curve.append((epoch, hit_rate(prepared, scorer(model, user_index, movie_index, prepared))[0]))

            train(inner, user_index, movie_index, liked_only, N_FACTORS, hidden, LEARNING_RATE, REGULARIZATION,
                  MAX_EPOCHS, SEED, callback=record)
            best = max(curve, key=lambda c: c[1])
            print(f"liked only {liked_only}, hidden {hidden}: {time.time() - start:.0f} s, best validation hit rate {best[1]:.3f} at epoch {best[0]}")
            print("   curve:", " ".join(f"{e}:{h:.3f}" for e, h in curve))
