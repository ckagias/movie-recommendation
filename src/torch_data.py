import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from split import load_ratings, split_ratings, split_validation

UNKNOWN = 0
SEED = 0
BATCH_SIZE = 1024


def index_ids(ids):
    return {id_: n for n, id_ in enumerate(np.unique(ids), start=1)}


class RatingsDataset(Dataset):
    def __init__(self, ratings, user_index, movie_index):
        self.users = torch.tensor(ratings["userId"].map(user_index).fillna(UNKNOWN).to_numpy(), dtype=torch.long)
        self.movies = torch.tensor(ratings["movieId"].map(movie_index).fillna(UNKNOWN).to_numpy(), dtype=torch.long)
        self.ratings = torch.tensor(ratings["rating"].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, n):
        return self.users[n], self.movies[n], self.ratings[n]


def make_datasets(train, validation, test):
    user_index = index_ids(train["userId"])
    movie_index = index_ids(train["movieId"])
    return (
        RatingsDataset(train, user_index, movie_index),
        RatingsDataset(validation, user_index, movie_index),
        RatingsDataset(test, user_index, movie_index),
        user_index,
        movie_index,
    )


def make_loader(dataset, shuffle, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


if __name__ == "__main__":
    ratings = load_ratings()
    train_all, test = split_ratings(ratings, SEED)
    train, validation = split_validation(train_all, SEED)
    train_set, validation_set, test_set, user_index, movie_index = make_datasets(train, validation, test)

    print(f"{'pile':<14}{'ratings':<10}{'unseen users':<15}{'unseen movies'}")
    for name, data in [("train", train_set), ("validation", validation_set), ("test", test_set)]:
        print(f"{name:<14}{len(data):<10}{int((data.users == UNKNOWN).sum()):<15}{int((data.movies == UNKNOWN).sum())}")

    print("\nusers / movies with an index (0 is kept for unknown):", len(user_index), "/", len(movie_index))
    print("largest user / movie index:", int(train_set.users.max()), "/", int(train_set.movies.max()))

    row = train.iloc[0]
    u, m, r = train_set[0]
    print("\nfirst train row:", int(row["userId"]), int(row["movieId"]), row["rating"])
    print("first dataset item:", int(u), int(m), float(r))
    id_of_user = {n: id_ for id_, n in user_index.items()}
    id_of_movie = {n: id_ for id_, n in movie_index.items()}
    print("indices map back to the same ids:", id_of_user[int(u)] == int(row["userId"]) and id_of_movie[int(m)] == int(row["movieId"]))

    loader = make_loader(train_set, shuffle=True)
    users, movies, stars = next(iter(loader))
    print("\none batch:")
    print("users  ", tuple(users.shape), users.dtype, "min", int(users.min()), "max", int(users.max()))
    print("movies ", tuple(movies.shape), movies.dtype, "min", int(movies.min()), "max", int(movies.max()))
    print("ratings", tuple(stars.shape), stars.dtype, "min", float(stars.min()), "max", float(stars.max()))

    n_batches = len(loader)
    print("\nbatches per epoch:", n_batches, "| last batch size:", len(train_set) - (n_batches - 1) * BATCH_SIZE)

    first_a = next(iter(make_loader(train_set, shuffle=True, seed=0)))[0]
    first_b = next(iter(make_loader(train_set, shuffle=True, seed=0)))[0]
    first_c = next(iter(make_loader(train_set, shuffle=True, seed=1)))[0]
    print("same seed gives the same first batch:", bool(torch.equal(first_a, first_b)))
    print("different seed gives a different first batch:", not torch.equal(first_a, first_c))

    seen = sum(len(b[2]) for b in make_loader(train_set, shuffle=True))
    print("ratings served in one epoch:", seen, "of", len(train_set))
