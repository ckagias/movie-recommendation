import numpy as np

from baselines import MAX_RATING, MIN_RATING, rmse
from split import load_ratings, split_ratings

N_FACTORS = 10
LEARNING_RATE = 0.01
N_EPOCHS = 30
INIT_SCALE = 0.1
SEED = 0


def index_ids(ids):
    return {id_: n for n, id_ in enumerate(np.unique(ids))}


def predict(model, data):
    u = data["userId"].map(model["user_index"]).to_numpy()
    m = data["movieId"].map(model["movie_index"]).fillna(-1).astype(int).to_numpy()
    known = m >= 0
    out = model["mu"] + model["user_bias"][u]
    out[known] += model["movie_bias"][m[known]] + np.sum(
        model["P"][u[known]] * model["Q"][m[known]], axis=1
    )
    return np.clip(out, MIN_RATING, MAX_RATING)


def fit(train, n_factors, learning_rate, n_epochs, seed, test=None):
    rng = np.random.default_rng(seed)

    user_index = index_ids(train["userId"])
    movie_index = index_ids(train["movieId"])
    users = train["userId"].map(user_index).to_numpy()
    movies = train["movieId"].map(movie_index).to_numpy()
    ratings = train["rating"].to_numpy()

    model = {
        "mu": ratings.mean(),
        "user_index": user_index,
        "movie_index": movie_index,
        "user_bias": np.zeros(len(user_index)),
        "movie_bias": np.zeros(len(movie_index)),
        "P": rng.normal(0, INIT_SCALE, (len(user_index), n_factors)),
        "Q": rng.normal(0, INIT_SCALE, (len(movie_index), n_factors)),
    }
    mu = model["mu"]
    user_bias, movie_bias, P, Q = (
        model["user_bias"],
        model["movie_bias"],
        model["P"],
        model["Q"],
    )

    history = []
    for epoch in range(1, n_epochs + 1):
        for n in rng.permutation(len(ratings)):
            u, m = users[n], movies[n]
            error = ratings[n] - (mu + user_bias[u] + movie_bias[m] + P[u] @ Q[m])
            user_bias[u] += learning_rate * error
            movie_bias[m] += learning_rate * error
            p_old = P[u].copy()
            P[u] += learning_rate * error * Q[m]
            Q[m] += learning_rate * error * p_old

        train_score = rmse(predict(model, train), ratings)
        test_score = rmse(predict(model, test), test["rating"].to_numpy()) if test is not None else None
        history.append((epoch, train_score, test_score))

    return model, history


if __name__ == "__main__":
    ratings = load_ratings()
    train, test = split_ratings(ratings, SEED)

    model, history = fit(train, N_FACTORS, LEARNING_RATE, N_EPOCHS, SEED, test)

    print(f"matrix factorization, {N_FACTORS} hidden numbers, no regularization, seed {SEED}")
    print(f"{'epoch':<8}{'train rmse':<14}{'test rmse'}")
    for epoch, train_score, test_score in history:
        print(f"{epoch:<8}{train_score:<14.3f}{test_score:.3f}")
