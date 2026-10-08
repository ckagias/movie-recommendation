import numpy as np
from scipy.sparse import csr_matrix

INIT_SCALE = 0.01


def interaction_matrix(frame, user_index, movie_index, liked_from=None):
    if liked_from is not None:
        frame = frame[frame["rating"] >= liked_from]
    rows = frame["userId"].map(user_index).to_numpy()
    cols = frame["movieId"].map(movie_index).to_numpy()
    shape = (len(user_index), len(movie_index))
    return csr_matrix((np.ones(len(frame)), (rows, cols)), shape=shape)


def solve_side(fixed, interactions, confidence, regularization):
    n_factors = fixed.shape[1]
    base = fixed.T @ fixed + regularization * np.eye(n_factors)
    solved = np.zeros((interactions.shape[0], n_factors))
    for row in range(interactions.shape[0]):
        columns = interactions.indices[interactions.indptr[row] : interactions.indptr[row + 1]]
        if len(columns) == 0:
            continue
        rated = fixed[columns]
        solved[row] = np.linalg.solve(base + (confidence - 1) * rated.T @ rated, confidence * rated.sum(axis=0))
    return solved


def loss(users, items, interactions, confidence, regularization):
    everything = np.sum((users.T @ users) * (items.T @ items))
    rows, columns = interactions.nonzero()
    scores = np.sum(users[rows] * items[columns], axis=1)
    rated_correction = np.sum(confidence * (1 - scores) ** 2 - scores**2)
    penalty = regularization * (np.sum(users**2) + np.sum(items**2))
    return everything + rated_correction + penalty


def fit(user_items, n_factors, regularization, confidence, n_iterations, seed, track_loss=False):
    rng = np.random.default_rng(seed)
    item_users = user_items.T.tocsr()
    items = rng.normal(0, INIT_SCALE, (user_items.shape[1], n_factors))
    history = []
    for _ in range(n_iterations):
        users = solve_side(items, user_items, confidence, regularization)
        items = solve_side(users, item_users, confidence, regularization)
        if track_loss:
            history.append(loss(users, items, user_items, confidence, regularization))
    return users, items, history
