import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from baselines import summarise
from library_hit_rate import hit_rate, prepare
from split import load_ratings, split_ratings, split_validation
from torch_data import index_ids
from torch_twotower import EVAL_EVERY, N_FACTORS, scorer, train

N_WORKERS = 6
N_SEEDS = 3
MAX_EPOCHS = 100
GRID = []


def run(task):
    seed, liked_only, max_epochs = task
    torch.set_num_threads(1)
    train_all, test = split_ratings(load_ratings(), seed)
    inner, validation = split_validation(train_all, seed)
    prepared = prepare(inner, validation)
    user_index, movie_index = index_ids(inner["userId"]), index_ids(inner["movieId"])

    curves = []
    for params in GRID:
        curve = []

        def record(epoch, model):
            curve.append(hit_rate(prepared, scorer(model, user_index, movie_index, prepared))[0])

        train(inner, user_index, movie_index, liked_only, N_FACTORS, params["hidden"], params["learning_rate"],
              params["regularization"], max_epochs, seed, n_negatives=params["n_negatives"], callback=record)
        curves.append(curve)
    return seed, liked_only, curves


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.003])
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.01])
    parser.add_argument("--negatives", type=int, nargs="+", default=[5])
    parser.add_argument("--hidden", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    GRID.extend(
        {"hidden": h, "learning_rate": lr, "regularization": reg, "n_negatives": n}
        for h, lr, reg, n in itertools.product(args.hidden, args.learning_rates, args.strengths, args.negatives)
    )
    start = time.time()
    tasks = [(seed, liked_only, args.epochs) for seed, liked_only in itertools.product(range(N_SEEDS), [False, True])]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        results = list(pool.map(run, tasks))
    print(f"{time.time() - start:.0f} s")

    for liked_only in (False, True):
        print(f"\nliked only: {liked_only}   (hit rate on the validation pile, mean over {N_SEEDS} shuffles of the curve, checked every 5 epochs)")
        print(f"{'hidden':<8}{'lr':<8}{'reg':<7}{'negatives':<11}{'best epoch':<12}{'hit rate at best epoch':<26}{'at the last epoch'}")
        for g, params in enumerate(GRID):
            curves = np.array([curves[g] for seed, lo, curves in results if lo == liked_only])
            mean_curve = curves.mean(axis=0)
            best = int(mean_curve.argmax())
            print(f"{params['hidden']:<8}{params['learning_rate']:<8}{params['regularization']:<7}{params['n_negatives']:<11}{(best + 1) * EVAL_EVERY:<12}{summarise(curves[:, best]):<26}{mean_curve[-1]:.3f}")
