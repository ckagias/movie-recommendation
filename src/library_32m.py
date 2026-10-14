import os

for name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(name, "1")

import json
import time

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from implicit.cpu.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender
from scipy.sparse import csr_matrix

from baselines_32m import LIKED_FROM, interval
from scale_eval import HitRateTest, bootstrap_interval, report, report_header
from split_32m import DATA, TEST, TRAIN, VALIDATION, split

ALS_ITERATIONS = 15
SEED = 0
RESULTS = DATA / "library_results.json"

ALS_FIRST = [
    {"factors": 32, "regularization": r, "alpha": a} for r in [0.01, 0.1, 1.0, 10.0] for a in [1, 10, 40]
]
BPR_FIRST = [
    {"factors": 64, "learning_rate": lr, "regularization": r, "iterations": 100}
    for lr in [0.01, 0.05]
    for r in [0.001, 0.01, 0.1]
]
KNN_ALL = [{"K": k} for k in [20, 50, 100, 200, 400]]

MAX_EXTENSIONS = 3
# setting: (step factor, may also go down, lowest allowed value when going down)
ALS_STEPS = {"regularization": (3.0, True, 0.001), "alpha": (2.5, False, 1), "factors": (2, False, 8)}
BPR_STEPS = {
    "regularization": (3.0, True, 0.0001),
    "learning_rate": (3.0, True, 0.0001),
    "factors": (2, False, 8),
    "iterations": (2, False, 50),
}
KNN_STEPS = {"K": (2, True, 5)}


def als_second(best):
    return [{**best, "factors": f} for f in [16, 64, 128]]


def bpr_second(best):
    return [{**best, "factors": f} for f in [32, 128]] + [{**best, "iterations": n} for n in [50, 200, 400]]


def als_scorer(test, matrix, params):
    model = AlternatingLeastSquares(iterations=ALS_ITERATIONS, random_state=SEED, **params)
    model.fit(matrix, show_progress=False)
    items = np.ascontiguousarray(np.asarray(model.item_factors)[test.pool_cols].T)
    users = np.asarray(model.user_factors)
    return lambda rows: users[rows] @ items


def bpr_scorer(test, matrix, params):
    model = BayesianPersonalizedRanking(random_state=SEED, **params)
    model.fit(matrix, show_progress=False)
    items = np.ascontiguousarray(np.asarray(model.item_factors)[test.pool_cols].T)
    users = np.asarray(model.user_factors)
    return lambda rows: users[rows] @ items


def knn_scorer(test, matrix, params):
    model = CosineRecommender(**params)
    model.fit(matrix, show_progress=False)
    similarity = model.similarity[:, test.pool_cols].tocsr()
    return lambda rows: (matrix[rows] @ similarity).toarray()


def liked_matrix(test, user, movie, rating, pile):
    keep = (pile == TRAIN) & (rating >= LIKED_FROM)
    rows = np.searchsorted(test.user_ids, user[keep])
    cols = np.searchsorted(test.movie_ids, movie[keep])
    ones = np.ones(keep.sum(), dtype=np.float32)
    return csr_matrix((ones, (rows, cols)), shape=test.matrix.shape)


def validate(test, matrix, make_scorer, grid, label):
    rows = []
    for params in grid:
        start = time.time()
        hit = test.hit_rate(make_scorer(test, matrix, params), VALIDATION)
        print(f"{label} {params}: validation hit rate {hit:.3f} ({time.time() - start:.0f} s)", flush=True)
        rows.append({"params": params, "validation hit rate": hit})
    return rows


def best_of(rows):
    return max(rows, key=lambda row: row["validation hit rate"])


def edge_note(rows, key, best):
    values = sorted({row["params"][key] for row in rows})
    if best["params"][key] in (values[0], values[-1]) and len(values) > 1:
        print(f"  note: best {key} = {best['params'][key]} is at the edge of the grid {values}", flush=True)


def extend_edges(test, matrix, make_scorer, rows, best, steps, label):
    for key, (factor, can_go_down, floor) in steps.items():
        for _ in range(MAX_EXTENSIONS):
            values = sorted({row["params"][key] for row in rows})
            value = best["params"][key]
            if value == values[-1]:
                new = type(value)(value * factor)
            elif value == values[0] and can_go_down and value / factor >= floor:
                new = type(value)(value / factor)
            else:
                break
            print(f"  best {key} = {value} is at the edge, trying {new}", flush=True)
            added = validate(test, matrix, make_scorer, [{**best["params"], key: new}], label)
            rows += added
            if best_of(rows) is best:
                break
            best = best_of(rows)
    return best


def tune(test, matrix, make_scorer, first, second_for, steps, label, saved_rows=None):
    if saved_rows:
        rows = saved_rows
        best = best_of(rows)
        print(f"{label}: continuing from {len(rows)} saved validation runs", flush=True)
    else:
        rows = validate(test, matrix, make_scorer, first, label)
        best = best_of(rows)
        if second_for is not None:
            rows += validate(test, matrix, make_scorer, second_for(best["params"]), label)
            best = best_of(rows)
    best = extend_edges(test, matrix, make_scorer, rows, best, steps, label)
    for key in steps:
        edge_note(rows, key, best)
    print(f"{label}: chosen {best['params']} with validation hit rate {best['validation hit rate']:.3f}\n", flush=True)
    return best, rows


if __name__ == "__main__":
    user, movie, rating, pile = split()
    test = HitRateTest(user, movie, rating, pile)
    test.describe()
    rng = np.random.default_rng(SEED)

    variants = {"rated at all": test.matrix, "liked only": liked_matrix(test, user, movie, rating, pile)}
    models = {
        "ALS": (als_scorer, ALS_FIRST, als_second, ALS_STEPS),
        "BPR": (bpr_scorer, BPR_FIRST, bpr_second, BPR_STEPS),
        "item-item cosine": (knn_scorer, KNN_ALL, None, KNN_STEPS),
    }

    baseline = test.score(test.popularity_scores, TEST)
    saved = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results = {"most rated": {"test hit rate": float(baseline["hit"].mean())}}
    final = {}
    for model_name, (make_scorer, first, second_for, keys) in models.items():
        for variant, matrix in variants.items():
            label = f"{model_name}, {variant}"
            saved_rows = saved.get(label, {}).get("all validation runs")
            best, rows = tune(test, matrix, make_scorer, first, second_for, keys, label, saved_rows)
            start = time.time()
            final[label] = test.score(make_scorer(test, matrix, best["params"]), TEST)
            print(f"{label}: scored once on the test pile ({time.time() - start:.0f} s)", flush=True)
            results[label] = {
                "chosen": best["params"],
                "validation hit rate": best["validation hit rate"],
                "test hit rate": float(final[label]["hit"].mean()),
                "all validation runs": rows,
            }
            RESULTS.write_text(json.dumps(results, indent=1))

    print(f"\nTest pile, {len(baseline['hit'])} users, settings chosen on the validation pile")
    report_header()
    report("most rated", baseline, rng)
    for label, result in final.items():
        report(label, result, rng)

    print("\nHit rate minus most rated, same users (95% interval over users)")
    for label, result in final.items():
        difference = result["hit"] - baseline["hit"]
        low, high = bootstrap_interval(difference, rng)
        print(f"{label:<40}{difference.mean():+.3f}  {interval(low, high)}")
