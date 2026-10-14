import json
import os
import threading
import time

import numpy as np

import als
from baselines_32m import interval
from library_32m import RESULTS, SEED, als_scorer
from scale_eval import HitRateTest, bootstrap_interval, report, report_header
from split_32m import TEST, VALIDATION, split

ITERATIONS = als.N_ITERATIONS
MEMORY_LIMIT_GB = 9.0
VARIANT = "ALS, rated at all"


def guard_memory(limit_gb):
    def watch():
        while True:
            with open("/proc/self/status") as status:
                kilobytes = int(next(line for line in status if line.startswith("VmRSS")).split()[1])
            if kilobytes / 1024**2 > limit_gb:
                print(f"stopping: resident memory passed {limit_gb} GB", flush=True)
                os._exit(1)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def hand_scorer(test, users, items):
    pool_items = np.ascontiguousarray(items[test.pool_cols].T)
    return lambda rows: users[rows] @ pool_items


if __name__ == "__main__":
    guard_memory(MEMORY_LIMIT_GB)
    user, movie, rating, pile = split()
    test = HitRateTest(user, movie, rating, pile)
    test.describe()
    print(flush=True)

    chosen = json.loads(RESULTS.read_text())[VARIANT]["chosen"]
    n_factors, regularization, confidence = chosen["factors"], chosen["regularization"], chosen["alpha"]
    print(f"\nsettings chosen for the library ALS on validation: {chosen}")

    user_items = test.matrix.astype(np.float64)
    item_users = user_items.T.tocsr()
    rng = np.random.default_rng(SEED)
    items = rng.normal(0, als.INIT_SCALE, (user_items.shape[1], n_factors))

    users = None
    previous = None
    for step in range(1, ITERATIONS + 1):
        start = time.time()
        users = als.solve_side(items, user_items, confidence, regularization)
        users_time = time.time() - start
        start = time.time()
        items = als.solve_side(users, item_users, confidence, regularization)
        items_time = time.time() - start
        current = als.loss(users, items, user_items, confidence, regularization)
        change = "" if previous is None else f", change {current - previous:+.4g}"
        print(
            f"iteration {step}: users {users_time:.0f} s, movies {items_time:.0f} s, loss {current:.6g}{change}",
            flush=True,
        )
        previous = current

    mine = hand_scorer(test, users, items)
    library = als_scorer(test, test.matrix, chosen)

    print("\nValidation pile, same settings, same users")
    report_header()
    rng = np.random.default_rng(SEED)
    validation = {"my ALS": test.score(mine, VALIDATION), "library ALS": test.score(library, VALIDATION)}
    for name, result in validation.items():
        report(name, result, rng)

    print("\nTest pile")
    report_header()
    final = {"my ALS": test.score(mine, TEST), "library ALS": test.score(library, TEST)}
    for name, result in final.items():
        report(name, result, rng)

    difference = final["my ALS"]["hit"] - final["library ALS"]["hit"]
    low, high = bootstrap_interval(difference, rng)
    print(f"\nmy ALS minus library ALS on test, hit rate: {difference.mean():+.4f}  {interval(low, high)}")
