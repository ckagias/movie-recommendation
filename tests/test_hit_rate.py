import numpy as np
import pandas as pd
import pytest

from split import DATA

pytestmark = pytest.mark.skipif(not (DATA / "ratings.csv").exists(), reason="data/ratings.csv is not downloaded")

N_POOL = 14
POOL = list(range(101, 101 + N_POOL))


def frame(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"])


@pytest.fixture
def toy(monkeypatch):
    import library_hit_rate

    monkeypatch.setattr(library_hit_rate, "MIN_TRAIN_RATINGS", 2)
    fit_pile = frame(
        [(3, m, 4.0, 0) for m in POOL] + [(4, m, 4.0, 0) for m in POOL] + [(1, 101, 5.0, 0), (1, 102, 5.0, 0), (2, 103, 5.0, 0)]
    )
    held_out = frame(
        [(1, 112, 4.5, 0), (1, 114, 4.0, 0), (1, 104, 3.0, 0), (1, 999, 5.0, 0), (2, 101, 4.0, 0)]
    )
    return library_hit_rate, library_hit_rate.prepare(fit_pile, held_out)


def test_prepare(toy):
    _, prepared = toy
    assert list(prepared["pool_ids"]) == POOL
    assert list(prepared["popularity"]) == [3, 3, 3] + [2] * 11
    assert {u: sorted(w) for u, w in prepared["relevant"].items()} == {1: [11, 13], 2: [0]}
    assert list(np.flatnonzero(prepared["seen"][1])) == [0, 1]
    assert list(np.flatnonzero(prepared["seen"][2])) == [2]


def test_hit_rate_and_recall_by_hand(toy):
    module, prepared = toy
    positions = np.arange(N_POOL, dtype=float)
    scores = {1: -positions, 2: positions}
    hit, recall, _ = module.hit_rate(prepared, lambda user_id: scores[user_id])
    assert hit == pytest.approx(0.5)
    assert recall == pytest.approx(0.25)


def test_top_n_breaks_ties_by_popularity():
    from hit_rate import TOP_N, top_n

    scores = np.zeros(15)
    popularity = np.arange(15)
    seen = np.zeros(15, dtype=bool)
    seen[14] = True
    assert list(top_n(scores, seen, popularity)) == list(range(13, 13 - TOP_N, -1))
