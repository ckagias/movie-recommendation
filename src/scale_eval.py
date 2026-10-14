import numpy as np

from baselines_32m import MIN_TRAIN_RATINGS, TOP_N, held_out_liked, interval
from scale_check import train_matrix
from split_32m import PILE_NAMES, TEST, VALIDATION

N_SAMPLE_USERS = 50000
BATCH_SIZE = 2000
N_BOOTSTRAP = 1000
SEED = 0


def bootstrap_interval(values, rng):
    picks = rng.integers(0, len(values), (N_BOOTSTRAP, len(values)))
    return np.percentile(values[picks].mean(axis=1), [2.5, 97.5])


class HitRateTest:
    def __init__(self, user, movie, rating, pile, n_users=N_SAMPLE_USERS, seed=SEED):
        self.matrix, self.user_ids, self.movie_ids = train_matrix(user, movie, pile)
        counts = self.matrix.getnnz(axis=0)
        self.pool_cols = np.flatnonzero(counts >= MIN_TRAIN_RATINGS)
        position = np.full(len(self.movie_ids), -1)
        position[self.pool_cols] = np.arange(len(self.pool_cols))
        self.popularity = counts[self.pool_cols]

        rng = np.random.default_rng(seed)
        self.sample = np.sort(rng.choice(len(self.user_ids), n_users, replace=False))
        self.seen = self.matrix[self.sample][:, self.pool_cols].tocsr()

        self.liked = {}
        self.share_in_pool = {}
        for code in (VALIDATION, TEST):
            liked, share = held_out_liked(
                user, movie, rating, pile, code, self.user_ids, self.movie_ids, position
            )
            liked = liked[self.sample].tocsr()
            self.liked[code] = (liked, np.flatnonzero(liked.getnnz(axis=1) > 0))
            self.share_in_pool[code] = share

    def score(self, score_batch, code):
        liked, kept = self.liked[code]
        flags = np.zeros(len(kept))
        recalls = np.zeros(len(kept))
        pick_popularity = np.zeros(len(kept))
        for start in range(0, len(kept), BATCH_SIZE):
            part = kept[start : start + BATCH_SIZE]
            scores = np.array(score_batch(self.sample[part]), dtype=np.float32)
            scores[self.seen[part].toarray().astype(bool)] = -np.inf
            top = np.argpartition(-scores, TOP_N, axis=1)[:, :TOP_N]
            wanted = liked[part].toarray()
            hits = np.take_along_axis(wanted, top, axis=1).sum(axis=1)
            flags[start : start + len(part)] = hits > 0
            recalls[start : start + len(part)] = hits / wanted.sum(axis=1)
            pick_popularity[start : start + len(part)] = np.median(self.popularity[top], axis=1)
        return {"hit": flags, "recall": recalls, "pick popularity": pick_popularity}

    def hit_rate(self, score_batch, code):
        return self.score(score_batch, code)["hit"].mean()

    def popularity_scores(self, rows):
        return np.broadcast_to(self.popularity.astype(np.float32), (len(rows), len(self.pool_cols)))

    def describe(self):
        print(f"candidate pool: {len(self.pool_cols)} movies with at least {MIN_TRAIN_RATINGS} train ratings")
        for code in (VALIDATION, TEST):
            print(
                f"{PILE_NAMES[code]} pile: {len(self.liked[code][1])} of {len(self.sample)} sampled users "
                f"have a liked rating in the pool ({self.share_in_pool[code]:.1%} of liked ratings are in the pool)"
            )


def report(name, result, rng):
    low, high = bootstrap_interval(result["hit"], rng)
    print(
        f"{name:<40}{result['hit'].mean():<10.3f}{interval(low, high):<20}"
        f"{result['recall'].mean():<10.3f}{result['pick popularity'].mean():.0f}"
    )


def report_header():
    print(f"{'method':<40}{'hit rate':<10}{'95% interval':<20}{'recall':<10}{'median ratings of picks'}")
