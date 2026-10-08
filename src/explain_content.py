import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movie_text import DATA, build_movie_texts

LIKED_FROM = 4.0
JUNK_TAGS = {"in netflix queue"}

movies = build_movie_texts()
ratings = pd.read_csv(DATA / "ratings.csv")

n_ratings = ratings.groupby("movieId").size()
popularity = movies["movieId"].map(n_ratings).fillna(0).astype(int).to_numpy()
row_of = {movie_id: row for row, movie_id in enumerate(movies["movieId"])}

X = TfidfVectorizer().fit_transform(movies["genres_text"] + " " + movies["decade_text"])


def build_tag_sets():
    tags = pd.read_csv(DATA / "tags.csv")
    tags["clean"] = (
        tags["tag"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.split()
        .str.join(" ")
    )
    tags = tags[(tags["clean"] != "") & ~tags["clean"].isin(JUNK_TAGS)]
    tags = tags.drop_duplicates(["movieId", "clean"])
    movies_per_tag = tags.groupby("clean")["movieId"].nunique()
    tags = tags[tags["clean"].map(movies_per_tag) >= 2]
    return tags.groupby("movieId")["clean"].agg(set)


tag_sets = movies["movieId"].map(build_tag_sets()).map(lambda s: s if isinstance(s, set) else set())
genre_sets = movies["genres"].map(
    lambda g: set() if g == "(no genres listed)" else set(g.split("|"))
)
decades = movies["decade_text"].str.replace("decade", "", regex=False)


def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def shared_features(i, j):
    decade = decades[i] if decades[i] != "" and decades[i] == decades[j] else ""
    return {
        "genres": genre_sets[i] & genre_sets[j],
        "decade": decade,
        "tags": tag_sets[i] & tag_sets[j],
    }


def explain(liked_row, recommended_row, liked_rating):
    shared = shared_features(liked_row, recommended_row)
    liked_title = movies.loc[liked_row, "title"]
    parts = []
    if shared["genres"]:
        label = "genre" if len(shared["genres"]) == 1 else "genres"
        parts.append(f"{label} {join_words(sorted(shared['genres']))}")
    if shared["decade"]:
        parts.append(f"the {shared['decade']} decade")
    if shared["tags"]:
        label = "tag" if len(shared["tags"]) == 1 else "tags"
        parts.append(f"{label} {join_words(sorted(shared['tags']))}")
    if not parts:
        return f"Closest to {liked_title}, which you rated {liked_rating}, but they share no listed feature."
    return f"Shares {join_words(parts)} with {liked_title}, which you rated {liked_rating}."


def recommend(user_id, n=10):
    mine = ratings[ratings["userId"] == user_id]
    liked = mine[mine["rating"] >= LIKED_FROM].sort_values("rating", ascending=False)
    liked_rows = liked["movieId"].map(row_of).to_numpy()

    similarity = cosine_similarity(X[liked_rows], X).round(6)
    closest = similarity.argmax(axis=0)
    score = similarity.max(axis=0)
    score[mine["movieId"].map(row_of).to_numpy()] = -1

    order = np.lexsort((-popularity, -score))[:n]
    rows = []
    for j in order:
        liked_pos = closest[j]
        rows.append(
            {
                "title": movies.loc[j, "title"],
                "score": score[j],
                "why": explain(liked_rows[liked_pos], j, liked["rating"].iloc[liked_pos]),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("movies with at least one cleaned tag:", sum(len(s) > 0 for s in tag_sets))
    for user_id in [1, 3, 414]:
        mine = ratings[ratings["userId"] == user_id]
        print(f"\n=== user {user_id}: {len(mine)} ratings, {(mine['rating'] >= LIKED_FROM).sum()} liked ===")
        for _, r in recommend(user_id, 5).iterrows():
            print(f"{r['title']}  (score {r['score']:.2f})\n    {r['why']}")
