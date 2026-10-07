from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from movie_text import build_movie_texts

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

K_VALUES = [5, 10, 15, 20]
N_SEEDS = 5

movies = build_movie_texts()
ratings = pd.read_csv(ROOT / "data" / "ratings.csv")
movies["text_for_clusters"] = movies["genres_text"] + " " + movies["decade_text"]
movies = movies[movies["text_for_clusters"].str.strip() != ""].reset_index(drop=True)

movies["mean_rating"] = movies["movieId"].map(ratings.groupby("movieId")["rating"].mean())
movies["n_ratings"] = movies["movieId"].map(ratings.groupby("movieId").size()).fillna(0)
movies["first_genre"] = movies["genres"].str.split("|").str[0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(movies["text_for_clusters"]).toarray()
words = vectorizer.get_feature_names_out()
genre_words = set(movies["genres_text"].str.split().explode())

print("movies clustered:", len(movies))
print("distinct feature vectors:", len({tuple(row) for row in X.round(6)}))
print("words in the vocabulary:", len(words))

pca = PCA(n_components=2, random_state=0)
coords = pca.fit_transform(X)
print("share of variation kept by the 2D map: {:.1%}".format(pca.explained_variance_ratio_.sum()))


def top_tag_words(cluster_rows, all_rows, n=3, min_movies=5):
    def tagged_sets(df):
        return df["tags_text"].str.split().map(set)

    in_cluster = tagged_sets(cluster_rows)
    overall = tagged_sets(all_rows)
    cluster_count = pd.Series([w for s in in_cluster for w in s]).value_counts()
    overall_count = pd.Series([w for s in overall for w in s]).value_counts()
    lift = {}
    for w, c in cluster_count.items():
        if c >= min_movies and len(w) > 2:
            lift[w] = (c / len(cluster_rows)) / (overall_count[w] / len(all_rows))
    return sorted(lift, key=lift.get, reverse=True)[:n]


summary = []
labels_by_k = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
rng = np.random.default_rng(0)
jitter = rng.normal(0, 0.012, size=coords.shape)

for ax, k in zip(axes.ravel(), K_VALUES):
    runs = [
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X) for seed in range(N_SEEDS)
    ]
    agreement = np.mean([adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(runs, 2)])
    km = runs[0]
    labels = km.labels_
    labels_by_k[k] = labels
    sil = silhouette_score(X, labels, sample_size=3000, random_state=0)
    sizes = np.bincount(labels, minlength=k)
    top_word_share = []
    for c in range(k):
        top_word = words[np.argmax(km.cluster_centers_[c])]
        has_word = movies["text_for_clusters"].str.split().map(lambda t: top_word in t)
        top_word_share.append(has_word[labels == c].mean())
    summary.append(
        {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "agreement between seeds (ARI)": agreement,
            "smallest cluster": sizes.min(),
            "largest cluster": sizes.max(),
            "movies that carry their cluster's top word": float(np.average(top_word_share, weights=sizes)),
            "match with first-listed genre (ARI)": adjusted_rand_score(movies["first_genre"], labels),
        }
    )
    ax.scatter(
        coords[:, 0] + jitter[:, 0], coords[:, 1] + jitter[:, 1],
        c=labels, cmap="tab20", s=4, alpha=0.5, linewidths=0,
    )
    ax.set_title(f"k = {k}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")

fig.suptitle("Movies on a 2D map, colored by K-Means cluster (points jittered slightly)")
fig.tight_layout()
fig.savefig(PLOTS / "pca_map.png", dpi=150, bbox_inches="tight")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
table = pd.DataFrame(summary).set_index("k")
print("\n--- clustering metrics ---")
print(table.round(3).T.to_string())

for k in K_VALUES:
    labels = labels_by_k[k]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(f"\n--- clusters for k = {k} ---")
    print(f"{'cluster':<8}{'movies':>7}{'avg stars':>11}{'avg #ratings':>14}   top words, then top tags")
    for c in np.argsort(-np.bincount(labels, minlength=k)):
        rows = movies[labels == c]
        top = [words[i] for i in np.argsort(-km.cluster_centers_[c])[:3]]
        tag_words = top_tag_words(rows, movies)
        print(
            f"{c:<8}{len(rows):>7}{rows['mean_rating'].mean():>11.2f}{rows['n_ratings'].mean():>14.1f}"
            f"   {', '.join(top)}  |  {', '.join(tag_words) or '-'}"
        )
    decade_clusters = sum(
        any(words[i].startswith("decade") for i in np.argsort(-km.cluster_centers_[c])[:3])
        for c in range(k)
    )
    print(f"clusters with a decade among the top 3 words: {decade_clusters} of {k}")
