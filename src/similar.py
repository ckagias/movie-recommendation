from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from movie_text import build_movie_texts

DATA = Path(__file__).resolve().parent.parent / "data"

movies = build_movie_texts()
n_ratings = pd.read_csv(DATA / "ratings.csv").groupby("movieId").size()
movies["n_ratings"] = movies["movieId"].map(n_ratings).fillna(0).astype(int)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(movies["text"])


def find_movie(title):
    exact = movies.index[movies["title"].str.lower() == title.lower()]
    if len(exact) == 1:
        return exact[0]
    partial = movies.index[movies["title"].str.contains(title, case=False, regex=False)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) == 0:
        raise ValueError(f"No movie found for '{title}'")
    options = movies.loc[partial[:10], "title"].tolist()
    raise ValueError(f"'{title}' matches several movies, be more specific: {options}")


def similar_to(title, n=5):
    i = find_movie(title)
    scores = cosine_similarity(X[i], X).ravel()
    result = movies[["title", "genres", "n_ratings"]].copy()
    result["score"] = scores.round(6)
    result = result.drop(index=i)
    result = result.sort_values(["score", "n_ratings"], ascending=False)
    return result.head(n).reset_index(drop=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("vocabulary size:", len(vectorizer.vocabulary_))
    print("matrix shape (movies x words):", X.shape)
    for title in ["Toy Story (1995)", "Heat (1995)", "Pulp Fiction", "Matrix, The (1999)"]:
        print("\n--- movies like", title, "---")
        print(similar_to(title))
