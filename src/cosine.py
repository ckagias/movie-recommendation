import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tfidf_by_hand import tfidf


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


titles = list(tfidf.columns)
vectors = {t: tfidf[t].to_numpy() for t in titles}

by_hand = pd.DataFrame(
    [[cosine(vectors[a], vectors[b]) for b in titles] for a in titles],
    index=titles,
    columns=titles,
)

by_sklearn = pd.DataFrame(
    cosine_similarity(tfidf.T.to_numpy()), index=titles, columns=titles
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print("--- cosine similarity, written by hand ---")
print(by_hand.round(3))
print("\nmatches scikit-learn:", np.allclose(by_hand, by_sklearn))

print("\n--- one pair, step by step ---")
a, b = vectors["Toy Story (1995)"], vectors["Jumanji (1995)"]
print("dot product:", round(float(np.dot(a, b)), 4))
print("length of Toy Story vector:", round(float(np.linalg.norm(a)), 4))
print("length of Jumanji vector:", round(float(np.linalg.norm(b)), 4))
print("dot / (length * length):", round(float(cosine(a, b)), 4))

print("\n--- length does not matter ---")
print("Toy Story vs Jumanji:", round(float(cosine(a, b)), 4))
print("Toy Story times 10 vs Jumanji:", round(float(cosine(a * 10, b)), 4))
