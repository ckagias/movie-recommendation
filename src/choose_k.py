from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score

from movie_text import build_movie_texts

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

K_VALUES = list(range(2, 31)) + [35, 40]
N_SEEDS = 5

movies = build_movie_texts()
text = movies["genres_text"] + " " + movies["decade_text"]
text = text[text.str.strip() != ""]
X = TfidfVectorizer().fit_transform(text).toarray()

rows = []
for k in K_VALUES:
    runs = [KMeans(n_clusters=k, n_init=10, random_state=s).fit(X) for s in range(N_SEEDS)]
    stability = np.mean([adjusted_rand_score(a.labels_, b.labels_) for a, b in combinations(runs, 2)])
    sil = np.mean(
        [silhouette_score(X, r.labels_, sample_size=3000, random_state=0) for r in runs]
    )
    rows.append(
        {
            "k": k,
            "inertia": np.mean([r.inertia_ for r in runs]),
            "silhouette": sil,
            "stability": stability,
            "smallest": int(min(np.bincount(r.labels_, minlength=k).min() for r in runs)),
        }
    )
    print(f"k={k:<3} silhouette {sil:.3f}  stability {stability:.3f}", flush=True)

table = pd.DataFrame(rows).set_index("k")
table["inertia drop"] = -table["inertia"].diff()
pd.set_option("display.width", 200)
print()
print(table.round(3).to_string())

print("\ntop 5 by silhouette:")
print(table.sort_values("silhouette", ascending=False).head(5)[["silhouette", "stability"]].round(3).to_string())
print("\ntop 5 by stability (k of 5 or more):")
print(table[table.index >= 5].sort_values("stability", ascending=False).head(5)[["silhouette", "stability"]].round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(table.index, table["inertia"], marker="o")
axes[0].set_title("Inertia (lower is tighter)")
axes[1].plot(table.index, table["silhouette"], marker="o")
axes[1].set_title("Silhouette (higher is cleaner)")
axes[2].plot(table.index, table["stability"], marker="o")
axes[2].set_title("Agreement between random starts (higher is steadier)")
for ax in axes:
    ax.set_xlabel("k, the number of clusters")
fig.tight_layout()
fig.savefig(PLOTS / "choose_k.png", dpi=150, bbox_inches="tight")
