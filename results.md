# Results log

A running record of every test, what it found and what it means. Raw material for the README. Update it after each step.

Dataset: MovieLens ml-latest-small, 100,836 ratings, 610 users, 9,742 movies. Harper and Konstan (2015), "The MovieLens Datasets: History and Context", ACM TiiS 5(4).

## Part 1: Looking at the data (`src/explore.py`)

| Measure | Value |
|---|---|
| Ratings | 100,836 |
| Users | 610 |
| Movies in movies.csv | 9,742 |
| Movies with at least one rating | 9,724 |
| Average rating | 3.502 |
| Share of the user-movie table that is filled in | 1.70% |
| Fewest ratings by one user | 20 |
| Median ratings per user | 70.5 |
| Median ratings per movie | 3 |
| Movies with only 1 rating | 3,446 |
| Movies with at least one tag | 1,572 (16.1%) |

What it shows:
- Ratings lean positive. 4.0 is the most common value and whole stars are used far more than half stars.
- A few movies take a large share of the ratings. Forrest Gump has 329 of a possible 610. The median movie has 3.
- The most active user gave 2,698 ratings.
- 98% of the user-movie table is empty. Predicting those cells is the whole task.
- Tags are thin: only 58 users ever tagged, and one of them (user 474) gave 41% of the 3,683 tag rows.

Consequences for later parts: a popularity baseline will look strong, new or rarely rated movies will be hard to predict, and content features will rely mostly on genres and decade.

## Part A: Content-based similarity

### How a movie becomes a text (`src/movie_text.py`)
Genres become lowercase words (hyphens removed, so "Sci-Fi" is `scifi`). The release year becomes a decade token such as `decade1990s`. User tags are lowercased and appended. Seven movies end up with an empty text because they have no genre and no year.

### TF-IDF and cosine similarity, built by hand (`src/tfidf.py`, `src/cosine.py`)
- TF is a word's share of the movie's words. IDF is `log(movies / movies containing the word)`.
- Cosine similarity is `(a . b) / (length a * length b)`. It compares direction, so a movie with many words can match one with few. Scaling a vector by 10 left the score unchanged (0.6447).
- The hand-written NumPy version matches scikit-learn's `cosine_similarity` on five test movies.

### Full model (`src/similar.py`)
scikit-learn TF-IDF over all movies gives a 9,742 by 1,759 table. `similar_to(title, n=5)` compares one movie against all others, with ties broken by number of ratings.

Examples:
- Toy Story: A Bug's Life (0.86), Toy Story 2 (0.65), Antz. Good matches.
- Pulp Fiction: Reservoir Dogs, The Big Lebowski, Django Unchained, Fight Club. Good, and tags helped.
- Heat: every result scores exactly 1.0 because Heat has no tags and many movies share its genres. The tie-break by popularity decides the order.
- The Matrix: Sliding Doors, House of Flying Daggers, Hero, Karate Kid. A failure. The Matrix has the tags `alternate universe` and `martial arts`, which are rare, so Sliding Doors and Karate Kid match on those and ignore that the rest of the movie does not match.

### The test: do "similar" movies get similar ratings? (`src/similarity_eval.py`)
- For a query movie, take its top 5 matches. For each match, take the users who rated both, and average the absolute difference between their two ratings. Lower is better.
- Query movies are those with at least 50 ratings (450 movies). A pair needs at least 5 shared raters to count.
- 100 validation movies pick the setting and 100 different test movies score it. This is repeated over 10 random shuffles and reported as mean (lowest to highest).
- Baselines: random partner movies score 1.003, and always using the 5 most-rated movies scores 1.017.

### Step 5: how much weight should tags get? (`src/tag_weights.py`)
Tag weight multiplies the tag part of each movie's vector. 0 means tags are ignored.

| Tag weight | Test gap in stars |
|---|---|
| 0 (genres and decade only) | 0.867 (0.840 to 0.889) |
| 0.25 | 0.899 |
| 0.5 | 0.913 |
| 1 | 0.911 |
| 2 | 0.889 |
| Random / most rated | 1.003 / 1.017 |

Finding: every setting beats both baselines by about 0.1 to 0.15 stars, so "similar" movies do get more similar ratings than chance. Genres and decade alone were best. Adding the raw tags made it slightly worse.

### Step 6: cleaning the tags (`src/clean_tags.py`)
What the tag data looked like: 3,683 tag rows, 1,589 distinct raw tags, 1,475 after lowercasing, 1,465 after removing punctuation. 923 of the 1,465 tags sit on only one movie. 38% of tag rows are multi-word phrases. `in netflix queue` is on 131 movies and describes a user's habit, not the movie.

Each step was added on top of the previous one, and each was scored at tag weights 0.5, 1, 2 and 4. The weight was picked on the validation movies and scored on the test movies.

| Cleaning step | Movies with a tag | Test gap at best weight (4) |
|---|---|---|
| 1 Current: words, repeats count | 1,572 | 0.882 |
| 2 Merge punctuation variants | 1,572 | 0.882 |
| 3 Keep phrases together (`martial_arts`) | 1,572 | 0.881 |
| 4 One vote per movie | 1,572 | 0.884 |
| 5 Drop tags on a single movie | 1,345 | 0.870 |
| 6 Drop `in netflix queue` | 1,231 | 0.869 |
| No tags at all | | 0.867 |

What helped:
- Steps 2, 3 and 4 changed nothing measurable. They touch only a handful of tags.
- Step 5, dropping tags that sit on a single movie, was the one change that helped, by about 0.014 stars. Such tags cannot link two movies, and they only dilute the weight of the other words.
- Step 6 changed almost nothing.

What did not help: even the fully cleaned tags did not beat using no tags. The best cleaned setting scored 0.869 against 0.867 for no tags, an average difference of -0.002 stars (it was better in 7 of 10 shuffles, but the average is slightly worse). That is well inside the shuffle-to-shuffle noise of about 0.05. Cleaning stopped the tags from hurting, but it did not make them useful.

### Limits of this test
- The best weight in step 6 was 4, the largest value tried, so a still larger weight was not tested.
- Only 16% of movies have tags (12% after cleaning), so tags can affect only a small part of the results.
- Different settings are partly scored on different pairs, because the share of pairs with enough shared raters varies (67% to 74%).
- Movie quality is mixed into the gap. Two well-liked movies have a small gap even if they are not alike.
- Tags come from users who also rated, which could leak a little into the test.
- The test uses all ratings, so it is a measure of similarity quality and not a held-out prediction test. Part C does the held-out test.

## Part B: A map of the movies (`src/clusters.py`, plot in `plots/pca_map.png`)

Setup: K-Means on the genres and decade vectors, since Part A showed that is the best base. The 8 movies with no genre and no year are left out, which leaves 9,734. K-Means was run with k = 5, 10, 15 and 20, and 5 different random starts for each k to see how much luck matters. PCA squeezes the vectors down to 2 dimensions for the plot.

What the data looks like:
- The vocabulary is only 31 words (the genres plus one token per decade).
- The 9,734 movies have only 2,216 distinct vectors, so many movies are exact copies of each other as far as the model can tell. On the map they stack into tight blobs, so the plot adds a tiny jitter to make them visible.
- The 2D map keeps only 19.9% of the variation, so it is a rough picture and not a faithful one.

| k | 5 | 10 | 15 | 20 |
|---|---|---|---|---|
| Silhouette (higher means cleaner clusters) | 0.200 | 0.222 | 0.220 | 0.245 |
| Agreement between random starts (ARI, 1 is identical) | 0.83 | 0.70 | 0.69 | 0.74 |
| Smallest cluster | 1,177 | 439 | 377 | 135 |
| Largest cluster | 2,850 | 1,444 | 1,300 | 837 |
| Match with first-listed genre (ARI) | 0.005 | 0.171 | 0.150 | 0.108 |

What the clusters are:
- k = 5: all five clusters are decades (2000s, 1990s, 2010s, 1960s and 70s, 1980s). Genres barely matter.
- k = 10: genres start to appear. Crime and thriller, action and sci-fi, horror, children and animation, and documentary each get their own cluster. The rest are still decade and drama or comedy mixes.
- k = 15 and 20: more genre clusters (mystery, fantasy, romance) and the old decades split into their own clusters (1930s, 1940s, 1950s). Silhouette keeps rising slightly, but the extra clusters are mostly decades, not new kinds of movie.
- Clusters that have a decade among their top 3 words: 5 of 5, 7 of 10, 11 of 15 and 15 of 20.

Cluster labels from tags were weak. A few made sense (`zombies`, `ghosts` for horror, `disney` and `animation` for children, `organized`, `mafia` and `drugs` for crime, `politics` and `business` for documentaries, `astaire` and `rogers` for the 1930s to 50s). Many were noise: `queue` and `netflix` (one user's habit), actor and character names such as `ferrell` and `jason`, and the word `and`.

What the clusters say about ratings (k = 10): average stars range from 2.93 (horror) to 3.78 (documentary), and old movies (1960s and 70s) average about 3.6. Action and sci-fi clusters get the most ratings per movie (about 21) and documentaries the fewest (about 3). I have not tested why. A likely reason is that only well-liked old movies and documentaries get watched and rated at all, but that is a guess.

Honest check: is it just genres again?
- Mostly yes, by construction. The only inputs are genres and decade, so the clusters can only be combinations of those two things. 90% to 97% of the movies in a cluster carry that cluster's top word.
- It is not only genres. The decade token is as strong as a genre, so the clusters at small k group by era first and genre second. The low match with first-listed genre (0.005 to 0.17) comes from that.
- The structure is weak. Silhouette is 0.20 to 0.25, so clusters overlap a lot, and the map shows one long band where the colors mix. Random restarts agree with each other only 69% to 83%, so the exact cluster boundaries depend on luck.
- Inertia falls steadily as k grows, so there is no clear elbow and no obviously right k. For the README map, k = 10 is the most readable choice. That is a judgment call and not a metric result.

Limits: the plot is a 2D squeeze of 31 dimensions that keeps about a fifth of the variation. The clusters use no rating information. Tags were left out of the features because Part A showed they did not help.

### Searching for a better k (`src/choose_k.py`, plot in `plots/choose_k.png`)

Instead of the four values from the plan, every k from 2 to 30 plus 35 and 40 was scored, each with 5 random starts. Three scores: inertia (how tight the clusters are), silhouette (how cleanly separated they are) and stability (how much random starts agree).

| k | Silhouette | Stability |
|---|---|---|
| 3 | 0.138 | 1.00 |
| 4 | 0.175 | 1.00 |
| 5 | 0.192 | 0.83 |
| 9 | 0.196 | 0.59 |
| 10 | 0.215 | 0.70 |
| 20 | 0.251 | 0.74 |
| 30 | 0.298 | 0.74 |
| 40 | 0.329 | 0.72 |

Findings:
- No k is clearly best. Inertia falls smoothly with no elbow. Silhouette rises all the way to k = 40 without a peak.
- The rising silhouette is not trustworthy here. The 9,734 movies have only 2,216 distinct vectors, so with more clusters each cluster becomes one repeated genre and decade combination, which scores well trivially. Taken to the extreme, one cluster per distinct vector would look perfect and tell us nothing.
- Stability is perfect at k = 3 and 4 but those clusters are too coarse to be useful (silhouette 0.14 and 0.18). It dips to 0.59 at k = 9, jumps back to 0.70 at k = 10, and sits on a plateau of about 0.70 to 0.74 from k = 20 to 30.
- The only visible structure is a small bump at k = 10 (silhouette 0.196 to 0.215 and stability 0.59 to 0.70 compared with k = 9) and the plateau from about 20. So the original guesses of 10 and 20 were reasonable, but the sweep did not find anything better.
- From about k = 24 onward some clusters become very small (as few as 22 movies at k = 40).

Decision: keep k = 10 for the README map because it is the most readable. A finer k of about 20 to 22 is a defensible alternative. Further tuning would not be worth it, since no score picks a clear winner.

## Part C: Predicting ratings (`src/split.py`, `src/baselines.py`, `src/mf.py`, `src/final_score.py`)

Setup: 20% of the ratings are hidden at random as the test pile and every model learns from the other 80%. Everything is repeated on 10 different shuffles. The score is RMSE, the typical miss in stars, and lower is better. The settings of the matrix factorization were chosen on a separate validation pile (10% of all ratings, cut out of the training pile), never on the test pile. About 4% of test ratings are for movies with no training rating. The movie-based models fall back to the overall average (plus the user's bias) for those.

Final test scores, average with range over 10 shuffles:

| Model | RMSE, all test ratings | RMSE, movies with no training rating |
|---|---|---|
| 1. Overall average | 1.043 (1.032 to 1.050) | 1.150 (1.112 to 1.203) |
| 2. Movie average | 0.977 (0.963 to 0.983) | 1.150 (1.112 to 1.203) |
| 3. Movie average + user bias | 0.894 (0.881 to 0.899) | 1.077 (1.041 to 1.137) |
| Matrix factorization | 0.856 (0.843 to 0.862) | 1.026 (0.994 to 1.073) |

Matrix factorization beats the best baseline on every one of the 10 shuffles, by 0.038 on average (0.037 to 0.040). That is 4% less error than baseline 3 and 18% less than guessing the average.

How the settings were chosen (all on the validation pile):
- Without regularization the model overfits quickly. With 10 hidden numbers the training RMSE fell from 0.78 to 0.57 while the score on unseen ratings got worse after about 10 epochs (0.872 up to 0.933).
- Without regularization the number of hidden numbers barely mattered. 5, 10 and 20 tied at about 0.884 and 50 was a little worse (0.890), because bigger lists overfit sooner.
- Regularization removed the overfitting. A strength of 0.1 was the sweet spot. Stronger penalties underfit, and at 0.3 all list sizes landed on 0.882. With regularization bigger lists stopped hurting, and 20 and 50 tied at 0.866 and 0.865.
- Final settings: 20 hidden numbers, strength 0.1, 55 epochs. 50 would have been as good, but 20 is simpler and cheaper.

What this does and does not show:
- The 10 out of 10 wins are less independent than they sound. Each shuffle hides a different random 20%, so the test piles overlap a lot, and the gain looking so steady (0.037 to 0.040) is partly because of that. The ranges show the effect of the shuffle, not of fresh data.
- The learning rate (0.01), the starting size of the hidden numbers and the 55 epochs were fixed by hand or by a coarse grid. A finer search might gain a little more.
- Matrix factorization is still bad on movies it has never seen (RMSE 1.03). It only knows the user's bias there. Part F looks at cold start properly, and the hybrid with content features is a stretch item.
- Only the rating miss is measured here. How good the top 10 lists are is a different question (hit rate in Part E).
