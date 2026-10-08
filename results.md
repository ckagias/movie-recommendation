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

## Part D: Explaining recommendations

### Step 1: content-based reasons (`src/explain_content.py`)
- A user's liked movies are those rated 4.0 or more. Every movie the user has not rated gets a score: its highest cosine similarity to any liked movie. The vectors use genres and decade only, because Part A found tags did not help.
- The reason names the liked movie that is closest, then lists what the two share: genres, decade, and cleaned tags (tags on at least 2 movies, no `in netflix queue`). Example: "Shares genre Comedy and the 1990s decade with My Cousin Vinny (1992), which you rated 5.0."
- Only 1,231 of 9,742 movies have a cleaned tag, so tags show up in the reason rarely (1 of 15 examples).
- Limits seen in the demo: all 15 examples scored 1.00. A heavy user has liked a movie with exactly the same genres and decade as many unseen movies, so the score cannot rank them and the tie-break by number of ratings decides. Among several tied liked movies, the highest rated one is named, which is a choice of the code, not evidence that it influenced the pick. The reason is truthful about what matches, but it is a weak reason for a heavy user.

### Step 2: matrix factorization reasons (`src/explain_mf.py`)
- The final model (20 hidden numbers, regularization 0.1, 55 epochs) is fitted on all 100,836 ratings, which takes about 42 seconds. The top 10 for a user are the unseen movies with the highest predicted rating. Movies with fewer than 20 ratings are skipped, because their hidden numbers rest on too little data.
- Reason, part 1: the prediction split into its pieces: overall average + the user's generosity + the movie's pull + taste match (the dot product of the two hidden-number lists).
- Reason, part 2: the 3 movies the user rated 4.0 or more whose hidden numbers have the highest cosine similarity to the recommended movie. Shared genres are added for readability. The model itself never sees genres.
- Sanity check on the hidden numbers: the closest popular movies to Toy Story are Aladdin (0.84) and Wallace & Gromit: The Wrong Trousers (0.73), to The Matrix Inception (0.81), to Pulp Fiction The Usual Suspects (0.77), True Romance (0.76), Fight Club and Reservoir Dogs. The model found sensible structure from ratings alone.
- The split shows when the neighbours are a real reason. User 1 is generous (+0.77) and the taste match is near zero (+0.00 to +0.11) for all their top picks, so those movies are recommended mostly because they are well liked by everyone, and the neighbour list adds little. User 3 is strict (-1.11) but the taste match is large (+0.78 to +1.29), and the neighbours (Saturn 3, Thing, Galaxy of Terror) make sense as a reason.
- Predictions above 5 are cut to 5.00 for display (user 1's top four all show 5.00), but ranking uses the uncut score.
- This is an explanation of the model, not a test. The model has seen every rating, so these numbers say nothing about accuracy.

### Step 3: command line demo (`src/demo.py`)
- Run from `src/`: `python demo.py USER [-n 10] [--refit]`. It prints a short profile of the user (number of ratings, average, the movies they rated highest), then the top N from the content-based method and the top N from matrix factorization, each with its reason.
- The fitted matrix factorization model is saved to `data/mf_model.pkl` (gitignored). The first run takes about 43 seconds to fit, later runs about 1.4 seconds. `--refit` trains it again.
- An unknown user id gives a clear error instead of a crash.
- Both methods use all of the user's ratings, so this is a demo of the explanations, not a test of quality. Quality is measured in Part E.
- Observed in the demo: the two methods often disagree. For user 3 the content-based top 3 are Jurassic Park, Independence Day and Alien (famous, matching genres), while matrix factorization gives Serenity, Waiting for Guffman and Payback. The content-based list is more popular and more obvious, and the learned one is more tied to this user's taste but harder to justify for a movie like Waiting for Guffman.

### Step 3b: notebook demo (`notebooks/demo.ipynb`)
- Same two methods as the command line demo, as a notebook: pick `USER_ID`, see the user's profile and a chart of their star ratings, the content-based top 10 with reasons, the matrix factorization top 10 as a table, the reason for the first pick, and a chart splitting each pick into movie pull and taste match.
- The chart makes the Step 2 finding visible. For user 3, Payback and Pleasantville are almost pure taste picks (movie pull about 0.0), and Wyatt Earp is recommended even though its movie pull is negative. The two methods share only one movie in their top 10 (Blade Runner).
- The notebook was run top to bottom with no errors, and its outputs are saved in the file.
- It reuses the functions in `src/` and the saved model in `data/mf_model.pkl`, so it adds no new modelling.

## Part E: Evaluation

### Step 2: hit rate at 10 (`src/hit_rate.py`)
Setup: the same 10 shuffles as Part C, so each shuffle hides the same 20% of the ratings. Every method learns from the other 80% and ranks the movies a user has not rated in that 80%. A hit is a hidden rating of 4.0 or more that lands in the user's top 10. The candidates are the movies with at least 20 training ratings (about 1,058), the same pool for every method. Users with no hidden liked movie in the pool are skipped (about 585 of 610 are scored per shuffle). Matrix factorization uses the final settings from Part C (20 hidden numbers, regularization 0.1, 55 epochs), and nothing was tuned for this test.

| Method | Hit rate at 10 (at least one hit) | Recall at 10 |
|---|---|---|
| Random | 0.113 (0.095 to 0.134) | 0.010 (0.007 to 0.014) |
| Most rated | 0.565 (0.549 to 0.587) | 0.118 (0.110 to 0.127) |
| Highest movie average | 0.262 (0.205 to 0.310) | 0.034 (0.022 to 0.040) |
| Content-based (genres and decade) | 0.500 (0.465 to 0.542) | 0.091 (0.082 to 0.105) |
| Matrix factorization | 0.319 (0.306 to 0.339) | 0.047 (0.039 to 0.053) |

What it shows:
- Matrix factorization beats random and the highest movie average, so its ranking carries signal. It loses to the non-personal "most rated" list on all 10 shuffles (by 0.21 to 0.27), and to the content-based list.
- The RMSE win in Part C does not carry over to top-10 lists. A model trained to shrink the rating error on rated movies is not trained to rank movies the user will go on to watch and like.
- The content-based list is close to the most-rated list. Many unseen movies tie at a similarity of 1.00 with some liked movie, and ties are broken by number of ratings, so the content-based list is largely popular movies that share genres. That is also why it beats the learned model here.
- "Highest movie average" ranks by one number per movie and has no personal part. Baseline 3 (average plus user bias) gives the same ranking, because a user bias shifts every movie by the same amount.

Diagnosis on shuffle 0 only (a check on the cause, not a tuning step, and nothing was chosen from it). The hit rate of matrix factorization split into its parts, with the candidate pool made stricter:

| Min training ratings | Pool size | Most rated | Matrix factorization | Taste part only | Movie bias only |
|---|---|---|---|---|---|
| 20 (used) | 1,058 | 0.583 | 0.334 | 0.175 | 0.264 |
| 50 | 326 | 0.615 | 0.459 | 0.282 | 0.438 |
| 100 | 80 | 0.708 | 0.629 | 0.602 | 0.565 |
| 200 | 6 | 1.000 | 1.000 | 1.000 | 1.000 |

- The matrix factorization picks lean obscure. The median pick has 44 ratings at the 20-rating cutoff, so the model favours well-liked movies that few people have rated. The hidden liked movies are somewhat better known (median 62 ratings over 10 shuffles, see Part F), so this explains part of the loss and not all of it.
- The personal part adds something. The full score (0.334) beats movie bias alone (0.264), which is the non-personal part. The taste part on its own is the weakest (0.175), so it only helps on top of the bias.
- Raising the cutoff narrows the gap only because the pool shrinks until every user gets nearly the same list. At 200 the pool has 6 movies and every method scores 1.000, which says nothing. A stricter cutoff does not fix the model.
- No sign of a coding error. Random lands close to its expected rate (about 10 hidden liked movies among 1,058 candidates gives roughly 0.1), and the loss to the most-rated list shows on all 10 shuffles.

Limits of the test:
- The hidden ratings are only for movies the user chose to watch, so the test rewards guessing what people watch, which favours popular movies. A liked movie the user never watched counts as a miss. The result says little about how pleasing the lists are to the user, and a high hit rate for "most rated" does not make it a good recommender.
- 32% of the hidden liked ratings (68.4% are in the pool) are for movies with fewer than 20 training ratings and cannot be hit by any method.
- The 20-rating cutoff, the 4.0 threshold for "liked" and the top 10 are fixed choices. The test piles overlap across shuffles, so the ranges show the effect of the shuffle, not of fresh data.
- Part F checks the popularity side directly (how many of each method's top 10 are among the most-rated movies).

### Step 3: does the content-based similarity work? (`src/final_similarity.py`)
Rerun of the Part A test on the final content setup (genres and decade, no tags, ties broken by number of ratings). Same method as before: for 100 test movies per shuffle (at least 50 ratings each), take the top 5 matches and average the rating gap between users who rated both, over 10 shuffles. No setting was chosen in this run. The setting was fixed in Part A.

| Partners | Rating gap in stars | Content-based is lower in |
|---|---|---|
| Content-based top 5 (genres and decade) | 0.867 (0.840 to 0.889) | |
| Random movies | 1.003 (0.951 to 1.039) | 10 of 10 shuffles |
| The 5 most-rated movies | 1.017 (0.971 to 1.043) | 10 of 10 shuffles |
| Random movies with at least 50 ratings | 0.954 (0.932 to 0.978) | 10 of 10 shuffles |

Average advantage over each baseline, same test movies in each shuffle: random 0.135 (0.080 to 0.170), most rated 0.150, random popular 0.087 (0.065 to 0.121). 74% of the match pairs (69% to 79%) had at least 5 shared raters and were scorable.

What it shows:
- The match lists are better than chance on every shuffle. Part A's numbers reproduce exactly (0.867, 1.003, 1.017).
- The new control tests the limit that movie quality is mixed into the gap. Random partners from movies with 50 or more ratings already score 0.954 instead of 1.003, so about a third of the advantage over plain random (0.049 of 0.135 stars) comes from picking well-known, well-liked movies. The remaining 0.087 stars is the part that genre and decade explain.
- The control is not a perfect match. Ties are broken by number of ratings, so the actual matches are probably more popular than a random movie with 50 ratings. The true effect of similarity could be somewhat smaller than 0.087.

Limits:
- The gap is a rating gap, not a quality score. It says users rate similar movies similarly. It does not say they would enjoy the match.
- Only pairs with at least 5 shared raters count (74%), and these are skewed towards popular movies.
- Only the 450 movies with 50 or more ratings are used as queries, so the result says nothing about rarely rated movies, which is where content features would matter most.
- The test uses all ratings, not a held-out pile. It measures how well genres and decade describe similarity, and was not used to fit anything beyond the tag weight in Part A.
- Hidden-number (matrix factorization) similarity is not scored on this test. The model learned from the same ratings that the gap is computed on, so a low gap would be close to guaranteed and prove little. A fair version needs held-out raters, and there are too few shared raters in 20% of the data to score pairs.
- Test movies overlap across shuffles, so the ranges show the effect of the shuffle and not of fresh data.

### Step 5: results table (`src/results_table.py`)
One script reruns every test and prints the final tables, with one matrix factorization fit per shuffle shared by the RMSE and hit rate scores (about 6 minutes). All numbers match the earlier runs (Part C, step 2 and step 3), which checks that nothing drifted. 10 shuffles, mean with lowest to highest in brackets.

| Model | RMSE in stars (lower is better) | Hit rate at 10 (higher is better) | Recall at 10 |
|---|---|---|---|
| 1. Overall average | 1.043 (1.032 to 1.050) | n/a | n/a |
| 2. Movie average | 0.977 (0.963 to 0.983) | 0.262 (0.205 to 0.310) | 0.034 (0.022 to 0.040) |
| 3. Movie average + user bias | 0.894 (0.881 to 0.899) | same as baseline 2 | same as baseline 2 |
| Content-based (genres and decade) | n/a | 0.500 (0.465 to 0.542) | 0.091 (0.082 to 0.105) |
| Matrix factorization | **0.856** (0.843 to 0.862) | 0.319 (0.306 to 0.339) | 0.047 (0.039 to 0.053) |
| Reference: random list | n/a | 0.113 (0.095 to 0.134) | 0.010 (0.007 to 0.014) |
| Reference: most rated movies | n/a | **0.565** (0.549 to 0.587) | **0.118** (0.110 to 0.127) |

| Similarity test: partners for 100 test movies | Rating gap in stars (lower is better) | Content-based is lower in |
|---|---|---|
| Content-based top 5 (genres and decade) | 0.867 (0.840 to 0.889) | |
| Random movies | 1.003 (0.951 to 1.039) | 10 of 10 shuffles |
| The 5 most-rated movies | 1.017 (0.971 to 1.043) | 10 of 10 shuffles |
| Random movies with at least 50 ratings | 0.954 (0.932 to 0.978) | 10 of 10 shuffles |

Reading the tables:
- Matrix factorization is best at predicting ratings and beats baseline 3 on 10 of 10 shuffles. It is not best at the top 10 lists. The non-personal most-rated list beats it on 10 of 10 shuffles, and the content-based list beats it as well. The two jobs give different winners.
- Baseline 3 has no separate hit rate. A user bias adds the same amount to every movie for that user, so it cannot change the order, and its list is the same as baseline 2.
- The content-based method cannot be scored by RMSE because it ranks movies and predicts no ratings. Its only scores are the hit rate and the similarity test.
- The hit rate numbers use the pool of movies with at least 20 training ratings and the limits listed in step 2.
- Baseline 1 gives every movie the same score, so it has no ranking.

## Part F: Mistakes and limits

### Does the model favor popular movies? (`src/popularity.py`)
The top 10 lists from the hit rate test (same 10 shuffles, same pool of movies with at least 20 training ratings) were measured for how well known their movies are. Popularity is the number of training ratings. "Most rated" means the 100 movies with the most training ratings, which have at least 91 ratings (90 to 92). The last row measures the hidden liked movies, the target the lists are trying to hit.

| List | Median ratings per recommended movie | Picks in the 100 most rated | Share of the pool ever recommended |
|---|---|---|---|
| Random | 35 (34 to 35) | 7.8% (7.2% to 8.2%) | 99.5% |
| The most rated movies | 182 (176 to 189) | 100.0% | 5.0% (4.7% to 5.5%) |
| Highest movie average | 37 (24 to 67) | 23.3% (7.2% to 40.8%) | 2.9% (2.5% to 3.3%) |
| Content-based | 103 (101 to 105) | 64.7% (61.2% to 67.5%) | 36.3% (33.9% to 39.6%) |
| Matrix factorization | 41 (37 to 44) | 23.3% (20.3% to 25.8%) | 30.8% (28.7% to 32.2%) |
| Hidden liked movies (the target) | 62 (60 to 63) | 31.0% (29.7% to 32.0%) | n/a |

What it shows:
- Matrix factorization does not favor popular movies. It leans the other way: its picks are less known than the movies users actually liked (median 41 against 62 ratings, 23% against 31% in the 100 most rated). The earlier diagnosis showed that the movie bias carries much of its score, and movies with few, high ratings get large biases (for example A Streetcar Named Desire, 20 ratings, bias +0.82 in the demo).
- The content-based list is the one that leans popular (median 103, 65% in the 100 most rated), because many movies tie at the top similarity and ties are broken by number of ratings. It is more popular than the target and less popular than the most-rated list.
- The hidden liked movies are over-represented among the best known movies: 31% are in the 100 most rated, against 9.5% if they were spread evenly over the 1,058 movies in the pool (100 of 1,058). That is why a non-personal most-rated list scores a hit rate of 0.565, even though it is more popular than the target.
- Personalization is real in how much is recommended. The most-rated list can reach only 5.0% of the pool and the highest movie average 2.9%, since nearly every user gets the same list (the users differ only by which of those movies they have already rated). Matrix factorization recommends 30.8% of the pool to someone and content-based 36.3%. These personal picks just do not land on the movies users go on to rate.
- The highest movie average is unstable between shuffles (median 24 to 67, 7% to 41% in the most rated), because the top of a ranking by averages is taken by movies with a few high ratings.

Limits:
- Popularity here is the number of training ratings, which grows with how many people chose to rate a movie, and not a direct measure of how famous it is.
- It describes the lists of the hit rate test, with its pool and its definition of liked (4.0 or more), not the demo lists.
- The lists have not been checked for repeats of the same franchise or the same genre mix. Coverage counts distinct movies only.

### The biggest misses (`src/biggest_misses.py`)
Matrix factorization on the test pile, 10 shuffles. The worst 1% of ratings (about 200 per shuffle) are misses of at least 2.55 stars (2.46 to 2.59), compared with an RMSE of 0.856 overall. The ten worst misses on shuffle 0:

| User | Movie | Actual | Guess | Movie's training ratings | User's average |
|---|---|---|---|---|---|
| 3 | Troll 2 (1990) | 5.0 | 0.68 | 1 | 2.39 |
| 543 | The Artist (2011) | 0.5 | 4.48 | 9 | 4.49 |
| 175 | Stay (2005) | 0.5 | 4.46 | 2 | 4.09 |
| 573 | Inside Man (2006) | 0.5 | 4.39 | 28 | 4.26 |
| 51 | Austin Powers: The Spy Who Shagged Me (1999) | 0.5 | 4.36 | 97 | 3.82 |
| 594 | Opera (1987) | 0.5 | 4.33 | 1 | 3.93 |
| 89 | Forrest Gump (1994) | 0.5 | 4.27 | 257 | 3.50 |
| 573 | Crouching Tiger, Hidden Dragon (2000) | 0.5 | 4.22 | 84 | 4.26 |
| 125 | Amelie (2001) | 0.5 | 4.18 | 97 | 3.87 |
| 527 | Schindler's List (1993) | 1.0 | 4.68 | 176 | 4.17 |

Worst 1% against all test ratings, mean over 10 shuffles (lowest to highest):

| | Worst 1% | All test ratings |
|---|---|---|
| Model guessed too high | 92.9% (90.1% to 96.0%) | 46.8% |
| Actual rating 1.5 or lower | 91.2% (87.6% to 94.1%) | 5.9% |
| Actual rating 4.5 or higher | 7.0% | 21.5% |
| Gap between the actual rating and the user's own average | 2.77 stars | 0.73 stars |
| Movie has no training rating | 9.1% (6.4% to 12.9%) | 4.1% |
| Movie has fewer than 5 training ratings | 22.2% | 14.9% |
| Median training ratings of the movie | 27 | 30 |
| Median training ratings of the user | 181 (134 to 201) | 306 |

What the biggest misses share:
- They are very low ratings (0.5 or 1.0) on movies the model expected the user to like. 9 of the 10 listed and 91% of the worst 1% are ratings of 1.5 or lower, against 6% of all ratings. The user rated against their own habit by 2.8 stars on average, against 0.7 for a typical rating.
- They are mostly not cold start. Median movie popularity is close to the overall figure (27 against 30 ratings), several are famous movies (Forrest Gump 257 training ratings, Schindler's List 176), and only 9% are movies with no training rating. Cold start does show up a little, since movies with no training rating are twice as common among the worst misses (9% against 4%). The users have a median of 181 training ratings, so they are not new users either, though they are less active than the typical user (306).
- Together they cost a lot. The worst 1% of ratings make up 12.1% of the total squared error (11.7% to 12.5%). By simple arithmetic the RMSE without them would be about 0.81 instead of 0.856.
- One of the ten goes the other way: user 3 gave Troll 2 a 5.0 while the model guessed 0.68. The movie has a single training rating, so its bias was fitted from one data point (the guess suggests that rating was low, but I did not check it). It is a movie with a cult reputation for being so bad it is good, which no feature in this data can express.
- Two of the ten come from the same user (573, who averages 4.26). I did not check how concentrated the worst misses are among users.

What to take from it:
- Nothing in this data tells a real dislike from a slip (a half star meant as 4.5) or from a different use of the scale, so these misses are probably close to unavoidable for a model that sees only ratings. That is a guess from the pattern, and I did not test it.
- A better model would help with the other 99% more than with these. Capping the guess or clipping the loss would not move the largest misses.

### Cold start (`src/cold_start.py`)
Two questions: how bad is the miss for movies with few ratings, and for users with few ratings? 10 shuffles, RMSE in stars as mean (lowest to highest). Matrix factorization uses the final settings. The script runs the shuffles in parallel and takes about 8 minutes.

New movies. Test ratings are grouped by how many training ratings their movie has:

| Training ratings of the movie | Share of test ratings | Baseline 2, movie average | Baseline 3, movie average + user bias | Matrix factorization |
|---|---|---|---|---|
| 0 | 4.1% | 1.150 (1.112 to 1.203) | 1.077 (1.041 to 1.137) | 1.026 (0.994 to 1.073) |
| 1 | 3.2% | 1.294 (1.221 to 1.360) | 1.226 (1.164 to 1.327) | 0.958 (0.934 to 1.032) |
| 2 to 4 | 7.6% | 1.087 (1.055 to 1.119) | 1.009 (0.978 to 1.041) | 0.917 (0.884 to 0.948) |
| 5 to 9 | 10.0% | 1.005 (0.992 to 1.028) | 0.915 (0.902 to 0.951) | 0.882 (0.865 to 0.914) |
| 10 to 19 | 14.2% | 0.952 (0.920 to 0.974) | 0.854 (0.822 to 0.876) | 0.836 (0.803 to 0.866) |
| 20 to 49 | 27.9% | 0.948 (0.939 to 0.961) | 0.858 (0.849 to 0.867) | 0.842 (0.829 to 0.851) |
| 50 or more | 33.1% | 0.916 (0.901 to 0.926) | 0.841 (0.829 to 0.852) | 0.818 (0.805 to 0.831) |

New users. Each shuffle, 100 random users keep only k of their training ratings (chosen at random), the models are refitted, and they are scored on those users' hidden test ratings (about 3,300 per shuffle). The last row scores the same users with their full history:

| Ratings kept | Baseline 2, movie average | Baseline 3, movie average + user bias | Matrix factorization |
|---|---|---|---|
| 0 | 0.986 (0.917 to 1.066) | 0.986 (0.917 to 1.066) | 0.964 (0.897 to 1.053) |
| 1 | 0.986 | 1.199 (1.092 to 1.395) | 0.967 (0.897 to 1.029) |
| 2 | 0.986 | 1.061 (1.008 to 1.122) | 0.961 (0.916 to 1.008) |
| 5 | 0.986 | 0.972 (0.934 to 1.046) | 0.935 (0.900 to 1.005) |
| 10 | 0.986 | 0.936 (0.887 to 0.997) | 0.909 (0.870 to 0.973) |
| 20 | 0.986 | 0.922 (0.875 to 0.996) | 0.893 (0.850 to 0.963) |
| All (full history) | 0.979 (0.911 to 1.059) | 0.899 (0.841 to 0.973) | 0.862 (0.813 to 0.932) |

What it shows:
- New movies cost accuracy, and it fades by about 10 ratings. Matrix factorization is at 1.026 for a movie with no ratings and 0.958 with one, against 0.856 on average. From 10 ratings on it is close to its overall level (0.84 to 0.82). Movies with 0 or 1 training rating are 7.3% of the test ratings.
- A single rating is worse than none for the simple baselines. Baseline 2 goes from 1.150 (no ratings) to 1.294 (one rating), because one rating, possibly an extreme one, is taken as the whole truth for the movie. Matrix factorization goes the other way (1.026 to 0.958), since regularization pulls a movie's bias back towards zero when it rests on little data.
- The same happens for new users. Baseline 3 at one kept rating (1.199) is much worse than baseline 2, which ignores the user (0.986), because one rating sets the user's whole bias. Matrix factorization stays at 0.967. It needs about 5 ratings to be clearly better than knowing nothing (0.935 against 0.964) and about 20 to get to 0.893. With the full history it reaches 0.862.
- Matrix factorization is the best of the three at every row, but the margin over baseline 3 is small once there are 10 or more ratings, and the shuffle-to-shuffle ranges overlap. The big advantage is at 0 to 5 ratings, and mostly comes from regularization and not from the hidden numbers.
- How bad the miss is: for a new user about 0.1 stars worse than the same users with their full history (0.964 against 0.862). For a new movie about 0.17 worse than the overall 0.856 (1.026 at 0 ratings). Neither is catastrophic. With no ratings the model has only averages to go on, and for a new user it is only slightly better than the plain movie average (0.964 against 0.986).

Limits:
- The cold users are simulated. They are existing users with ratings removed, who may rate differently from real new users, who could be less engaged or have no history at all.
- The cold-user runs train on about 19% fewer ratings, because the 100 users hold about 15,000 of the 80,669 training ratings. So the gap to the full-history row mixes "fewer ratings for the user" with "less training data overall", and the two are not separated.
- The 100 users change with the shuffle, and their test ratings number about 3,300, so the ranges are wide (for example 0.897 to 1.053 at 0 ratings kept).
- Test movies and users overlap across shuffles, as in the other parts.
- The hybrid with content features (genres) for new movies was not tried. It is a stretch item in the plan.

## What failed, and what I would try next

Written from the results above. Every number is from this document.

### What failed
1. **Matrix factorization loses the top 10 test.** It has the lowest rating error (RMSE 0.856, best of the three baselines 0.894, wins 10 of 10 shuffles), but its top 10 lists hit a hidden liked movie for 0.319 of users. A plain list of the most-rated movies gets 0.565, and the content-based list 0.500. Matrix factorization loses to the most-rated list on all 10 shuffles. It was trained to predict ratings of movies people rated, not to find the movies they go on to watch, and its picks lean obscure (median 41 ratings against 62 for the hidden liked movies).
2. **Tags did not help.** In the similarity test, genres and decade alone were best (rating gap 0.867 stars). Raw tags made it worse (0.911 at weight 1). Cleaning the tags stopped the harm but not more (0.869 against 0.867 with no tags). Only 16% of movies have a tag at all, and one user wrote 41% of the tag rows. Tags also broke some searches, such as The Matrix, whose rare tags matched Sliding Doors and Karate Kid.
3. **The content features are thin.** MovieLens has no cast, director or plot, so the content side only sees genres and decade. The 9,734 movies with a genre or year collapse into 2,216 distinct vectors, so many movies look identical to the model. In the content-based top 10, many movies tie at similarity 1.00 and the order is decided by number of ratings, which pushes the list towards popular movies (65% of its picks are among the 100 most rated).
4. **The clusters are mostly genre and decade, with weak structure.** Silhouette was 0.20 to 0.25 for k = 5 to 20, no k was clearly best, and the apparent rise of silhouette at larger k is an artefact of the repeated vectors. The 2D map keeps 19.9% of the variation. Cluster labels from tags were mostly noise.
5. **Matrix factorization overfit at first.** With no regularization the error on unseen ratings fell to 0.872 after about 10 epochs and then rose to 0.933, while the training error kept falling. Regularization (0.1) fixed it. The learning rate was fixed by hand and not tuned.
6. **New movies and new users are hard.** Matrix factorization misses by 1.026 stars on a movie with no training rating and 0.958 with one (0.856 overall), and by 0.964 for a simulated new user (0.862 with full history). For the simple baselines one rating is worse than none, because they take a single rating as the whole truth.
7. **The biggest misses look unpredictable.** The worst 1% of ratings are mostly 0.5 or 1.0 stars on movies the user usually rates high (91% are 1.5 or lower, against 6% of all ratings), and they make up 12.1% of the squared error. Ratings alone do not tell a real dislike from a slip or an unusual use of the scale. This is a reading of the pattern, not a tested explanation.
8. **The explanations are partial.** The content-based reason is true but weak for heavy users: all 15 examples scored 1.00, and the liked movie it names is the highest rated among ties, not proof it caused the pick. For matrix factorization, the breakdown of the prediction (average, generosity, movie pull, taste match) is exact, but the "closest liked movies" are a similarity view of the hidden numbers and not how the score is computed. For some users, such as user 1, the taste match is near zero, so the neighbours add little.

### Where the evidence is weaker than it looks
- The 10 shuffles use random 80/20 splits of the same ratings, so their test piles overlap. The ranges show the effect of the shuffle, not of fresh data, and "wins 10 of 10" is less independent than it sounds.
- The hit rate test counts only movies the user chose to rate. A liked movie the user never watched is a miss, and the test favours popular movies. 32% of the hidden liked ratings are outside the candidate pool (fewer than 20 training ratings), so no method can hit them.
- The cold-start users are simulated by removing ratings from existing users, and their runs train on about 19% less data.
- The similarity test uses all ratings and not a held-out pile. Hidden-number similarity was not scored on it because the model learned from the same ratings.
- Only `ml-latest-small` (100,836 ratings) was used, with a random split and no time-based split.
- One claim was corrected along the way: the hidden liked movies are not "mostly well known". Their median is 62 ratings, with 31% among the 100 most rated.

### What I would try next
1. A hybrid re-rank for the top 10: popular movies filtered by predicted rating, with the blend chosen on a validation pile and scored once on the test pile. The popularity analysis suggests the target sits in the middle of the popularity range.
2. A model trained on who rated what, ignoring the stars (an implicit-feedback ranking model such as BPR), since that is the job the hit rate test measures.
3. Content features for new movies, so a movie with no ratings is not guessed from the averages alone.
4. A time-based split (learn from older ratings, test on newer ones), which is harder and closer to real use.
5. A larger dataset (the 32M version) to see whether the ranking of the methods holds.
