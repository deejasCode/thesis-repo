"""Step 8a (error analysis): one grouped run (similarity 0.7, seed 0) in which every email is predicted by models that did not see it or its near-copies. Prints the error counts, feature means, model confidence, error rate by length and the overlap between the style and word models. Saves the emails that at least one model got wrong (errors_for_reading.csv) and 25 random human emails that both models got right (random_correct_humans.csv) for reading by hand."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

THRESHOLD = 0.7   # similarity threshold used to build the groups
SEED = 0

# load the data (phishing emails only, same order in both files)
emails = pd.read_csv("clean_emails_v3.csv")
emails = emails[emails["is_phishing"] == 1].reset_index(drop=True)
style = pd.read_csv("stylometric_features.csv")
assert len(emails) == len(style) and (emails["is_ai"].values == style["is_ai"].values).all()
y = emails["is_ai"].values
texts = emails["text_strict"].fillna("").values
reading = emails["text_norm" if "text_norm" in emails.columns else "text_strict"].fillna("").values
X = style.drop(columns="is_ai")

# groups (built from text similarity only)
tfidf = TfidfVectorizer().fit_transform(texts)
similarity = (tfidf @ tfidf.T).toarray()
check_rows = [61, 65, 68, 55, 18, 47, 2]          # rows of the CSV to compare
print(pd.DataFrame(similarity[np.ix_(check_rows, check_rows)], index=check_rows, columns=check_rows).round(2).to_string())
_, groups = connected_components(csr_matrix(similarity >= THRESHOLD), directed=False)

# one prediction per email, made by a model that did not see it or its near-twins 
pred_style = np.zeros(len(y), dtype=int)
prob_style = np.zeros(len(y))
pred_words = np.zeros(len(y), dtype=int)
folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(texts, y, groups=groups)
for train, test in folds:
    forest = RandomForestClassifier(n_estimators=300, random_state=42).fit(X.values[train], y[train])
    prob_style[test] = forest.predict_proba(X.values[test])[:, 1]
    pred_style[test] = forest.predict(X.values[test])
    words = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)).fit(texts[train], y[train])
    pred_words[test] = words.predict(texts[test])

results = X.copy()
results["is_ai"] = y
results["pred_style"] = pred_style
results["pred_words"] = pred_words
results["prob_ai"] = prob_style

kinds = np.select(
    [(y == 0) & (pred_style == 0), (y == 0) & (pred_style == 1), (y == 1) & (pred_style == 1), (y == 1) & (pred_style == 0)],
    ["human, correct", "human, called AI", "AI, correct", "AI, called human"], default="")
results["kind"] = kinds

# 1. how many errors 
print(f"Stylometric random forest, grouped folds (similarity >= {THRESHOLD}), one prediction per email\n")
print(results["kind"].value_counts().reindex(["human, correct", "human, called AI", "AI, correct", "AI, called human"], fill_value=0).to_string())
n_h, n_a = int((y == 0).sum()), int((y == 1).sum())
print(f"\nfalse-positive rate: {(kinds == 'human, called AI').sum() / n_h:.3f}   "
      f"missed-AI rate: {(kinds == 'AI, called human').sum() / n_a:.3f}")

# 2. how do the mistakes differ from the correct ones? (feature means) 
print("\nAverage feature values by group (columns: human correct / human wrongly called AI / AI correct / AI wrongly called human)\n")
order = ["human, correct", "human, called AI", "AI, correct", "AI, called human"]
table = results.groupby("kind")[list(X.columns)].mean().reindex(order).T
table.columns = ["human OK", "human->AI", "AI OK", "AI->human"]
print(table.round(3).to_string())

# 3. how sure was the model when it was wrong? 
wrong_h = results[results["kind"] == "human, called AI"]["prob_ai"]
wrong_a = results[results["kind"] == "AI, called human"]["prob_ai"]
print(f"\nProbability of 'AI' given by the model: human emails wrongly called AI: average {wrong_h.mean():.2f}; "
      f"AI emails wrongly called human: average {wrong_a.mean() if len(wrong_a) else float('nan'):.2f}")

# 4. error rate by email length 
length_group = pd.qcut(results["n_words"], 4, duplicates="drop")
by_length = results.assign(wrong=(results["pred_style"] != results["is_ai"]), length=length_group)
print("\nShare of wrong predictions by email length (words), per true class\n")
print(by_length.pivot_table(index="length", columns="is_ai", values="wrong", aggfunc="mean", observed=True)
      .rename(columns={0: "human emails", 1: "AI emails"}).round(3).to_string())

# 5. do the stylometric model and the word model make the same mistakes?
wrong_style = pred_style != y
wrong_words = pred_words != y
print(f"\nStylometric model wrong: {wrong_style.sum()} | word model wrong: {wrong_words.sum()} | "
      f"both wrong on the same email: {(wrong_style & wrong_words).sum()} | only stylometric wrong: {(wrong_style & ~wrong_words).sum()}")

# 6. save the mistakes to read them 
out = results[wrong_style | wrong_words].copy()
out["email_excerpt"] = [reading[i][:400].replace("\n", " ") for i in out.index]
cols = ["kind", "prob_ai", "pred_words", "n_words", "email_excerpt"]
out[cols].sort_values(["kind", "prob_ai"]).to_csv("errors_for_reading.csv", index_label="row")
print(f"\nSaved {len(out)} emails to errors_for_reading.csv (open it in Excel or VS Code and read them)")

# 7. a random sample of human emails that BOTH models classified correctly
typical = results[(y == 0) & (pred_style == 0) & (pred_words == 0)]
sample = typical.sample(n=min(25, len(typical)), random_state=SEED).copy()
sample["email_excerpt"] = [reading[i][:400].replace("\n", " ") for i in sample.index]
sample[["kind", "prob_ai", "n_words", "email_excerpt"]].sort_index().to_csv("random_correct_humans.csv", index_label="row")
print(f"\nSaved {len(sample)} randomly chosen correctly classified human emails (out of {len(typical)}) to random_correct_humans.csv")