"""Step 7: final evaluation for the thesis.

Part A: model comparison on random and grouped splits, with scores for BOTH classes
        (human and AI) and a majority-class baseline.
Part B: feature-group analysis (SQ1) for the stylometric random forest, with the same
        repeated grouped cross-validation.

Run:  py final_eval.py        (needs clean_emails_v3.csv and stylometric_features.csv)
Saves: final_results.csv, ablation_results.csv
"""
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPEATS = 5                       # the 5-fold procedure is repeated 5 times with a new shuffle each time
THRESHOLDS = [0.9, 0.7, 0.5]      # similarity thresholds for the groups (0.7 = main choice)
ABLATION_THRESHOLDS = [0.7]       # Part B is run on the random split and on these grouped splits

# ---------- load the data (phishing emails only, same order in both files) ----------
emails = pd.read_csv("clean_emails_v3.csv")
emails = emails[emails["is_phishing"] == 1].reset_index(drop=True)
style = pd.read_csv("stylometric_features.csv")
assert len(emails) == len(style), "the two files have a different number of rows"
assert (emails["is_ai"].values == style["is_ai"].values).all(), "the two files are not in the same order"
y = emails["is_ai"].values
texts = emails["text_strict"].fillna("").values
X_style = style.drop(columns="is_ai")
print(f"{len(emails)} phishing emails ({int(y.sum())} AI, {int((1 - y).sum())} human)\n")

# ---------- groups of near-identical emails (built from the text only, never from the labels) ----------
tfidf = TfidfVectorizer().fit_transform(texts)
similarity = (tfidf @ tfidf.T).toarray()
group_ids = {}
for thr in THRESHOLDS:
    n_groups, labels = connected_components(csr_matrix(similarity >= thr), directed=False)
    group_ids[thr] = labels
    print(f"similarity >= {thr}: {n_groups} groups, largest has {np.bincount(labels).max()} emails")
print()


def get_folds(split_name, repeat):
    """Return the list of (train, test) index pairs for one repeat of one kind of split."""
    if split_name == "random":
        return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat).split(texts, y))
    thr = float(split_name.split()[1])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat)
    return list(splitter.split(texts, y, groups=group_ids[thr]))


def scores(y_true, y_pred):
    """Precision, recall and F1 for each class. Class 0 = human, class 1 = AI."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {"human_precision": p[0], "human_recall": r[0], "human_f1": f[0],
            "ai_precision": p[1], "ai_recall": r[1], "ai_f1": f[1],
            "false_positive_rate": 1 - r[0]}     # human emails wrongly called AI


def run(split_names, make_models, label):
    """Fit every model on every fold of every repeat and collect the scores."""
    rows, start = [], time.time()
    for split_name in split_names:
        for repeat in range(REPEATS):
            try:
                folds = get_folds(split_name, repeat)
            except ValueError as error:
                print(f"skipped '{split_name}' (repeat {repeat}): {error}")
                continue
            for fold, (train, test) in enumerate(folds):
                if len(set(y[test])) < 2 or len(set(y[train])) < 2:
                    print(f"skipped one fold of '{split_name}' (repeat {repeat}, fold {fold}): only one class")
                    continue
                for model_name, (kind, model) in make_models().items():
                    data = texts if kind == "text" else X_style.values if kind == "style" else X_style[kind].values
                    model.fit(data[train], y[train])
                    rows.append({"split": split_name, "repeat": repeat, "fold": fold, "model": model_name,
                                 **scores(y[test], model.predict(data[test]))})
        print(f"[{label}] finished '{split_name}' after {time.time() - start:.0f} s")
    return pd.DataFrame(rows)


def summary(results, split_names, model_names, columns):
    """Print mean +- standard deviation over all test folds."""
    header = f"{'split':<13}{'model':<38}" + "".join(f"{c.replace('_', ' '):<19}" for c in columns)
    print(header)
    for split_name in split_names:
        for model_name in model_names:
            part = results[(results["split"] == split_name) & (results["model"] == model_name)]
            if part.empty:
                print(f"{split_name:<13}{model_name:<38}no valid folds")
                continue
            cells = "".join(f"{part[c].mean():.3f} +- {part[c].std():.3f}   " for c in columns)
            print(f"{split_name:<13}{model_name:<38}{cells}")
        print()


# =====================================================================================
# PART A: model comparison, both classes, with a majority-class baseline
# =====================================================================================
FUNCTION_WORDS = sorted(ENGLISH_STOP_WORDS)


def models_a():
    return {
        "Majority class (baseline)": ("text", DummyClassifier(strategy="most_frequent")),
        "TF-IDF + LR (all words)": ("text", make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))),
        "TF-IDF + LR (function words only)": ("text", make_pipeline(TfidfVectorizer(vocabulary=FUNCTION_WORDS), LogisticRegression(max_iter=1000))),
        "Stylometric, logistic regression": ("style", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        "Stylometric, random forest": ("style", RandomForestClassifier(n_estimators=300, random_state=42)),
    }


splits_a = ["random"] + [f"grouped {t}" for t in THRESHOLDS]
results_a = run(splits_a, models_a, "A")
results_a.to_csv("final_results.csv", index=False)
print(f"\nPART A: mean +- standard deviation over {5 * REPEATS} test folds (5 folds x {REPEATS} repeats)\n")
print("Human class (class 0): precision, recall, F1")
summary(results_a, splits_a, list(models_a()), ["human_precision", "human_recall", "human_f1"])
print("AI class (class 1): precision, recall, F1")
summary(results_a, splits_a, list(models_a()), ["ai_precision", "ai_recall", "ai_f1"])

# =====================================================================================
# PART B: feature groups for the stylometric random forest (SQ1)
# =====================================================================================
ALL = list(X_style.columns)
GROUPS = {
    "length and rhythm": ["n_words", "avg_sentence_len", "sentence_len_std"],
    "word choice": ["avg_word_len", "lexical_diversity"],
    "punctuation and capitals": ["comma_rate", "exclamation_rate", "question_rate", "caps_letter_ratio", "allcaps_word_rate"],
    "voice": ["first_person_rate", "second_person_rate", "urgency_rate", "imperative_share"],
}
CAPS = ["caps_letter_ratio", "allcaps_word_rate"]
feature_sets = {"all 14 features": ALL}
for name, cols in GROUPS.items():
    feature_sets[f"only {name}"] = cols
for name, cols in GROUPS.items():
    feature_sets[f"without {name}"] = [c for c in ALL if c not in cols]
feature_sets["without the 2 capital-letter features"] = [c for c in ALL if c not in CAPS]
assert all(set(cols) <= set(ALL) for cols in feature_sets.values()), "a feature name is wrong"


def models_b():
    return {name: (cols, RandomForestClassifier(n_estimators=300, random_state=42)) for name, cols in feature_sets.items()}


splits_b = ["random"] + [f"grouped {t}" for t in ABLATION_THRESHOLDS]
results_b = run(splits_b, models_b, "B")
results_b.to_csv("ablation_results.csv", index=False)
print(f"\nPART B: stylometric random forest with different feature groups (mean +- standard deviation)\n")
summary(results_b, splits_b, list(feature_sets), ["ai_f1", "human_f1", "false_positive_rate"])
print("Saved final_results.csv and ablation_results.csv")