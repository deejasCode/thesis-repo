"""Step 6: threshold check. Draws 10 random pairs of emails just above and 10 just below each similarity threshold (0.9, 0.7, 0.5) and saves them for reading by hand (reads clean_emails_v3.csv, writes pair_check.txt and pair_check.csv)."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

THRESHOLDS = [0.9, 0.7, 0.5]
WINDOW = 0.05        # "near the threshold" = within 0.05 of it
PAIRS_PER_SIDE = 10  # 10 pairs just above + 10 just below each threshold = 20
SEED = 0


emails = pd.read_csv("clean_emails_v3.csv")
emails = emails[emails["is_phishing"] == 1].reset_index(drop=True)
texts = emails["text_strict"].fillna("").values
tfidf = TfidfVectorizer().fit_transform(texts)
similarity = (tfidf @ tfidf.T).toarray()

# every pair once (upper triangle, no email paired with itself)
rows_a, rows_b = np.triu_indices(len(emails), k=1)
sim = similarity[rows_a, rows_b]
rng = np.random.default_rng(SEED)


def label(i):
    return "AI" if emails.loc[i, "is_ai"] == 1 else "human"


READ_COLUMN = "text_norm" if "text_norm" in emails.columns else "text_strict"


def excerpt(i, n=600):
    return " ".join(str(emails.loc[i, READ_COLUMN]).split())[:n]


chosen = []
for thr in THRESHOLDS:
    for side, low, high in [("just above", thr, thr + WINDOW), ("just below", thr - WINDOW, thr)]:
        idx = np.where((sim >= low) & (sim < high))[0]
        n_take = min(PAIRS_PER_SIDE, len(idx))
        print(f"threshold {thr}, {side} ({low:.2f} to {high:.2f}): {len(idx)} pairs available, showing {n_take}")
        for k in rng.choice(idx, size=n_take, replace=False):
            chosen.append({"threshold": thr, "side": side, "similarity": round(float(sim[k]), 3),
                           "row_a": int(rows_a[k]), "row_b": int(rows_b[k])})

pairs = pd.DataFrame(chosen)
pairs["label_a"] = [label(i) for i in pairs["row_a"]]
pairs["label_b"] = [label(i) for i in pairs["row_b"]]

# readable file to read, and a sheet to write the verdicts in 
with open("pair_check.txt", "w", encoding="utf-8") as f:
    for n, p in enumerate(pairs.itertuples(), start=1):
        f.write("=" * 90 + "\n")
        f.write(f"PAIR {n}   threshold {p.threshold}, {p.side}, similarity {p.similarity}\n")
        f.write("=" * 90 + "\n")
        f.write(f"--- email A (row {p.row_a}, {p.label_a}) ---\n{excerpt(p.row_a)}\n\n")
        f.write(f"--- email B (row {p.row_b}, {p.label_b}) ---\n{excerpt(p.row_b)}\n\n")

pairs.insert(0, "pair", range(1, len(pairs) + 1))
pairs["verdict"] = ""   # fill in: same / topic
pairs["note"] = ""
pairs.to_csv("pair_check.csv", index=False)
print(f"\n{len(pairs)} pairs written to pair_check.txt (to read) and pair_check.csv (for verdicts)")