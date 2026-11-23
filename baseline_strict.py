"""Step 4: strict cleaning. Builds text_strict by removing the opening greeting, email addresses, links, dataset leftovers (recipient name and domain) and numbers; writes clean_emails_v3.csv. Also compares a TF-IDF model on text_ctrl and text_strict."""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

GREETING = re.compile(r"^(Dear|Hi|Hey|Hello)\b[^,:!\n]{0,40}[,:!]\s*", re.IGNORECASE)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"https?://\S+|www\.\S+")
LEFTOVERS = re.compile(r"\b(?:jose|monkey|dear)\b", re.IGNORECASE)
NUMBER = re.compile(r"\b\d[\d,.:/-]*\b")

def strict(s):
    s = GREETING.sub("", s, count=1)   # opening greeting
    s = EMAIL.sub(" ", s)              # email addresses deleted completely
    s = URL.sub(" ", s)                # links deleted completely (no placeholder)
    s = LEFTOVERS.sub(" ", s)          # dataset leftovers and the word "dear"
    s = re.sub(r"\S*monkey\S*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\.org\b", " ", s, flags=re.IGNORECASE)
    s = NUMBER.sub(" ", s)             # numbers, dates and years
    return re.sub(r"\s+", " ", s).strip()

t = pd.read_csv("clean_emails_v2.csv")
t["text_strict"] = t["text_norm"].astype(str).apply(strict)
t.to_csv("clean_emails_v3.csv", index=False)

ph = t[t["is_phishing"] == 1].reset_index(drop=True)
y = ph["is_ai"]
human = ph[ph["is_ai"] == 0]
ai = ph[ph["is_ai"] == 1]

print("percent of phishing emails containing each word, before -> after cleaning")
print(f"{'word':10s}{'human before':>14s}{'human after':>13s}{'AI before':>11s}{'AI after':>10s}")
for word in ["jose", "monkey", "org", "dear", "2022", "url"]:
    pat = rf"\b{word}\b"
    row = []
    for g in (human, ai):
        row.append(g["text_ctrl"].str.contains(pat, case=False, regex=True).mean() * 100)
        row.append(g["text_strict"].str.contains(pat, case=False, regex=True).mean() * 100)
    print(f"{word:10s}{row[0]:13.0f}%{row[1]:12.0f}%{row[2]:10.0f}%{row[3]:9.0f}%")

def report(name, pred, y_test):
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f = f1_score(y_test, pred, zero_division=0)
    print(f"{name:28s} precision {p:.2f} | recall {r:.2f} | F1 {f:.2f} | false-positive rate {fp / (fp + tn):.2f}")

idx_train, idx_test = train_test_split(ph.index, test_size=0.2, random_state=42, stratify=y)
y_train, y_test = y[idx_train], y[idx_test]
print("\nsame practice/quiz split as before\n")
for col in ["text_ctrl", "text_strict"]:
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(ph.loc[idx_train, col])
    X_test = vec.transform(ph.loc[idx_test, col])
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    report(f"TF-IDF + LR ({col})", model.predict(X_test), y_test)

# inspection only (not a score): which words does the strict version rely on?
vec = TfidfVectorizer()
X = vec.fit_transform(ph["text_strict"])
model = LogisticRegression(max_iter=1000).fit(X, y)
words = np.array(vec.get_feature_names_out())
w = model.coef_[0]
present = X > 0
sa = np.asarray(present[y.values == 1].mean(axis=0)).ravel() * 100
sh = np.asarray(present[y.values == 0].mean(axis=0)).ravel() * 100
order = np.argsort(w)
print("\nWords pushing towards AI (weight, % AI emails, % human emails)")
for i in order[::-1][:12]:
    print(f"  {words[i]:14s}{w[i]:6.2f}{sa[i]:8.0f}{sh[i]:8.0f}")
print("Words pushing towards HUMAN")
for i in order[:12]:
    print(f"  {words[i]:14s}{w[i]:6.2f}{sa[i]:8.0f}{sh[i]:8.0f}")