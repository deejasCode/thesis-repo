"""Step 3: normalises the text. text_norm removes invisible characters and extra spaces; text_ctrl also replaces links by URL and removes the opening greeting (reads clean_emails.csv, writes clean_emails_v2.csv)."""

import re
import pandas as pd

t = pd.read_csv("clean_emails.csv")
t["text"] = t["text"].astype(str)

INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
URL = re.compile(r"https?://\S+|www\.\S+")
GREETING = re.compile(r"^(Dear|Hi|Hey|Hello)\b[^,:!\n]{0,40}[,:!]\s*", re.IGNORECASE)

def normalise(s):
    s = INVISIBLE.sub("", s)
    s = re.sub(r"\s+", " ", s)   # also catches \xa0
    return s.strip()

def controlled(s):
    s = URL.sub("URL", s)
    s = GREETING.sub("", s, count=1)
    return s.strip()

t["text_norm"] = t["text"].apply(normalise)
t["text_ctrl"] = t["text_norm"].apply(controlled)

print("texts changed by normalisation:", int((t["text"] != t["text_norm"]).sum()), "of", len(t))
for ai in (0, 1):
    g = t[(t["is_phishing"] == 1) & (t["is_ai"] == ai)]
    print("is_ai", ai,
          "| still contain a non-breaking space:", int(g["text_norm"].str.contains("\xa0").sum()),
          "| still start with a greeting (controlled version):",
          round(g["text_ctrl"].str.match(r"\s*(Dear|Hi|Hey|Hello)\b").mean() * 100), "%")

print("\nBefore and after, 3 random phishing emails:")
for _, r in t[t["is_phishing"] == 1].sample(3, random_state=5).iterrows():
    print("RAW :", repr(r["text"][:90]))
    print("CTRL:", repr(r["text_ctrl"][:90]), "\n")

t.to_csv("clean_emails_v2.csv", index=False)
print("saved clean_emails_v2.csv")