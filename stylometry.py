"""Step 5: computes the 14 stylometric features for each email and saves them to stylometric_features.csv."""

import re
import statistics
import pandas as pd

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours"}
URGENCY = {"urgent", "urgently", "immediately", "now", "today", "asap", "expire",
           "expires", "expired", "suspended", "suspend", "deadline", "warning",
           "important", "required", "action", "limited", "within", "hours"}
IMPERATIVE_STARTS = {"click", "verify", "confirm", "update", "reply", "contact",
                     "log", "login", "sign", "visit", "follow", "open", "download",
                     "send", "provide", "call", "please", "kindly", "do", "ensure",
                     "review", "check", "complete", "reset"}

WORD = re.compile(r"[A-Za-z']+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def features(text):
    sentences = [s for s in SENT_SPLIT.split(text) if WORD.search(s)]
    words = WORD.findall(text)
    lower = [w.lower() for w in words]
    n_words = max(len(words), 1)
    n_sent = max(len(sentences), 1)
    sent_lens = [len(WORD.findall(s)) for s in sentences] or [0]
    letters = [c for c in text if c.isalpha()]
    first100 = lower[:100]

    return {
        "n_words": len(words),
        "avg_sentence_len": statistics.mean(sent_lens),
        "sentence_len_std": statistics.pstdev(sent_lens),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "comma_rate": text.count(",") / n_words,
        "exclamation_rate": text.count("!") / n_words,
        "question_rate": text.count("?") / n_words,
        "caps_letter_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "allcaps_word_rate": sum(w.isupper() and len(w) > 1 for w in words) / n_words,
        "first_person_rate": sum(w in FIRST_PERSON for w in lower) / n_words,
        "second_person_rate": sum(w in SECOND_PERSON for w in lower) / n_words,
        "urgency_rate": sum(w in URGENCY for w in lower) / n_words,
        "imperative_share": sum(
            (WORD.findall(s) or [""])[0].lower() in IMPERATIVE_STARTS for s in sentences
        ) / n_sent,
        "lexical_diversity": len(set(first100)) / max(len(first100), 1),
    }


if __name__ == "__main__":
    df = pd.read_csv("clean_emails_v3.csv")
    df = df[df["is_phishing"] == 1].reset_index(drop=True)
    feats = pd.DataFrame([features(t) for t in df["text_strict"]])
    feats["is_ai"] = df["is_ai"].values
    feats.to_csv("stylometric_features.csv", index=False)

    table = feats.groupby("is_ai").mean().T
    table.columns = ["human", "AI"]
    table["AI / human"] = table["AI"] / table["human"]
    print(f"{len(feats)} phishing emails, {feats.shape[1] - 1} features\n")
    print(table.round(3).to_string())
    print("\nSaved: stylometric_features.csv")