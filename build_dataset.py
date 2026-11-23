"""Step 1: reads the four dataset files and builds combined_emails.csv with the labels is_ai and is_phishing."""

import pandas as pd

# 1. Human emails: normal CSVs, the email text is in the 'body' column
h_legit = pd.read_csv("human-generated/legit.csv")
h_phish = pd.read_csv("human-generated/phishing.csv")

# 2. AI legit emails: loads normally
a_legit = pd.read_csv("llm-generated/legit.csv")

# 3. AI phishing emails: read line by line, split at the LAST comma only
with open("llm-generated/phishing.csv", encoding="utf-8") as fh:
    lines = fh.read().splitlines()[1:]          # skip the header line

print("lines ending in ',1':", sum(l.endswith(",1") for l in lines), "of", len(lines))
a_phish_text = [l.rsplit(",", 1)[0] for l in lines]

# 4. Combine everything into one table
table = pd.concat([
    pd.DataFrame({"text": h_legit["body"],  "is_ai": 0, "is_phishing": 0}),
    pd.DataFrame({"text": h_phish["body"],  "is_ai": 0, "is_phishing": 1}),
    pd.DataFrame({"text": a_legit["text"],  "is_ai": 1, "is_phishing": 0}),
    pd.DataFrame({"text": a_phish_text,     "is_ai": 1, "is_phishing": 1}),
], ignore_index=True)

# 5. Quality checks
print("\nrows before cleaning:", len(table))
print("missing/empty texts:", int(table["text"].isna().sum() + (table["text"].astype(str).str.strip() == "").sum()))
print("duplicate texts:", int(table.duplicated(subset="text").sum()))

table["words"] = table["text"].astype(str).str.split().str.len()
print("\nrows per group:")
print(table.groupby(["is_ai", "is_phishing"]).size())
print("\naverage words per email, per group:")
print(table.groupby(["is_ai", "is_phishing"])["words"].mean().round(1))

table.to_csv("combined_emails.csv", index=False)
print("\nsaved combined_emails.csv")