"""Step 2: cleans the table by removing empty texts and exact duplicates (reads combined_emails.csv, writes clean_emails.csv)."""
import pandas as pd

t = pd.read_csv("combined_emails.csv")
t["text"] = t["text"].fillna("")
t["words"] = t["text"].str.split().str.len()

print("start:", len(t))
t = t[t["words"] > 0]
print("after removing empty texts:", len(t))
t = t.drop_duplicates(subset="text")
print("after removing exact duplicates:", len(t))

t[["text", "is_ai", "is_phishing"]].to_csv("clean_emails.csv", index=False)
print(t.groupby(["is_ai", "is_phishing"]).size())
print("saved clean_emails.csv")