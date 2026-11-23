"""Step 8b (error analysis): reads errors_for_reading.csv and lists the emails that both the style model and the word model got wrong (writes both_wrong.csv)."""

import pandas as pd

errors = pd.read_csv("errors_for_reading.csv")

# the stylometric model was wrong if 'kind' is one of the two mistake kinds;
# the word model was wrong if its prediction (pred_words) points the other way
human_flagged = (errors["kind"] == "human, called AI") & (errors["pred_words"] == 1)
ai_missed = (errors["kind"] == "AI, called human") & (errors["pred_words"] == 0)
both = errors[human_flagged | ai_missed]

print(f"{len(both)} emails wrong in both models "
      f"({int(human_flagged.sum())} human called AI, {int(ai_missed.sum())} AI called human)\n")
print(both[["row", "kind", "prob_ai", "n_words"]].to_string(index=False))
both.to_csv("both_wrong.csv", index=False)
print("\nSaved both_wrong.csv (with the email excerpts for reading in Excel or VS Code; you can sort by prob_ai to see the most confident mistakes first")