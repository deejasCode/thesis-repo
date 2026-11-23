# Detecting AI-Generated Phishing Emails: Code for the Bachelor Thesis

Code of the bachelor thesis "Detecting AI-Generated Phishing and Social Engineering Content: A Practical Detection Approach"
(Khadeeja Awan, IU International University of Applied Sciences, 2026).

The code compares models that tell AI-written phishing emails from human-written ones:
a majority-class baseline, TF-IDF with logistic regression (all words and function words only),
and 14 stylometric features with logistic regression and with a random forest.
Evaluation uses 5-fold cross-validation repeated 5 times, with a random split and with
grouped splits (similar emails always end up in the same fold).

## Data

The data is not included. It is the public dataset of Greco et al. (2024) https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails/data. Download it and place the two folders
`human-generated/` and `llm-generated/` in the same folder as the scripts.
The study uses the 1,503 phishing emails (1,000 AI, 503 human after removing duplicates).

No email text is included in this repository, because some human emails contain personal data.

## Requirements

Python 3.15.0 (other versions were not tested). Install the libraries with:

    py -m pip install -r requirements.txt

## Run order

Run the scripts from the folder that contains them.

| Step | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1 | `py build_dataset.py` | `human-generated/`, `llm-generated/` | `combined_emails.csv` |
| 2 | `py clean_data.py` | `combined_emails.csv` | `clean_emails.csv` |
| 3 | `py normalise.py` | `clean_emails.csv` | `clean_emails_v2.csv` |
| 4 | `py baseline_strict.py` | `clean_emails_v2.csv` | `clean_emails_v3.csv` |
| 5 | `py stylometry.py` | `clean_emails_v3.csv` | `stylometric_features.csv` |
| 6 | `py pair_check.py` | `clean_emails_v3.csv` | the 60 pairs used to choose the threshold |
| 7 | `py final_eval.py` | `clean_emails_v3.csv`, `stylometric_features.csv` | `final_results.csv`, `ablation_results.csv` |
| 8a | `py error_analysis.py` | `clean_emails_v3.csv`, `stylometric_features.csv` | `errors_for_reading.csv`, `random_correct_humans.csv` |
| 8b | `py both_wrong.py` | `errors_for_reading.csv` | `both_wrong.csv` (emails that both models got wrong) |
| 9 | `py make_figures.py` | `No file: the values of Tables B.2 and B.3 are typed into the script` | the two figures of the results chapter |

Step 7 gives the main results of the thesis and takes about four minutes.
Steps 1 to 4 prepare the data, so they only need to run once.
The reading and labelling of pairs, errors and topics was done by hand in spreadsheets and is not part of the scripts.

## Reproducibility

All random choices have a fixed seed: the repeat number (0 to 4) for the folds and `random_state=42` for the random forest.
With the same library versions, `final_eval.py` gives the numbers reported in the thesis.

## Files not in the run order

Early exploration and check scripts (`explore_data.py`, `check_*.py`, `inspect_data.py`, `peek.py`, `top_words.py` and similar) were used while getting to know the data
and are not needed to reproduce the results.