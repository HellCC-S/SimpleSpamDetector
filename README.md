# Spam Detector (TF-IDF + Naive Bayes)

A small, practical email spam classifier written in **Python 3** using **scikit-learn**.  
The model is a single `Pipeline`:

- **TF-IDF Vectorizer** (unigrams + bigrams, English stopwords, `min_df=2`)
- **Multinomial Naive Bayes** classifier

It supports:

- Training with a train/test split (stratified when possible)
- Evaluation on the held-out test set (classification report + confusion matrix)
- Saving/loading the model as a `.pkl` file (the whole Pipeline)
- Interactive CLI prediction (type emails, get predictions + probabilities)
- Batch prediction from a CSV file (append predictions to a new CSV)
- A configurable **spam probability threshold** for decisions

---

## 1) What This Project Does

This project learns to classify text messages (emails) into two categories:

- `spam`: promotional/scam-like emails
- `ham`: normal emails (non-spam)

### How it works (high level)
1. **TF-IDF** converts raw text into numeric features:
   - Words and short phrases (1-gram and 2-gram) become feature dimensions
   - TF-IDF down-weights very common words and up-weights informative words
2. **Naive Bayes** learns which features are more likely in spam vs ham:
   - It outputs a probability for each class: `P(spam)` and `P(ham)`
3. The final label is decided by thresholding:
   - If `P(spam) >= threshold` → `spam`
   - Otherwise → `ham`

---

## 2) Repository Layout (Suggested)

.
├── spam_detector.py # main CLI script
├── emails.csv # your own training dataset (optional)
├── emails_500.csv # generated demo dataset (optional)
├── spam_model.pkl # generated after training
├── new_emails.csv # optional: batch prediction input
└── preds.csv # optional: batch prediction output

yaml
复制代码

If you are working in a clean folder, the minimum is:

- `spam_detector.py`
- a training CSV file (ex: `emails_500.csv`)

---

## 3) Requirements

### Python
- Recommended: **Python 3.9+** (3.8 often works too)

### Python packages
- `pandas`
- `joblib`
- `scikit-learn`

Install:

```bash
pip install pandas joblib scikit-learn
Verify installation:

bash
复制代码
python -c "import sklearn, pandas, joblib; print('OK')"
4) Training Data Format
Required columns
Your training CSV must contain:

text — the email content

label — the ground-truth label (spam or ham)

Example:

csv
复制代码
text,label
"Hi, are we still meeting tomorrow?",ham
"Limited time offer! Claim your prize now!",spam
Notes
The script normalizes labels to lowercase (Spam → spam, HAM → ham)

text is coerced to string; empty values become "nan" if present in CSV

For best results, keep the dataset balanced (or at least not extremely skewed)

5) Model Details (What Exactly Is Being Trained)
Inside build_model():

TfidfVectorizer(...)

lowercase=True: normalize case

stop_words="english": removes common English words like “the”, “and”

ngram_range=(1,2): uses unigrams and bigrams

min_df=2: ignores very rare terms (appearing in only 1 document)

MultinomialNB()

a classic, strong baseline for text classification

fast to train and surprisingly effective for spam detection

Because it’s a Pipeline, saving the model saves both:

the vectorizer

the classifier

So you can directly load and predict without re-fitting TF-IDF.

6) Usage
6.1 Train a new model
If your training file is emails_500.csv:

bash
复制代码
python spam_detector.py --mode train --csv emails_500.csv --model-path spam_model.pkl
What you’ll see:

Data loading confirmation

Model saved message

A classification report (precision/recall/F1)

Confusion matrix (true vs predicted)

Demo predictions on a few sample strings

Optional training flags:

bash
复制代码
python spam_detector.py --mode train --csv emails_500.csv --model-path spam_model.pkl \
  --test-size 0.2 --random-state 42
--test-size: fraction of data used for testing (default 0.2)

--random-state: seed for reproducibility (default 42)

6.2 Predict interactively (type your own emails)
After training, you should have spam_model.pkl.

Run:

bash
复制代码
python spam_detector.py --mode predict --model-path spam_model.pkl --threshold 0.6
Then type messages like:

ruby
复制代码
Email> Congratulations! You won a free gift card. Click here now.
 => Prediction: spam
    P(spam) = 0.92, P(ham) = 0.08
Exit with:

quit / exit

Ctrl+C

6.3 Batch prediction from a CSV file
Prepare new_emails.csv with a text column:

csv
复制代码
text
"Hello, please review the attached report."
"Win a free gift card now! Click here!"
Run:

bash
复制代码
python spam_detector.py --mode predict --model-path spam_model.pkl \
  --predict-file new_emails.csv --output-csv preds.csv --threshold 0.6
Output file preds.csv will include:

original text

a new column pred_label (spam or ham)

7) Understanding Evaluation Output
7.1 Classification report
Printed by classification_report(y_test, y_pred):

precision: of items predicted as spam, how many were actually spam

recall: of all true spam items, how many were detected

f1-score: harmonic mean of precision and recall

support: number of true samples in that class

Spam filters often care a lot about:

high precision for spam (avoid marking real emails as spam)

or high recall for spam (catch most spam), depending on use case

7.2 Confusion matrix
Printed in a fixed format when both classes exist:

rows: true labels

columns: predicted labels

Example interpretation:

true_spam, pred_ham are false negatives (missed spam)

true_ham, pred_spam are false positives (ham wrongly flagged)

8) How to Choose a Threshold
Default --threshold is 0.6.

Increase threshold (e.g. 0.8):

More conservative

Fewer false positives (ham incorrectly flagged as spam)

More false negatives (spam that slips through)

Decrease threshold (e.g. 0.4):

More aggressive

Fewer false negatives

More false positives

Typical tuning approach:

Evaluate on validation/test

Decide what is worse for your scenario:

false positives (blocking real emails)

false negatives (letting spam through)

Choose threshold accordingly

9) Common Pitfalls
9.1 Labels not matching spam/ham
If your dataset uses something else (e.g. 1/0, yes/no), you have two options:

Convert labels in the CSV to spam/ham

Or modify the code to map your labels to those strings

The thresholding function expects "spam" and "ham" to exist in model.classes_.

9.2 Very small dataset
With only a few dozen samples, performance can be unstable.
Try to increase data size and diversity.

9.3 Domain mismatch
A model trained on one style of emails might struggle on another:

marketing emails vs personal messages

formal office emails vs casual chat-like emails

Add training data that matches your target domain.

10) Extending This Project (Optional Ideas)
If you want to evolve this into a stronger resume-worthy project:

Add a validation split for threshold tuning

Log metrics and persist them (JSON/CSV)

Build a small web API (FastAPI) for prediction

Add a simple UI (CLI is good; a small web UI is even better)

Compare models:

Logistic Regression

Linear SVM

Calibrated probabilities (for better thresholding)

Add preprocessing:

normalize URLs/emails/numbers

remove excessive punctuation

Add unit tests for:

CSV validation

prediction output shape/types

threshold edge cases

11) Quick Start Checklist
Put spam_detector.py and emails_500.csv in the same directory

Train:

bash
复制代码
python spam_detector.py --mode train --csv emails_500.csv --model-path spam_model.pkl
Predict:

bash
复制代码
python spam_detector.py --mode predict --model-path spam_model.pkl --threshold 0.6
Batch predict:

bash
复制代码
python spam_detector.py --mode predict --model-path spam_model.pkl \
  --predict-file new_emails.csv --output-csv preds.csv --threshold 0.6
