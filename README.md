# Spam Detector (TF-IDF + Naive Bayes)

A simple email spam classifier written in Python 3 using scikit-learn.

Model pipeline:
- TF-IDF Vectorizer (unigrams + bigrams)
- Multinomial Naive Bayes

Features:
- Train + evaluate with a train/test split
- Print classification report + confusion matrix
- Save/load the entire model pipeline as a .pkl file
- Interactive CLI prediction
- Batch prediction from a CSV file
- Custom decision threshold based on P(spam)

---

## Files

Suggested structure:

├── spam_detector.py   
├── emails_500.csv          # training data (example)  
└── spam_model.pkl          # created after training  

---

## Requirements

Python:
- Python 3.9+ recommended (3.8 often works)

Packages:
- pandas
- joblib
- scikit-learn

Install:

pip install pandas joblib scikit-learn

---

## Training CSV Format

Your training CSV must contain two columns:

- text  : email content
- label : spam or ham

Example:

text,label
"Hi, are we still meeting tomorrow?",ham
"Limited time offer! Claim your prize now!",spam

Notes:
- Labels are normalized to lowercase in the script.
- Text is cast to string.

---

## Usage

### 1) Train

Train a new model and save it:

python spam_detector.py --mode train --csv emails_500.csv --model-path spam_model.pkl

Optional flags:

python spam_detector.py --mode train --csv emails_500.csv --model-path spam_model.pkl --test-size 0.2 --random-state 42

What it does:
- Splits the dataset into train/test
- Trains TF-IDF + Naive Bayes
- Saves the trained pipeline to spam_model.pkl
- Prints evaluation metrics and a confusion matrix
- Prints a few demo predictions

---

### 2) Predict (Interactive)

Load a trained model and type text to classify:

python spam_detector.py --mode predict --model-path spam_model.pkl --threshold 0.6

Exit:
- Type quit or exit
- Or press Ctrl+C

---

### 3) Predict (Batch)

Input CSV must have a text column:

text
"Hello, please review the attached report."
"Win a free gift card now! Click here!"

Run:

python spam_detector.py --mode predict --model-path spam_model.pkl --predict-file new_emails.csv --output-csv preds.csv --threshold 0.6

Output:
- preds.csv will include a new column: pred_label

---

## Threshold Decision Rule

The script uses predict_proba() and applies a threshold on P(spam):

- If P(spam) >= threshold -> predict spam
- Else -> predict ham

Default threshold is 0.6.
- Higher threshold: fewer false positives (more conservative)
- Lower threshold: fewer false negatives (more aggressive)

---

## Notes

- The thresholding code expects the model classes to include the strings "spam" and "ham".
- If your dataset uses labels like 0/1, convert them to spam/ham (or adjust the script).

---

## License

Free to use for learning and personal projects.
