#!/usr/bin/env python3
"""
spam_detector.py

A simple email spam detector using:
  TF-IDF (unigrams + bigrams) -> Multinomial Naive Bayes

Training CSV requirements:
  - text  : email body/content
  - label : "spam" / "ham" (labels are normalized to lowercase)

Examples:
  Train:
    python spam_detector.py --mode train --csv emails.csv --model-path spam_model.pkl

  Predict (interactive):
    python spam_detector.py --mode predict --model-path spam_model.pkl --threshold 0.6

  Predict (batch):
    python spam_detector.py --mode predict --model-path spam_model.pkl \
      --predict-file new_emails.csv --output-csv preds.csv --threshold 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DEFAULT_CSV = "emails.csv"
DEFAULT_MODEL_PATH = "spam_model.pkl"


def load_data(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    """Load a labeled dataset from CSV and normalize text/labels."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Validate required columns
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Normalize data types and label format
    X = df["text"].astype(str)
    y = df["label"].astype(str).str.strip().str.lower()
    return X, y


def build_model(
    *,
    min_df: int = 2,
    ngram_range: Tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
) -> Pipeline:
    """Create a sklearn Pipeline: TF-IDF vectorizer -> Multinomial Naive Bayes."""
    return Pipeline(steps=[
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
        )),
        ("clf", MultinomialNB()),
    ])


def _print_confusion(cm, labels=("ham", "spam")) -> None:
    """Pretty-print a 2x2 confusion matrix (rows=true labels, cols=pred labels)."""
    if cm.shape != (2, 2):
        print("Confusion matrix:")
        print(cm)
        return

    ham, spam = labels
    print("Confusion matrix (rows=true, cols=pred):")
    print(f"            pred_{ham:<4}  pred_{spam:<4}")
    print(f"true_{ham:<4}   {cm[0, 0]:9d} {cm[0, 1]:9d}")
    print(f"true_{spam:<4}  {cm[1, 0]:9d} {cm[1, 1]:9d}")
    print()


def train_and_evaluate(
    *,
    csv_path: Path,
    model_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Pipeline:
    """Train on a train split, evaluate on a test split, then save the trained model."""
    print(f"Loading data from: {csv_path}")
    X, y = load_data(csv_path)

    # Use stratification only when we have at least two classes
    stratify_arg = y if y.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    # Build and train the model
    model = build_model()
    model.fit(X_train, y_train)

    # Save the trained pipeline (vectorizer + classifier)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}\n")

    # Evaluate on the held-out test set
    y_pred = model.predict(X_test)
    print("=== Test Performance ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Prefer a fixed label order for a stable 2x2 confusion matrix when possible
    label_order = ["ham", "spam"]
    present = set(y_test.unique()) | set(y_pred)
    if set(label_order).issubset(present):
        cm = confusion_matrix(y_test, y_pred, labels=label_order)
        _print_confusion(cm, labels=("ham", "spam"))
    else:
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print()

    # Demo predictions for a few example messages
    examples = [
        "Congratulations, you won a free iPhone, click the link now!",
        "Hi Mom, I will call you tonight.",
        "Limited time offer, claim your prize now!!!",
        "Please see the attached report for tomorrow's meeting.",
    ]
    print("=== Demo Predictions ===")
    for text, label in zip(examples, model.predict(examples)):
        print(f"[{label}] {text}")
    print()

    return model


def predict_with_threshold(
    model: Pipeline,
    texts: Sequence[str],
    threshold: float = 0.6
) -> List[Tuple[str, float, float]]:
    """
    Predict using a custom decision threshold on P(spam).

    If P(spam) >= threshold => 'spam', else 'ham'.
    Returns a list of (pred_label, p_spam, p_ham).
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    proba = model.predict_proba(texts)
    classes = list(getattr(model, "classes_", []))
    if not classes:
        raise RuntimeError("Model has no 'classes_' attribute; cannot map probabilities.")

    # Map probability columns to class names
    try:
        spam_idx = classes.index("spam")
        ham_idx = classes.index("ham")
    except ValueError as e:
        raise ValueError(f"Model classes are {classes}, expected to include 'spam' and 'ham'.") from e

    results: List[Tuple[str, float, float]] = []
    for p in proba:
        p_spam = float(p[spam_idx])
        p_ham = float(p[ham_idx])
        pred = "spam" if p_spam >= threshold else "ham"
        results.append((pred, p_spam, p_ham))

    return results


def interactive_loop(model: Pipeline, threshold: float = 0.6) -> None:
    """Run an interactive CLI loop for classifying user-entered email text."""
    print("\n=== Interactive Spam Detector ===")
    print("Type an email text and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print(f"(Current spam threshold: {threshold:.2f})\n")

    has_proba = hasattr(model, "predict_proba")
    if not has_proba:
        print("Warning: model does not support predict_proba(). Probabilities will not be shown.\n")

    while True:
        try:
            user_input = input("Email> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        if has_proba:
            label, p_spam, p_ham = predict_with_threshold(model, [user_input], threshold)[0]
            print(f" => Prediction: {label}")
            print(f"    P(spam) = {p_spam:.2f}, P(ham) = {p_ham:.2f}\n")
        else:
            pred = model.predict([user_input])[0]
            print(f" => Prediction: {pred}\n")


def predict_file(model: Pipeline, input_csv: Path, output_csv: Path, threshold: float = 0.6) -> None:
    """Batch-predict 'text' rows from input_csv and write results to output_csv."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV for prediction must contain a 'text' column.")

    texts = df["text"].astype(str).tolist()
    results = predict_with_threshold(model, texts, threshold)
    df["pred_label"] = [r[0] for r in results]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training/prediction configuration."""
    parser = argparse.ArgumentParser(description="Email Spam Detector (TF-IDF + Naive Bayes)")

    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="Mode: 'train' to train a new model, 'predict' to use an existing model."
    )

    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to training CSV file (used in train mode)."
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to save/load the model."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Spam decision threshold on P(spam) (predict mode)."
    )

    parser.add_argument(
        "--predict-file",
        help="In predict mode: path to a CSV with a 'text' column to classify."
    )

    parser.add_argument(
        "--output-csv",
        help="In predict mode with --predict-file: output CSV path for predictions."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size in train mode (default=0.2)."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default=42)."
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: train a model or load a model for interactive/batch prediction."""
    args = parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model_path)

    if args.mode == "train":
        train_and_evaluate(
            csv_path=csv_path,
            model_path=model_path,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        return

    # Predict mode: load model from disk
    print(f"Loading model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model: Pipeline = joblib.load(model_path)

    # Batch prediction if an input file is provided; otherwise run interactive loop
    if args.predict_file:
        if not args.output_csv:
            raise ValueError("In predict mode with --predict-file, --output-csv must also be provided.")
        predict_file(
            model,
            input_csv=Path(args.predict_file),
            output_csv=Path(args.output_csv),
            threshold=args.threshold,
        )
    else:
        interactive_loop(model, threshold=args.threshold)


if __name__ == "__main__":
    main()