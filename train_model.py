"""
SpamShield - Machine Learning Training Pipeline
================================================
This script cleans the SMS/Email dataset, fits a TF-IDF + Multinomial Naive Bayes
classification pipeline, evaluates performance on held-out test data, and saves
the model pipeline and evaluation metrics for deployment.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpamShield.Train")

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_detector_model.joblib"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_IMG = ARTIFACTS_DIR / "confusion_matrix.png"


def load_dataset(filepath=DATASET_PATH):
    """
    Load the dataset CSV supporting multiple encodings and column naming conventions.
    Supported column schemas:
      - 'label' and 'message'
      - 'v1' and 'v2' (standard Kaggle SMS Spam Collection)
      - 'Category' and 'Message'
      - 'type' and 'text'
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found at: {filepath}. Please place 'spam.csv' inside the 'dataset/' folder."
        )

    encodings_to_try = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            logger.info("Successfully loaded dataset with encoding: %s", encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.warning("Failed loading with %s: %s", encoding, e)

    if df is None:
        raise ValueError(f"Could not read {filepath} with any standard encoding (utf-8, latin-1, cp1252).")

    # Column normalization mapping
    col_map = {col.strip().lower(): col for col in df.columns}
    
    # Identify label column
    label_col = None
    for candidate in ["label", "v1", "category", "type", "class", "target"]:
        if candidate in col_map:
            label_col = col_map[candidate]
            break
            
    # Identify message column
    message_col = None
    for candidate in ["message", "v2", "text", "content", "email", "sms"]:
        if candidate in col_map:
            message_col = col_map[candidate]
            break

    if not label_col or not message_col:
        raise ValueError(
            f"Could not identify label and message columns. Available columns: {list(df.columns)}. "
            "Please ensure columns are named 'label'/'v1' and 'message'/'v2'."
        )

    # Standardize column names
    clean_df = pd.DataFrame()
    clean_df["label"] = df[label_col].astype(str)
    clean_df["message"] = df[message_col].astype(str)

    return clean_df


def clean_dataset(df, min_records=50):
    """
    Clean dataset according to documented policies:
      1. Remove null, empty, or whitespace-only message rows.
      2. Normalize labels to 'ham' and 'spam'.
      3. Drop unknown labels.
      4. Remove duplicate messages (keeping first occurrence) to prevent test-set data leakage.
      5. Validate class balance and count sufficiency.
    """
    raw_count = len(df)
    logger.info("Starting data cleaning. Raw record count: %d", raw_count)

    # Step 1: Strip whitespace
    df["message"] = df["message"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Step 2: Remove empty messages
    empty_mask = (df["message"] == "") | (df["message"].str.lower() == "nan")
    missing_count = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # Step 3: Normalize labels
    label_map = {
        "ham": "ham",
        "legit": "ham",
        "legitimate": "ham",
        "0": "ham",
        "false": "ham",
        "spam": "spam",
        "1": "spam",
        "true": "spam"
    }
    df["label"] = df["label"].map(label_map)
    unknown_labels = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).copy()

    # Step 4: Remove duplicate records based on message text
    duplicates_count = int(df.duplicated(subset=["message"]).sum())
    df = df.drop_duplicates(subset=["message"], keep="first").copy()
    
    clean_count = len(df)
    class_counts = df["label"].value_counts().to_dict()
    ham_count = class_counts.get("ham", 0)
    spam_count = class_counts.get("spam", 0)

    logger.info(
        "Data cleaning complete. Clean records: %d (Duplicates removed: %d, Missing/Empty: %d, Unknown labels: %d)",
        clean_count, duplicates_count, missing_count, unknown_labels
    )
    logger.info("Class distribution -> Ham: %d, Spam: %d", ham_count, spam_count)

    # Validation
    if clean_count < min_records:
        raise ValueError(
            f"Dataset is too small for training ({clean_count} records). Minimum required is {min_records} records."
        )
    if min_records >= 50 and (ham_count < 10 or spam_count < 10):
        raise ValueError(
            f"Severe class imbalance or single-class dataset detected (Ham: {ham_count}, Spam: {spam_count}). "
            "Both 'ham' and 'spam' must have at least 10 records."
        )

    stats = {
        "raw_records": raw_count,
        "clean_records": clean_count,
        "duplicates_removed": duplicates_count,
        "missing_removed": missing_count + unknown_labels,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_percentage": round((ham_count / clean_count) * 100, 2),
        "spam_percentage": round((spam_count / clean_count) * 100, 2),
    }

    return df, stats


def build_pipeline():
    """
    Construct the Scikit-learn Pipeline combining TF-IDF feature extraction
    and a Multinomial Naive Bayes classifier.
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.1)
        )
    ])
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evaluate the model on held-out test data and compute metrics.
    Classes are strictly ordered as ['ham', 'spam'].
    """
    classes = ["ham", "spam"]
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Accuracy
    accuracy = float(accuracy_score(y_test, y_pred))

    # Precision, Recall, F1 for each class
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, zero_division=0
    )

    # Macro & Weighted metrics
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Confusion Matrix: [[TN, FP], [FN, TP]] with classes=['ham', 'spam']
    # Row 0: True Ham -> [TN (pred Ham), FP (pred Spam)]
    # Row 1: True Spam -> [FN (pred Ham), TP (pred Spam)]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report_dict = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision_macro": round(float(prec_macro) * 100, 2),
        "recall_macro": round(float(rec_macro) * 100, 2),
        "f1_macro": round(float(f1_macro) * 100, 2),
        "precision_weighted": round(float(prec_weighted) * 100, 2),
        "recall_weighted": round(float(rec_weighted) * 100, 2),
        "f1_weighted": round(float(f1_weighted) * 100, 2),
        "by_class": {
            "ham": {
                "precision": round(float(prec[0]) * 100, 2),
                "recall": round(float(rec[0]) * 100, 2),
                "f1_score": round(float(f1[0]) * 100, 2),
                "support": int(support[0])
            },
            "spam": {
                "precision": round(float(prec[1]) * 100, 2),
                "recall": round(float(rec[1]) * 100, 2),
                "f1_score": round(float(f1[1]) * 100, 2),
                "support": int(support[1])
            }
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": classes,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "description": {
                "TN": "Legitimate messages correctly identified as Ham",
                "FP": "Legitimate messages falsely flagged as Spam (False Alarm)",
                "FN": "Spam messages missed and classified as Ham (Spam Leak)",
                "TP": "Spam messages correctly caught and blocked"
            }
        },
        "classification_report_raw": report_dict
    }

    return metrics, cm


def save_confusion_matrix_plot(cm, save_path=CONFUSION_MATRIX_IMG):
    """
    Generate and save a visual confusion matrix image using matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im)

        classes = ["Ham (Legit)", "Spam"]
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title="SpamShield Confusion Matrix\nHeld-Out Test Set",
            ylabel="Actual Class",
            xlabel="Predicted Class"
        )

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                ax.text(
                    j, i, f"{val:,}\n({val/cm.sum():.1%})",
                    ha="center", va="center",
                    color="white" if val > thresh else "#172033",
                    fontweight="bold"
                )

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix plot saved to %s", save_path)
    except Exception as e:
        logger.warning("Could not generate confusion matrix plot: %s", e)


def train_model(dataset_path=DATASET_PATH, test_size=0.2, random_state=42):
    """
    Orchestrate full training, evaluation, artifact export workflow.
    """
    logger.info("=== Starting SpamShield Model Training Pipeline ===")

    # 1. Load dataset
    raw_df = load_dataset(dataset_path)

    # 2. Clean dataset
    clean_df, stats = clean_dataset(raw_df)

    # 3. Stratified Train / Test split
    X = clean_df["message"]
    y = clean_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        "Train/Test split complete. Training samples: %d, Testing samples: %d",
        len(X_train), len(X_test)
    )

    # 4. Build and fit pipeline
    pipeline = build_pipeline()
    logger.info("Fitting TF-IDF Vectorizer and Multinomial Naive Bayes on training data...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on held-out test data
    logger.info("Evaluating fitted pipeline on held-out test data...")
    metrics, cm = evaluate_model(pipeline, X_test, y_test)

    # 6. Save model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Trained model pipeline successfully saved to: %s", MODEL_PATH)

    # 7. Generate confusion matrix image
    save_confusion_matrix_plot(cm, CONFUSION_MATRIX_IMG)

    # 8. Save evaluation metadata
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    metadata = {
        "model_name": "SpamShield Text Classifier",
        "algorithm": "Multinomial Naive Bayes (alpha=0.1)",
        "vectorizer": "TF-IDF Vectorizer (ngram_range=(1,2), max_features=5000, sublinear_tf=True)",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": random_state,
        "test_size": test_size,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "dataset_stats": stats,
        "metrics": metrics
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model metrics metadata successfully saved to: %s", METRICS_PATH)

    logger.info("=== Training Finished Successfully! ===")
    logger.info("Test Accuracy: %.2f%%", metrics["accuracy"])
    logger.info(
        "Spam - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%",
        metrics["by_class"]["spam"]["precision"],
        metrics["by_class"]["spam"]["recall"],
        metrics["by_class"]["spam"]["f1_score"]
    )
    logger.info(
        "Ham  - Precision: %.2f%%, Recall: %.2f%%, F1: %.2f%%",
        metrics["by_class"]["ham"]["precision"],
        metrics["by_class"]["ham"]["recall"],
        metrics["by_class"]["ham"]["f1_score"]
    )

    return metadata


if __name__ == "__main__":
    train_model()
