"""
SpamShield - Flask Application
==============================
Production-grade Web Application for Email & Message Spam Detection using Scikit-Learn.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import joblib
import pandas as pd

import database

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SpamShield.App")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_detector_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "model_metrics.json"
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"

# Flask initialization
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "spamshield-secure-dev-key-982341278")

# Initialize database
database.init_db()

# Global model cache
_cached_model = None
_cached_metrics = None
_cached_dataset_df = None


def get_model():
    """Load and cache the trained ML pipeline safely."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(MODEL_PATH):
        logger.warning("Model file not found at: %s", MODEL_PATH)
        return None

    try:
        _cached_model = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded ML model pipeline from: %s", MODEL_PATH)
        return _cached_model
    except Exception as e:
        logger.error("Error loading model: %s", e)
        return None


def get_metrics():
    """Load cached model evaluation metrics."""
    global _cached_metrics
    if _cached_metrics is not None:
        return _cached_metrics

    if not os.path.exists(METRICS_PATH):
        return None

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            _cached_metrics = json.load(f)
        return _cached_metrics
    except Exception as e:
        logger.error("Error loading metrics: %s", e)
        return None


def get_dataset_df():
    """Load and cache cleaned dataset for the explorer."""
    global _cached_dataset_df
    if _cached_dataset_df is not None:
        return _cached_dataset_df

    if not os.path.exists(DATASET_PATH):
        return None

    try:
        from train_model import load_dataset, clean_dataset
        raw_df = load_dataset(DATASET_PATH)
        clean_df, _ = clean_dataset(raw_df)
        _cached_dataset_df = clean_df
        return _cached_dataset_df
    except Exception as e:
        logger.error("Error loading dataset DataFrame: %s", e)
        return None


# ---------------------------------------------------------------------------
# Template Context Helpers
# ---------------------------------------------------------------------------
@app.context_processor
def inject_global_data():
    """Make basic status and branding universally available to templates."""
    model_loaded = os.path.exists(MODEL_PATH)
    metrics = get_metrics()
    return {
        "model_loaded": model_loaded,
        "metrics_available": metrics is not None,
        "now_year": datetime.now().year,
        "product_name": "SpamShield",
        "tagline": "Smarter Email. Safer Inbox."
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    """Dashboard Landing Page: Key Metrics, Charts, Recent Activity."""
    metrics = get_metrics()
    recent_predictions = database.get_recent_predictions(limit=5)
    
    # Dataset statistics
    if metrics and "dataset_stats" in metrics:
        dataset_stats = metrics["dataset_stats"]
    else:
        dataset_stats = {
            "clean_records": "N/A",
            "spam_count": "N/A",
            "ham_count": "N/A",
            "duplicates_removed": 0
        }

    # Model accuracy
    if metrics and "metrics" in metrics:
        accuracy = metrics["metrics"].get("accuracy", "N/A")
    else:
        accuracy = "N/A"

    return render_template(
        "dashboard.html",
        active_page="dashboard",
        dataset_stats=dataset_stats,
        accuracy=accuracy,
        metrics=metrics,
        recent_predictions=recent_predictions
    )


@app.route("/detect")
def detect():
    """Email Spam Classification Workspace."""
    return render_template("detect.html", active_page="detect")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST API Endpoint to classify email or message.
    Accepts JSON: {"subject": "...", "message": "..."}
    Returns JSON with classification, probabilities, and confidence.
    """
    data = request.get_json(silent=True)
    if not data:
        # Fallback to form data if submitted via traditional form
        data = request.form

    subject = (data.get("subject") or "").strip()
    message = (data.get("message") or "").strip()

    # Input validation
    if not message:
        return jsonify({
            "success": False,
            "error": "Message content is required. Please enter or paste an email message to analyze."
        }), 400

    if len(message) > 50000:
        return jsonify({
            "success": False,
            "error": "Message exceeds the maximum allowable length (50,000 characters)."
        }), 400

    # Combine subject and message consistently
    if subject:
        combined_text = f"Subject: {subject}\n\n{message}"
    else:
        combined_text = message

    # Load model
    model = get_model()
    if model is None:
        return jsonify({
            "success": False,
            "error": "Machine learning model pipeline is not available. Please train the model first by running 'python train_model.py'."
        }), 503

    try:
        # Prediction
        prediction_raw = model.predict([combined_text])[0]
        prediction = str(prediction_raw).lower()
        
        # Probabilities
        probabilities = {}
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([combined_text])[0]
            classes = list(model.classes_)
            for cls_name, prob_val in zip(classes, probs):
                probabilities[str(cls_name).lower()] = round(float(prob_val) * 100, 2)
            
            spam_prob = probabilities.get("spam", 0.0) / 100.0
            ham_prob = probabilities.get("ham", 0.0) / 100.0
            confidence = round(max(spam_prob, ham_prob) * 100, 2)
        else:
            spam_prob = 1.0 if prediction == "spam" else 0.0
            ham_prob = 1.0 if prediction == "ham" else 0.0
            confidence = 100.0
            probabilities = {
                "spam": spam_prob * 100,
                "ham": ham_prob * 100
            }

        # Label and Risk Assessment
        if prediction == "spam":
            label = "SPAM"
            risk_level = "High Risk" if spam_prob >= 0.8 else "Moderate Risk"
        else:
            label = "NOT SPAM"
            risk_level = "Low Risk" if ham_prob >= 0.8 else "Needs Review"

        char_count = len(combined_text)
        word_count = len(combined_text.split())

        # Save to SQLite Prediction History
        pred_id = database.save_prediction(
            subject=subject,
            message_preview=message[:200],
            char_count=char_count,
            word_count=word_count,
            prediction=prediction,
            label=label,
            confidence=confidence,
            spam_prob=spam_prob,
            ham_prob=ham_prob,
            risk_level=risk_level
        )

        return jsonify({
            "success": True,
            "id": pred_id,
            "prediction": prediction,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "risk_level": risk_level,
            "char_count": char_count,
            "word_count": word_count,
            "analyzed_text": combined_text,
            "subject": subject,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        return jsonify({
            "success": False,
            "error": f"An internal error occurred while processing the prediction: {str(e)}"
        }), 500


@app.route("/dataset")
def dataset():
    """Dataset Explorer Page with statistics, charts, and record table."""
    metrics = get_metrics()
    df = get_dataset_df()

    if metrics and "dataset_stats" in metrics:
        stats = metrics["dataset_stats"]
    elif df is not None:
        ham_c = int((df["label"] == "ham").sum())
        spam_c = int((df["label"] == "spam").sum())
        total_c = len(df)
        stats = {
            "raw_records": total_c,
            "clean_records": total_c,
            "duplicates_removed": 0,
            "missing_removed": 0,
            "ham_count": ham_c,
            "spam_count": spam_c,
            "ham_percentage": round((ham_c / total_c) * 100, 2) if total_c else 0,
            "spam_percentage": round((spam_c / total_c) * 100, 2) if total_c else 0,
        }
    else:
        stats = None

    # Initial preview records
    preview_records = []
    if df is not None:
        preview_records = df.head(50).to_dict(orient="records")

    return render_template(
        "dataset.html",
        active_page="dataset",
        stats=stats,
        preview_records=preview_records,
        total_in_df=len(df) if df is not None else 0
    )


@app.route("/api/dataset/records")
def api_dataset_records():
    """API endpoint to support searchable, filterable, paginated records in dataset explorer."""
    df = get_dataset_df()
    if df is None:
        return jsonify({"records": [], "total": 0, "page": 1, "total_pages": 0})

    search = request.args.get("search", "").strip().lower()
    label_filter = request.args.get("label", "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(10, int(request.args.get("per_page", 20))))

    filtered = df
    if label_filter in ["ham", "spam"]:
        filtered = filtered[filtered["label"] == label_filter]

    if search:
        filtered = filtered[filtered["message"].str.lower().str.contains(search, na=False)]

    total = len(filtered)
    total_pages = max(1, (total + per_page - 1) // per_page)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_records = filtered.iloc[start_idx:end_idx].to_dict(orient="records")
    return jsonify({
        "records": page_records,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })


@app.route("/performance")
def performance():
    """Model Performance Analytics & Confusion Matrix."""
    metrics = get_metrics()
    return render_template(
        "performance.html",
        active_page="performance",
        metrics=metrics
    )


@app.route("/history")
def history():
    """Prediction History Page with search and filtering."""
    search = request.args.get("q", "").strip()
    filter_class = request.args.get("filter", "").strip()

    predictions = database.get_all_predictions(search=search, filter_class=filter_class)
    return render_template(
        "history.html",
        active_page="history",
        predictions=predictions,
        search_query=search,
        current_filter=filter_class
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clear all records in prediction history."""
    database.clear_prediction_history()
    flash("Prediction history has been successfully cleared.", "success")
    return redirect(url_for("history"))


@app.route("/about")
def about():
    """About Project Page with technical overview and documentation."""
    metrics = get_metrics()
    return render_template("about.html", active_page="about", metrics=metrics)


@app.route("/api/stats")
def api_stats():
    """JSON API endpoint returning real computed metrics for dashboard charts."""
    metrics = get_metrics()
    if not metrics:
        return jsonify({"available": False, "error": "Metrics not yet generated."}), 404

    dataset_stats = metrics.get("dataset_stats", {})
    model_metrics = metrics.get("metrics", {})

    return jsonify({
        "available": True,
        "class_distribution": {
            "labels": ["Legitimate (Ham)", "Spam"],
            "counts": [
                dataset_stats.get("ham_count", 0),
                dataset_stats.get("spam_count", 0)
            ],
            "percentages": [
                dataset_stats.get("ham_percentage", 0),
                dataset_stats.get("spam_percentage", 0)
            ]
        },
        "model_performance": {
            "classes": ["Ham (Legitimate)", "Spam", "Macro Average"],
            "precision": [
                model_metrics.get("by_class", {}).get("ham", {}).get("precision", 0),
                model_metrics.get("by_class", {}).get("spam", {}).get("precision", 0),
                model_metrics.get("precision_macro", 0)
            ],
            "recall": [
                model_metrics.get("by_class", {}).get("ham", {}).get("recall", 0),
                model_metrics.get("by_class", {}).get("spam", {}).get("recall", 0),
                model_metrics.get("recall_macro", 0)
            ],
            "f1_score": [
                model_metrics.get("by_class", {}).get("ham", {}).get("f1_score", 0),
                model_metrics.get("by_class", {}).get("spam", {}).get("f1_score", 0),
                model_metrics.get("f1_macro", 0)
            ],
            "accuracy": model_metrics.get("accuracy", 0)
        },
        "confusion_matrix": model_metrics.get("confusion_matrix", {})
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
