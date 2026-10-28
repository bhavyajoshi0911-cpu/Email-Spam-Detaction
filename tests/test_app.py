"""
SpamShield - Automated Test Suite
=================================
Validates data cleaning, ML model inference, Flask routes, API endpoints,
and database operations.
"""

import os
import json
import pytest
import pandas as pd
from pathlib import Path

# Adjust path to import from root
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from train_model import clean_dataset, build_pipeline
import database


@pytest.fixture
def client():
    """Create Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_data_cleaning_policy():
    """Verify data cleaning properly strips, maps, and deduplicates."""
    sample_data = pd.DataFrame([
        {"label": "ham", "message": "Hey are you coming tomorrow?"},
        {"label": "ham", "message": "Hey are you coming tomorrow?"},  # duplicate
        {"label": "spam", "message": "WINNER! Free tickets call now!"},
        {"label": "1", "message": "Claim free cash instantly!"},      # label 1 -> spam
        {"label": "0", "message": "Let us meet at the library"},      # label 0 -> ham
        {"label": "ham", "message": "   "},                           # empty message
        {"label": "spam", "message": "nan"},                          # nan message
    ])

    cleaned_df, stats = clean_dataset(sample_data, min_records=2)

    assert stats["duplicates_removed"] == 1
    assert stats["missing_removed"] == 2
    assert len(cleaned_df) == 4
    assert set(cleaned_df["label"].unique()) == {"ham", "spam"}
    # Check deduplication kept unique messages only
    assert cleaned_df["message"].duplicated().sum() == 0


def test_model_pipeline_inference():
    """Test building pipeline, fitting on mini dataset, and predicting."""
    X_train = [
        "Are we still on for lunch tomorrow?",
        "Please review the attached project specs.",
        "URGENT: Win $10,000 cash prize claim now!",
        "Free entry call 08000928018 to receive award!"
    ]
    y_train = ["ham", "ham", "spam", "spam"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    ham_pred = pipeline.predict(["Can you send the quarterly report?"])[0]
    spam_pred = pipeline.predict(["WINNER! Claim free cash reward now!"])[0]

    assert ham_pred in ["ham", "spam"]
    assert spam_pred in ["ham", "spam"]
    assert hasattr(pipeline, "predict_proba")


def test_flask_navigation_routes(client):
    """Test all top-level web pages load successfully with 200 OK."""
    routes = ["/", "/detect", "/dataset", "/performance", "/history", "/about"]
    for route in routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} returned status {response.status_code}"


def test_api_stats_endpoint(client):
    """Test /api/stats returns valid JSON with class distribution and model performance."""
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert data["available"] is True
    assert "class_distribution" in data
    assert "model_performance" in data
    assert len(data["class_distribution"]["labels"]) == 2


def test_predict_endpoint_validation(client):
    """Test /predict input validation rules."""
    # Test 1: Empty message rejected
    res_empty = client.post("/predict", json={"subject": "Test", "message": ""})
    assert res_empty.status_code == 400
    assert res_empty.get_json()["success"] is False

    # Test 2: Whitespace-only message rejected
    res_space = client.post("/predict", json={"subject": "Test", "message": "   \n\t  "})
    assert res_space.status_code == 400
    assert res_space.get_json()["success"] is False


def test_predict_endpoint_classifications(client):
    """Test classifying legitimate and spam sample messages via /predict."""
    # Test Legitimate (Ham)
    legit_payload = {
        "subject": "Status Meeting Notes",
        "message": "Hi Sarah, thank you for organizing the presentation. The team will follow up on Tuesday morning."
    }
    res_legit = client.post("/predict", json=legit_payload)
    assert res_legit.status_code == 200
    legit_data = res_legit.get_json()
    assert legit_data["success"] is True
    assert legit_data["prediction"] in ["ham", "spam"]
    assert "confidence" in legit_data
    assert "probabilities" in legit_data

    # Test Spam
    spam_payload = {
        "subject": "CONGRATULATIONS PRIZE CLAIM",
        "message": "URGENT! You have won a 1000 cash award! Call 09061701461 to claim your free reward immediately!"
    }
    res_spam = client.post("/predict", json=spam_payload)
    assert res_spam.status_code == 200
    spam_data = res_spam.get_json()
    assert spam_data["success"] is True
    assert spam_data["prediction"] == "spam"
    assert spam_data["label"] == "SPAM"


def test_prediction_history_db():
    """Verify SQLite database persists and clears predictions."""
    database.init_db()

    # Save dummy prediction
    pred_id = database.save_prediction(
        subject="Test Alert",
        message_preview="Checking database logging",
        char_count=50,
        word_count=8,
        prediction="ham",
        label="NOT SPAM",
        confidence=95.5,
        spam_prob=0.045,
        ham_prob=0.955,
        risk_level="Low Risk"
    )
    assert pred_id is not None

    # Retrieve
    records = database.get_recent_predictions(limit=5)
    assert len(records) > 0
    assert any(r["id"] == pred_id for r in records)

    # Clear history
    database.clear_prediction_history()
    records_after = database.get_recent_predictions(limit=5)
    assert len(records_after) == 0
