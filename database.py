"""
SpamShield - Database Layer (SQLite)
====================================
Provides persistent storage for prediction history and privacy-preserving logs.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent / "predictions.db"


def get_db_connection():
    """Create and return a database connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                subject TEXT,
                message_preview TEXT NOT NULL,
                char_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                prediction TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                spam_probability REAL NOT NULL,
                ham_probability REAL NOT NULL,
                risk_level TEXT NOT NULL
            );
        """)
        conn.commit()


def save_prediction(subject, message_preview, char_count, word_count,
                    prediction, label, confidence, spam_prob, ham_prob, risk_level):
    """
    Save a prediction record into the SQLite database.
    Privacy note: message_preview stores only the first 250 characters.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions (
                timestamp, subject, message_preview, char_count, word_count,
                prediction, label, confidence, spam_probability, ham_probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            subject or "",
            message_preview[:250],
            char_count,
            word_count,
            prediction,
            label,
            round(confidence, 2),
            round(spam_prob * 100, 2),
            round(ham_prob * 100, 2),
            risk_level
        ))
        conn.commit()
        return cursor.lastrowid


def get_recent_predictions(limit=10):
    """Retrieve the most recent predictions for the dashboard."""
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT id, timestamp, subject, message_preview, char_count, word_count,
                   prediction, label, confidence, spam_probability, ham_probability, risk_level
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_all_predictions(search=None, filter_class=None):
    """Retrieve filtered prediction history."""
    query = """
        SELECT id, timestamp, subject, message_preview, char_count, word_count,
               prediction, label, confidence, spam_probability, ham_probability, risk_level
        FROM predictions
        WHERE 1=1
    """
    params = []
    if search:
        query += " AND (message_preview LIKE ? OR subject LIKE ?)"
        term = f"%{search.strip()}%"
        params.extend([term, term])
    if filter_class and filter_class.lower() in ["spam", "ham"]:
        query += " AND prediction = ?"
        params.append(filter_class.lower())

    query += " ORDER BY id DESC"

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def clear_prediction_history():
    """Clear all records from prediction history."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()
