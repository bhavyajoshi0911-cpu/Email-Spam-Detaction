# SpamShield — Email Spam Detection System
> **"Smarter Email. Safer Inbox."**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.45%25-brightgreen.svg)]()

SpamShield is a complete, production-ready, portfolio-quality web application for automated email and message spam classification. Powered by a supervised machine learning pipeline (**TF-IDF Feature Extraction + Multinomial Naive Bayes**), SpamShield classifies incoming messages into **Spam** or **Not Spam (Ham)** with empirical statistical confidence scores and a modern SaaS interface.

---

## 1. Project Overview

SpamShield addresses the problem of unwanted and malicious message delivery. Rather than relying on simulated logic or hardcoded keyword rules, the system trains on real-world text corpora, fits a Scikit-learn classification pipeline, and serves predictions through an asynchronous Flask REST API accompanied by a responsive analytics dashboard.

### Core Value Highlights
- **Real Supervised ML**: Evaluated strictly on a held-out 20% test split (1,032 unseen messages), achieving **98.45% accuracy** and **98.28% spam precision**.
- **Interactive SaaS Dashboard**: Real-time KPI statistics cards, interactive Chart.js visualizations (corpus distribution & metric performance), and recent audit logs.
- **Dedicated Classification Workspace**: Single-page AJAX message analysis with preset test samples, character/word counting, progress bar probabilities, and risk assessments.
- **Corpus Explorer**: Interactive, searchable, and filterable inspection table of the SMS Spam Collection dataset with pagination.
- **Auditable Prediction History**: SQLite-backed audit log with search, class filtering, privacy-preserving snippet truncation, and clearing capabilities.
- **Beginner-Friendly Architecture**: Modular, maintainable Python code with clear docstrings, reproducible seeds, and zero paid API dependencies.

---

## 2. Technology Stack

### Backend & Machine Learning
- **Language**: Python 3.10+ (Tested on Python 3.14)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/) (loading, schema mapping, deduplication) & [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
  - `TfidfVectorizer` (N-grams 1–2, sublinear term frequency, English stop words)
  - `MultinomialNB` (Laplace additive smoothing $\alpha = 0.1$)
  - `Pipeline` (leak-free text vectorization and classification)
  - `train_test_split` (stratified 80/20 train/test split)
- **Model Serialization**: [Joblib](https://joblib.readthedocs.io/)
- **Web Framework**: [Flask 3.x](https://flask.palletsprojects.com/)
- **Database**: SQLite3 (native, zero-configuration local persistence)
- **Testing**: [Pytest](https://docs.pytest.org/)

### Frontend & UI/UX
- **Templates**: Semantic HTML5 & Jinja2 template inheritance
- **Styling**: Custom CSS3 design system (Inter typography, modern color tokens, flex/grid layouts)
- **Visualizations**: [Chart.js 4.x](https://www.chartjs.org/) (doughnut and grouped bar charts)
- **Icons**: [Bootstrap Icons](https://icons.getbootstrap.com/)
- **Client Logic**: Vanilla ES6+ JavaScript (Asynchronous `fetch` API, zero framework bloat)

---

## 3. Brand Identity & Design System

SpamShield uses a modern security SaaS aesthetic:

| Role | Color Name | Hex Code | Purpose |
| :--- | :--- | :--- | :--- |
| **Primary Navy** | Deep Obsidian | `#14213D` | Sidebar navigation, headers, dark cards |
| **Primary Blue** | Cobalt Blue | `#2563EB` | Active buttons, primary accents, links |
| **Light Blue** | Ice Blue | `#EFF6FF` | Card accents, hover states, info pills |
| **Page Background**| Ghost Gray | `#F5F7FB` | Main workspace background |
| **Card Background**| Pure White | `#FFFFFF` | Content containers and stat cards |
| **Success Green** | Emerald | `#16A34A` | Legitimate (Ham) verdicts, high recall |
| **Spam Red** | Crimson | `#DC2626` | Spam warnings, risk tags, high precision |
| **Accent Purple** | Violet | `#7C3AED` | F1-Score metrics, secondary charts |
| **Main Text** | Charcoal Slate | `#172033` | Headings and high-contrast body text |
| **Borders** | Slate Gray | `#E2E8F0` | Structural dividers and input borders |

---

## 4. Folder Structure

```text
EMAIL SPAM DETECTION SYSTEM/
│
├── app.py                      # Core Flask web application & REST routes
├── train_model.py              # ML pipeline, cleaning, training & evaluation
├── database.py                 # SQLite persistence layer for prediction history
├── download_dataset.py         # Helper utility to download SMS corpus if needed
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation
├── .gitignore                  # Git exclusion rules
│
├── dataset/
│   ├── .gitkeep
│   └── spam.csv                # Benchmark corpus (5,574 raw / 5,160 clean records)
│
├── models/
│   ├── .gitkeep
│   └── spam_detector_model.joblib # Serialized fitted Scikit-learn Pipeline
│
├── artifacts/
│   ├── .gitkeep
│   ├── model_metrics.json      # Evaluation metrics, confusion matrix & metadata
│   └── confusion_matrix.png    # High-resolution matplotlib confusion matrix plot
│
├── templates/
│   ├── base.html               # Persistent navy sidebar, header & layout wrapper
│   ├── dashboard.html          # Dashboard with 4 stat cards, 2 charts & recent logs
│   ├── detect.html             # Classification workspace with preset samples & AJAX results
│   ├── dataset.html            # Dataset explorer, cleaning stats & paginated table
│   ├── performance.html        # Confusion matrix, classification report & guide
│   ├── history.html            # Persistent prediction history with search & filter
│   └── about.html              # Architectural documentation & technical explanations
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern SaaS stylesheet & responsive breakpoints
│   ├── js/
│   │   ├── app.js              # Interactivity, preset loaders & AJAX prediction
│   │   └── charts.js           # Chart.js initialization from /api/stats
│   └── images/
│       └── logo.svg            # Custom vector shield branding logo
│
└── tests/
    └── test_app.py             # Pytest automated test suite (cleaning, ML, routes)
```

---

## 5. Dataset Details & Cleaning Methodology

The application uses the **SMS Spam Collection** benchmark dataset containing 5,574 text messages labeled as `ham` or `spam`.

### Data Cleaning Policies
1. **Fallback Decoding**: Tries UTF-8, Latin-1, CP1252, and ISO-8859-1 to handle disparate text exports.
2. **Column Normalization**: Dynamically detects schema variants (e.g., `v1`/`v2`, `Category`/`Message`, `label`/`message`) and standardizes them.
3. **Empty Message Purging**: Strips whitespace and discards null, empty, or placeholder strings (`NaN`).
4. **Label Harmonization**: Maps variations (`0`, `legit`, `false` $\rightarrow$ `ham`; `1`, `true` $\rightarrow$ `spam`).
5. **Deduplication Policy**: Exact duplicate messages are pruned (keeping first occurrence). This prevents data leakage where identical messages could appear in both train and test partitions.
6. **Stratified Split**: Uses an 80/20 stratified split (`random_state=42`) ensuring the 87.5% / 12.5% natural class proportion is strictly maintained across train and test sets.

---

## 6. Machine Learning Pipeline

The pipeline is implemented using Scikit-learn's `Pipeline` object to ensure zero test-set leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ("classifier", MultinomialNB(alpha=0.1))
])
```

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Transforms variable-length text strings into weighted sparse vectors. Unigrams and bigrams (e.g., `"cash prize"`, `"claim now"`) are extracted. Sublinear term frequency damping reduces the dominance of repeated words.
- **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' Theorem:
  $$P(\text{Spam} \mid \mathbf{x}) \propto P(\text{Spam}) \prod_{i=1}^n P(x_i \mid \text{Spam})$$
  Laplace smoothing ($\alpha = 0.1$) prevents zero-probability bottlenecks on unseen vocabulary.

---

## 7. Model Evaluation Results

Evaluation conducted strictly on the **1,032 held-out test messages**:

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.45%** | Percentage of all test messages classified correctly. |
| **Spam Precision** | **98.28%** | Out of all predicted spam, 98.28% were truly spam (only 2 false alarms). |
| **Spam Recall** | **89.06%** | Out of all true spam, 89.06% were intercepted. |
| **Spam F1-Score** | **93.44%** | Harmonic mean balancing precision and recall. |
| **Ham Precision** | **98.47%** | True legitimate precision rate. |
| **Ham Recall** | **99.78%** | 902 out of 904 legitimate emails reached the inbox. |

### Visual Confusion Matrix (Held-Out Test Set)

```text
                     Predicted HAM     Predicted SPAM
Actual HAM (904):        902 (TN)            2 (FP)     <- 99.8% inbox pass rate
Actual SPAM (128):        14 (FN)          114 (TP)     <- 89.1% catch rate
```

- **True Negatives (TN = 902)**: Benign messages accurately delivered.
- **False Positives (FP = 2)**: Legitimate messages erroneously marked as spam (False Alarms).
- **False Negatives (FN = 14)**: Spam messages that bypassed detection.
- **True Positives (TP = 114)**: Confirmed spam caught and blocked.

---

## 8. Installation & Setup Guide

### Step 1: Clone or Navigate to the Project Folder
```powershell
cd "c:\Users\DELL\OneDrive\Desktop\Python Projects\EMAIL SPAM DETECTION SYSTEM"
```

### Step 2: (Optional) Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: (If needed) Download Dataset
The repository already includes `dataset/spam.csv`. To re-download or update:
```powershell
python download_dataset.py
```

### Step 5: Train and Evaluate the ML Model
```powershell
python train_model.py
```
*Outputs `models/spam_detector_model.joblib`, `artifacts/model_metrics.json`, and `artifacts/confusion_matrix.png`.*

### Step 6: Start the Flask Web Application
```powershell
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 9. Running Automated Tests

Run the test suite using Pytest:
```powershell
pytest tests/test_app.py -v
```
All 7 unit and integration tests validate:
- Cleaning rules, deduplication, and label normalization
- Feature extraction and classification inference
- Flask routing and template rendering
- Real REST API JSON output and error codes (400, 503)
- SQLite database persistence and audit-clearing

---

## 10. How to Test Predictions

1. Navigate to the **Detect Spam** page (`/detect`).
2. Click any of the **Quick Sample Message** pills:
   - **Legitimate Meeting**: Team quarterly sync notification.
   - **Order Confirmation**: Shipping invoice dispatch.
   - **Spam Prize Winner**: High-urgency prize claim.
   - **Suspicious Payment**: Phishing security alert with a mock credential URL.
3. Click **Analyze Message**.
4. The result card animates immediately without page reload, showing:
   - Classification verdict badge (**SPAM** in red or **NOT SPAM** in green)
   - Dual-color probability bar (e.g. 99.8% Spam vs 0.2% Ham)
   - Risk Assessment tier (`High Risk`, `Moderate Risk`, `Low Risk`)
   - Expandable raw text accordion
   - Link to inspect detailed model metrics

---

## 11. Common Errors and Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `Model file not found` / `Model Offline` | Model has not been trained yet. | Run `python train_model.py` in your terminal to generate `models/spam_detector_model.joblib`. |
| `Dataset file not found at dataset/spam.csv` | Missing dataset file. | Run `python download_dataset.py` to fetch the benchmark SMS Spam Collection. |
| `Port 5000 in use` | Another process is occupying port 5000. | Start Flask on an alternative port: `flask run --port 5005` or modify `app.run(port=5005)` in `app.py`. |
| `DeprecationWarning: Setting shape on NumPy array` | Minor warning between Scikit-learn and NumPy 2.5. | Cosmetic deprecation notice; does not affect training or prediction accuracy. |

---

## 12. Security Disclaimers & Real-World Limitations

1. **Statistical Scope**: Machine learning text classification is probabilistic. A prediction does not constitute a certified guarantee of safety or maliciousness.
2. **Domain Boundaries**: Trained on SMS and short messages. Full enterprise emails have distinct characteristics including header metadata (DKIM, SPF, DMARC), MIME structures, and attachments.
3. **Defense in Depth**: Real-world security infrastructure combines statistical NLP with SPF/DKIM validation, IP reputation networks, dynamic link sandboxing, and antivirus scanning.
4. **Privacy**: Never input confidential credentials, sensitive banking keys, or private medical data into demonstration systems.

---

## 13. License

This project is licensed under the **MIT License**. Free for educational, portfolio, and research usage.
