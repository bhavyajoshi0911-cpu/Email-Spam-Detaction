"""
SpamShield - Public Dataset Downloader
======================================
Downloads and prepares the SMS Spam Collection dataset if missing.
"""

import os
import urllib.request
from pathlib import Path
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DATASET_PATH = DATASET_DIR / "spam.csv"
SOURCE_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def download_and_prepare_dataset():
    """Download the benchmark SMS Spam Collection and save as CSV."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    if DATASET_PATH.exists():
        print(f"[OK] Dataset already present at: {DATASET_PATH}")
        return DATASET_PATH

    print(f"Downloading SMS Spam Collection from {SOURCE_URL}...")
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "SpamShield-Downloader/1.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read().decode("utf-8").strip().split("\n")

    records = []
    for line in content:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            records.append({"label": parts[0].strip(), "message": parts[1].strip()})

    df = pd.DataFrame(records)
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    print(f"[SUCCESS] Downloaded and saved {len(df):,} records to {DATASET_PATH}")
    return DATASET_PATH


if __name__ == "__main__":
    download_and_prepare_dataset()
