"""
Dataset Downloader
Downloads and prepares public phishing datasets for this project.

Datasets used:
  - PhishTank (URLs)        : https://www.phishtank.com/developer_info.php
  - Alexa Top 1M (legit URLs): via S3 mirror
  - SpamAssassin (emails)   : https://spamassassin.apache.org/old/publiccorpus/
"""

import os
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── URL Datasets ────────────────────────────────────────────────────────────

ALEXA_URL = "https://s3.amazonaws.com/alexa-static/top-1m.csv.zip"
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"  # requires free account for full


def download_alexa(n_samples: int = 5000):
    """Download Alexa top domains as legitimate URL samples."""
    zip_path = RAW_DIR / "alexa_top1m.zip"
    csv_path = RAW_DIR / "top-1m.csv"

    if not csv_path.exists():
        print("Downloading Alexa Top 1M...")
        urllib.request.urlretrieve(ALEXA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_DIR)
        print(f"  ✓ Saved → {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=["rank", "domain"])
    df = df.head(n_samples)
    df["url"] = "https://" + df["domain"]
    df["label"] = 0  # legitimate
    return df[["url", "label"]]


def load_phishtank_csv(csv_path: str) -> pd.DataFrame:
    """
    Load PhishTank CSV export.
    Download from: https://www.phishtank.com/developer_info.php
    Expected columns: url, phish_detail_url, submission_time, ...
    """
    df = pd.read_csv(csv_path)
    if "url" not in df.columns:
        raise ValueError("Expected 'url' column in PhishTank CSV.")
    df["label"] = 1  # phishing
    return df[["url", "label"]].dropna()


def build_url_dataset(phishtank_csv: str, n_legit: int = 5000, output_name: str = "url_dataset.csv"):
    """
    Combine PhishTank phishing URLs with Alexa legit URLs.
    Saves balanced CSV to data/processed/.
    """
    print("Building URL dataset...")

    phishing_df = load_phishtank_csv(phishtank_csv)
    legit_df = download_alexa(n_legit)

    # Balance classes
    n = min(len(phishing_df), len(legit_df))
    combined = pd.concat([
        phishing_df.sample(n, random_state=42),
        legit_df.sample(n, random_state=42),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = PROCESSED_DIR / output_name
    combined.to_csv(out_path, index=False)
    print(f"✓ URL dataset saved → {out_path} ({len(combined)} rows, balanced)")
    return combined


# ─── Email Datasets ───────────────────────────────────────────────────────────

SPAMASSASSIN_URLS = {
    "spam": "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    "easy_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "hard_ham": "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
}


def download_spamassassin():
    """Download SpamAssassin public corpus."""
    import tarfile

    for name, url in SPAMASSASSIN_URLS.items():
        out_path = RAW_DIR / f"{name}.tar.bz2"
        extract_dir = RAW_DIR / name

        if extract_dir.exists():
            print(f"  ✓ {name} already extracted.")
            continue

        print(f"Downloading SpamAssassin {name}...")
        urllib.request.urlretrieve(url, out_path)

        with tarfile.open(out_path, "r:bz2") as tar:
            tar.extractall(RAW_DIR)
        print(f"  ✓ Extracted → {extract_dir}")


def build_email_dataset(output_name: str = "email_dataset.csv"):
    """
    Read SpamAssassin emails and build a CSV with columns: raw_email, label.
    label=1 for spam/phishing, label=0 for ham.
    """
    import glob

    download_spamassassin()

    records = []

    spam_files = glob.glob(str(RAW_DIR / "spam*/**/*"), recursive=True)
    ham_files = glob.glob(str(RAW_DIR / "*ham*/**/*"), recursive=True)

    for fpath in spam_files:
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    records.append({"raw_email": f.read(), "label": 1})
            except Exception:
                continue

    for fpath in ham_files:
        if os.path.isfile(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    records.append({"raw_email": f.read(), "label": 0})
            except Exception:
                continue

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    out_path = PROCESSED_DIR / output_name
    df.to_csv(out_path, index=False)
    print(f"✓ Email dataset saved → {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    import sys

    print("=== Phishing Detector — Dataset Preparation ===\n")

    # Email dataset (SpamAssassin — no auth needed)
    build_email_dataset()

    # URL dataset — needs PhishTank CSV path as arg
    if len(sys.argv) > 1:
        build_url_dataset(sys.argv[1])
    else:
        print("\nTo build URL dataset, download PhishTank CSV from:")
        print("  https://www.phishtank.com/developer_info.php")
        print("Then run: python -m data.download_datasets <path_to_phishtank.csv>")
