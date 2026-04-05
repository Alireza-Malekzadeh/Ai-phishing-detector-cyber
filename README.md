# 🎣 AI-Based Phishing Detector

> An end-to-end machine learning system for detecting phishing URLs and emails.  
> Built as part of a Master of Cybersecurity portfolio project.

---

## 📌 Overview

This project builds a dual-mode phishing detection system using handcrafted features + classical ML models:

| Mode | Input | Approach |
|---|---|---|
| **URL Detector** | Raw URL string | Feature engineering + Random Forest / XGBoost |
| **Email Detector** | Raw email (headers + body) | NLP + structural features + Gradient Boosting |

Both detectors are served via a **Streamlit demo app** and a **FastAPI REST API**.

---

## 🏗️ Project Structure

```
phishing-detector/
├── data/
│   ├── raw/                   # Downloaded raw datasets
│   ├── processed/             # Cleaned CSVs for training
│   └── download_datasets.py   # Dataset preparation script
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   ├── url_detector/
│   │   ├── feature_extractor.py   # URL feature engineering
│   │   └── trainer.py             # Training pipeline + MLflow
│   ├── email_detector/
│   │   ├── feature_extractor.py   # Email feature engineering
│   │   └── trainer.py             # Training pipeline + MLflow
│   └── utils/
├── models/                    # Saved .joblib models + evaluation plots
├── app/
│   └── streamlit_app.py       # Interactive demo
├── tests/                     # Unit tests
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Alireza-Malekzadeh/ai-cyber-threat-detector.git
cd phishing-detector
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download datasets

```bash
# Emails (SpamAssassin — automatic)
python data/download_datasets.py

# URLs: Download PhishTank CSV from https://www.phishtank.com/developer_info.php
# Then run:
python data/download_datasets.py path/to/phishtank.csv
```

### 3. Train models

```bash
# URL detector
python -m src.url_detector.trainer data/processed/url_dataset.csv

# Email detector
python -m src.email_detector.trainer data/processed/email_dataset.csv
```

### 4. Run the demo app

```bash
streamlit run app/streamlit_app.py
```

### 5. Track experiments

```bash
mlflow ui  # opens at http://localhost:5000
```

---

## 📊 Datasets

| Dataset | Type | Source |
|---|---|---|
| PhishTank | Phishing URLs | [phishtank.com](https://www.phishtank.com/developer_info.php) |
| Alexa Top 1M | Legitimate URLs | S3 public mirror |
| SpamAssassin | Spam + Ham emails | [spamassassin.apache.org](https://spamassassin.apache.org/old/publiccorpus/) |

---

## 🧠 Features

### URL Features (21 total)
- Length-based: URL length, domain length, path length
- Character-based: dots, hyphens, @-symbols, digit count
- Domain-based: IP usage, subdomain depth, shortening services
- Security: HTTPS presence, port usage
- Suspicion: keyword matches, double-slash redirects

### Email Features (18 total)
- Structural: HTML presence, link count, image count
- Header signals: Reply-To mismatch, suspicious subject
- Content signals: urgency keywords, brand impersonation, sensitive terms
- Obfuscation: IP-based links, shortened URLs, excessive caps

---

## 📈 Evaluation

Models are evaluated using:
- **Precision / Recall / F1** per class (not just accuracy — dataset is imbalanced!)
- **ROC-AUC** for ranking ability
- **Confusion matrix** visualization
- **SHAP values** for explainability *(coming in Phase 2)*

---

## 🗺️ Roadmap

- [x] Project structure & feature extractors
- [x] URL detector training pipeline
- [x] Email detector training pipeline  
- [x] Streamlit demo app
- [ ] SHAP explainability layer
- [ ] FastAPI REST endpoint
- [ ] Docker containerization
- [ ] Comparative study (ML vs. fine-tuned BERT) → *article draft*


---

## 👤 Author

**Alireza Malekzadeh**  

