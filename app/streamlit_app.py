"""
Phishing Detector — Streamlit Demo App
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.url_detector.feature_extractor import extract_features as extract_url_features, URLFeatures
from src.email_detector.feature_extractor import extract_features as extract_email_features, EmailFeatures

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Phishing Detector",
    page_icon="🎣",
    layout="centered"
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .result-safe {
        background: linear-gradient(135deg, #1a3a2a, #0f2d1f);
        border: 1px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-phishing {
        background: linear-gradient(135deg, #3a1a1a, #2d0f0f);
        border: 1px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .feature-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    url_path = "models/url_detector_best.joblib"
    email_path = "models/email_detector_best.joblib"

    if os.path.exists(url_path):
        models["url"] = joblib.load(url_path)
    if os.path.exists(email_path):
        models["email"] = joblib.load(email_path)

    return models


models = load_models()

# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🎣 AI Phishing Detector")
st.markdown("*Master's Cybersecurity Project — AI-powered phishing analysis*")
st.divider()

tab1, tab2 = st.tabs(["🔗 URL Detector", "📧 Email Detector"])

# ─── URL Tab ──────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyze a URL")
    url_input = st.text_input(
        "Enter URL",
        placeholder="https://example.com/login",
        label_visibility="collapsed"
    )

    if st.button("Analyze URL", type="primary", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Extracting features..."):
                features = extract_url_features(url_input.strip())
                feature_list = features.to_list()

            # Show prediction if model is available
            if "url" in models:
                X = np.array(feature_list).reshape(1, -1)
                pred = models["url"].predict(X)[0]
                prob = models["url"].predict_proba(X)[0][1]

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-phishing">
                        <h2>🚨 PHISHING DETECTED</h2>
                        <p>Confidence: <strong>{prob*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h2>✅ LIKELY SAFE</h2>
                        <p>Confidence: <strong>{(1-prob)*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⚠️ No trained model found. Train first with `src/url_detector/trainer.py`. Showing features only.")

            # Feature breakdown
            st.markdown("#### Feature Breakdown")
            names = URLFeatures.feature_names()
            cols = st.columns(2)
            for i, (name, val) in enumerate(zip(names, feature_list)):
                with cols[i % 2]:
                    flag = "🔴" if (
                        (name == "has_ip_address" and val) or
                        (name == "uses_https" and not val) or
                        (name == "num_suspicious_keywords" and val > 0) or
                        (name == "is_shortened" and val)
                    ) else "⚪"
                    st.markdown(
                        f'<div class="feature-card">{flag} <b>{name}</b>: {val}</div>',
                        unsafe_allow_html=True
                    )


# ─── Email Tab ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze an Email")
    email_input = st.text_area(
        "Paste raw email content (including headers)",
        height=250,
        placeholder="From: security@paypa1-alerts.com\nSubject: URGENT: Your account is suspended!\n\n..."
    )

    if st.button("Analyze Email", type="primary", use_container_width=True):
        if not email_input.strip():
            st.warning("Please paste some email content.")
        else:
            with st.spinner("Analyzing email..."):
                features = extract_email_features(email_input.strip())
                feature_list = features.to_list()

            if "email" in models:
                X = np.array(feature_list).reshape(1, -1)
                pred = models["email"].predict(X)[0]
                prob = models["email"].predict_proba(X)[0][1]

                if pred == 1:
                    st.markdown(f"""
                    <div class="result-phishing">
                        <h2>🚨 PHISHING EMAIL DETECTED</h2>
                        <p>Confidence: <strong>{prob*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-safe">
                        <h2>✅ LIKELY LEGITIMATE</h2>
                        <p>Confidence: <strong>{(1-prob)*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⚠️ No trained model found. Train first with `src/email_detector/trainer.py`. Showing features only.")

            st.markdown("#### Feature Breakdown")
            names = EmailFeatures.feature_names()
            cols = st.columns(2)
            for i, (name, val) in enumerate(zip(names, feature_list)):
                with cols[i % 2]:
                    flag = "🔴" if (
                        (name == "has_reply_to_mismatch" and val) or
                        (name == "num_urgency_keywords" and val > 1) or
                        (name == "has_ip_links" and val) or
                        (name == "has_shortened_links" and val)
                    ) else "⚪"
                    st.markdown(
                        f'<div class="feature-card">{flag} <b>{name}</b>: {val}</div>',
                        unsafe_allow_html=True
                    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Master of Cybersecurity — AI Phishing Detector Project | Built with scikit-learn + Streamlit")
