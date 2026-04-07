"""
Phishing Detector — Streamlit Demo App
Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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


# ─── URL Feature extraction (111 features matching training data) ─────────────
def extract_url_features_for_model(url: str) -> pd.DataFrame:
    """
    Extract the same 111 features used during training from the
    GregaVrbancic/Phishing-Dataset format.
    We compute what we can and fill the rest with neutral defaults.
    """
    import re
    import urllib.parse
    import tldextract

    if not url.startswith(("http://", "https://")):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    parsed = urllib.parse.urlparse(url_to_parse)
    ext = tldextract.extract(url_to_parse)
    full = url_to_parse

    def count(char, s): return s.count(char)
    def digits(s): return sum(c.isdigit() for c in s)

    path = parsed.path or ""
    domain = parsed.netloc or ""
    params = parsed.query or ""

    features = {
        "qty_dot_url": count(".", full),
        "qty_hyphen_url": count("-", full),
        "qty_underline_url": count("_", full),
        "qty_slash_url": count("/", full),
        "qty_questionmark_url": count("?", full),
        "qty_equal_url": count("=", full),
        "qty_at_url": count("@", full),
        "qty_and_url": count("&", full),
        "qty_exclamation_url": count("!", full),
        "qty_space_url": count(" ", full),
        "qty_tilde_url": count("~", full),
        "qty_comma_url": count(",", full),
        "qty_plus_url": count("+", full),
        "qty_asterisk_url": count("*", full),
        "qty_hashtag_url": count("#", full),
        "qty_dollar_url": count("$", full),
        "qty_percent_url": count("%", full),
        "qty_tld_url": 1 if ext.suffix else 0,
        "length_url": len(full),
        "qty_dot_domain": count(".", domain),
        "qty_hyphen_domain": count("-", domain),
        "qty_underline_domain": count("_", domain),
        "qty_slash_domain": count("/", domain),
        "qty_questionmark_domain": count("?", domain),
        "qty_equal_domain": count("=", domain),
        "qty_at_domain": count("@", domain),
        "qty_and_domain": count("&", domain),
        "qty_exclamation_domain": count("!", domain),
        "qty_space_domain": count(" ", domain),
        "qty_tilde_domain": count("~", domain),
        "qty_comma_domain": count(",", domain),
        "qty_plus_domain": count("+", domain),
        "qty_asterisk_domain": count("*", domain),
        "qty_hashtag_domain": count("#", domain),
        "qty_dollar_domain": count("$", domain),
        "qty_percent_domain": count("%", domain),
        "qty_vowels_domain": sum(c in "aeiouAEIOU" for c in domain),
        "domain_length": len(domain),
        "domain_in_ip": 1 if re.match(r"(\d{1,3}\.){3}\d{1,3}", domain) else 0,
        "server_client_domain": 1 if "server" in domain or "client" in domain else 0,
        "qty_dot_directory": count(".", path),
        "qty_hyphen_directory": count("-", path),
        "qty_underline_directory": count("_", path),
        "qty_slash_directory": count("/", path),
        "qty_questionmark_directory": count("?", path),
        "qty_equal_directory": count("=", path),
        "qty_at_directory": count("@", path),
        "qty_and_directory": count("&", path),
        "qty_exclamation_directory": count("!", path),
        "qty_space_directory": count(" ", path),
        "qty_tilde_directory": count("~", path),
        "qty_comma_directory": count(",", path),
        "qty_plus_directory": count("+", path),
        "qty_asterisk_directory": count("*", path),
        "qty_hashtag_directory": count("#", path),
        "qty_dollar_directory": count("$", path),
        "qty_percent_directory": count("%", path),
        "directory_length": len(path),
        "qty_dot_file": 0,
        "qty_hyphen_file": 0,
        "qty_underline_file": 0,
        "qty_slash_file": 0,
        "qty_questionmark_file": 0,
        "qty_equal_file": 0,
        "qty_at_file": 0,
        "qty_and_file": 0,
        "qty_exclamation_file": 0,
        "qty_space_file": 0,
        "qty_tilde_file": 0,
        "qty_comma_file": 0,
        "qty_plus_file": 0,
        "qty_asterisk_file": 0,
        "qty_hashtag_file": 0,
        "qty_dollar_file": 0,
        "qty_percent_file": 0,
        "file_length": 0,
        "qty_dot_params": count(".", params),
        "qty_hyphen_params": count("-", params),
        "qty_underline_params": count("_", params),
        "qty_slash_params": count("/", params),
        "qty_questionmark_params": count("?", params),
        "qty_equal_params": count("=", params),
        "qty_at_params": count("@", params),
        "qty_and_params": count("&", params),
        "qty_exclamation_params": count("!", params),
        "qty_space_params": count(" ", params),
        "qty_tilde_params": count("~", params),
        "qty_comma_params": count(",", params),
        "qty_plus_params": count("+", params),
        "qty_asterisk_params": count("*", params),
        "qty_hashtag_params": count("#", params),
        "qty_dollar_params": count("$", params),
        "qty_percent_params": count("%", params),
        "params_length": len(params),
        "tld_present_params": 1 if re.search(r"\.(com|org|net|gov)", params) else 0,
        "qty_params": len(params.split("&")) if params else 0,
        "email_in_url": 1 if "@" in full else 0,
        "time_response": 0,
        "domain_spf": 0,
        "asn_ip": 0,
        "time_domain_activation": 0,
        "time_domain_expiration": 0,
        "qty_ip_resolved": 0,
        "qty_nameservers": 0,
        "qty_mx_servers": 0,
        "ttl_hostname": 0,
        "tls_ssl_certificate": 1 if url_to_parse.startswith("https") else 0,
        "qty_redirects": 0,
        "url_google_index": 0,
        "domain_google_index": 0,
        "url_shortened": 1 if any(s in domain for s in ["bit.ly", "tinyurl", "goo.gl", "ow.ly"]) else 0,
    }

    return pd.DataFrame([features])


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
            with st.spinner("Analyzing..."):
                X = extract_url_features_for_model(url_input.strip())

            if "url" in models:
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

                st.markdown("#### Key Signals")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("URL Length", X["length_url"].values[0])
                with col2:
                    st.metric("HTTPS", "Yes ✅" if X["tls_ssl_certificate"].values[0] else "No ⚠️")
                with col3:
                    st.metric("Shortened", "Yes ⚠️" if X["url_shortened"].values[0] else "No ✅")
            else:
                st.warning("URL model not found. Run: python3 -m src.url_detector.trainer")

# ─── Email Tab ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Analyze an Email")
    email_input = st.text_area(
        "Paste raw email content",
        height=250,
        placeholder="From: security@paypa1-alerts.com\nSubject: URGENT!\n\n..."
    )

    if st.button("Analyze Email", type="primary", use_container_width=True):
        if not email_input.strip():
            st.warning("Please paste some email content.")
        else:
            with st.spinner("Analyzing..."):
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

                st.markdown("#### Key Signals")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Urgency Keywords", features.num_urgency_keywords)
                with col2:
                    st.metric("Reply-To Mismatch", "Yes ⚠️" if features.has_reply_to_mismatch else "No ✅")
                with col3:
                    st.metric("IP Links", "Yes ⚠️" if features.has_ip_links else "No ✅")
            else:
                st.warning("Email model not found. Run: python3 -m src.email_detector.trainer")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Master of Cybersecurity — AI Phishing Detector | Built with scikit-learn + Streamlit")