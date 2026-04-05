"""
Email Feature Extractor
Extracts features from raw email content for phishing detection.
"""

import re
import email
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


URGENCY_KEYWORDS = [
    "urgent", "immediately", "account suspended", "verify now",
    "limited time", "act now", "click here", "confirm your",
    "unusual activity", "security alert", "your account will be",
    "expires", "24 hours", "48 hours", "respond immediately",
    "failure to", "violation", "suspended", "blocked", "restricted"
]

SENSITIVE_KEYWORDS = [
    "password", "credit card", "social security", "ssn", "bank account",
    "pin", "cvv", "routing number", "login credentials", "username"
]

TRUSTED_BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "facebook", "instagram", "ebay", "chase", "bank of america",
    "wells fargo", "citibank", "irs", "fedex", "ups", "dhl"
]


@dataclass
class EmailFeatures:
    # Structural
    has_html: bool = False
    num_links: int = 0
    num_images: int = 0
    body_length: int = 0
    num_recipients: int = 0

    # Header signals
    has_reply_to_mismatch: bool = False
    sender_domain_in_body: bool = False
    has_suspicious_subject: bool = False

    # Content signals
    num_urgency_keywords: int = 0
    num_sensitive_keywords: int = 0
    num_brand_impersonation: int = 0
    link_to_text_ratio: float = 0.0
    num_external_links: int = 0
    num_mismatched_links: int = 0  # display text != actual URL

    # Obfuscation signals
    has_ip_links: bool = False
    has_shortened_links: bool = False
    num_exclamation_marks: int = 0
    num_capital_runs: int = 0  # sequences of ALL CAPS words

    def to_list(self) -> list:
        return [
            int(self.has_html),
            self.num_links,
            self.num_images,
            self.body_length,
            self.num_recipients,
            int(self.has_reply_to_mismatch),
            int(self.sender_domain_in_body),
            int(self.has_suspicious_subject),
            self.num_urgency_keywords,
            self.num_sensitive_keywords,
            self.num_brand_impersonation,
            self.link_to_text_ratio,
            self.num_external_links,
            self.num_mismatched_links,
            int(self.has_ip_links),
            int(self.has_shortened_links),
            self.num_exclamation_marks,
            self.num_capital_runs,
        ]

    @staticmethod
    def feature_names() -> list:
        return [
            "has_html", "num_links", "num_images", "body_length",
            "num_recipients", "has_reply_to_mismatch", "sender_domain_in_body",
            "has_suspicious_subject", "num_urgency_keywords", "num_sensitive_keywords",
            "num_brand_impersonation", "link_to_text_ratio", "num_external_links",
            "num_mismatched_links", "has_ip_links", "has_shortened_links",
            "num_exclamation_marks", "num_capital_runs",
        ]


SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly"
}

IP_PATTERN = re.compile(r"https?://(\d{1,3}\.){3}\d{1,3}")
HREF_PATTERN = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
IMG_PATTERN = re.compile(r"<img", re.IGNORECASE)
CAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")


def extract_features(raw_email: str) -> EmailFeatures:
    """
    Extract phishing features from a raw email string.

    Args:
        raw_email: Full raw email content (headers + body)

    Returns:
        EmailFeatures dataclass
    """
    features = EmailFeatures()
    msg = email.message_from_string(raw_email)

    # --- Headers ---
    sender = msg.get("From", "")
    reply_to = msg.get("Reply-To", "")
    subject = msg.get("Subject", "").lower()
    to_field = msg.get("To", "")

    features.num_recipients = len(to_field.split(","))

    # Reply-To mismatch: sender domain != reply-to domain
    def _get_domain(addr: str) -> str:
        match = re.search(r"@([\w.\-]+)", addr)
        return match.group(1).lower() if match else ""

    sender_domain = _get_domain(sender)
    reply_domain = _get_domain(reply_to)
    if reply_to and sender_domain and reply_domain:
        features.has_reply_to_mismatch = sender_domain != reply_domain

    features.has_suspicious_subject = any(
        kw in subject for kw in URGENCY_KEYWORDS
    )

    # --- Body ---
    body = ""
    html_body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    decoded = payload.decode("utf-8", errors="ignore")
                    if ct == "text/plain":
                        body += decoded
                    elif ct == "text/html":
                        html_body += decoded
                        features.has_html = True
            except Exception:
                continue
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="ignore")
            if "<html" in body.lower():
                features.has_html = True
                html_body = body

    full_text = (body + " " + html_body).lower()
    features.body_length = len(full_text)

    # --- Link analysis ---
    links = HREF_PATTERN.findall(html_body)
    features.num_links = len(links)
    features.num_images = len(IMG_PATTERN.findall(html_body))

    word_count = len(full_text.split())
    features.link_to_text_ratio = features.num_links / max(word_count, 1)

    for link in links:
        try:
            parsed = urlparse(link)
            domain = parsed.netloc.lower()

            if IP_PATTERN.match(link):
                features.has_ip_links = True

            if any(s in domain for s in SHORTENING_SERVICES):
                features.has_shortened_links = True

            if parsed.scheme in ("http", "https") and sender_domain:
                if sender_domain not in domain:
                    features.num_external_links += 1
        except Exception:
            continue

    # --- Sender domain in body (legit emails usually match) ---
    if sender_domain and sender_domain in full_text:
        features.sender_domain_in_body = True

    # --- Content signals ---
    features.num_urgency_keywords = sum(kw in full_text for kw in URGENCY_KEYWORDS)
    features.num_sensitive_keywords = sum(kw in full_text for kw in SENSITIVE_KEYWORDS)
    features.num_brand_impersonation = sum(b in full_text for b in TRUSTED_BRANDS)

    # --- Obfuscation signals ---
    features.num_exclamation_marks = full_text.count("!")
    features.num_capital_runs = len(CAPS_PATTERN.findall(body + " " + html_body))

    return features


if __name__ == "__main__":
    sample = """From: security@paypa1-alerts.com
Reply-To: collect@phisher.ru
To: victim@gmail.com
Subject: URGENT: Your account has been suspended!

Dear Customer,

Your PayPal account has been SUSPENDED due to unusual activity.
Click here immediately to verify your credentials: http://192.168.1.100/paypal/login

Failure to verify within 24 hours will result in permanent account closure.

PayPal Security Team
"""
    f = extract_features(sample)
    print("Email Feature Extraction Demo")
    print("=" * 40)
    for name, val in zip(EmailFeatures.feature_names(), f.to_list()):
        print(f"  {name:<30} {val}")
