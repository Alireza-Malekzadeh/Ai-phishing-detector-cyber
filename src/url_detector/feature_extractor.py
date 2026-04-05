"""
URL Feature Extractor
Extracts handcrafted features from a URL for phishing detection.
"""

import re
import urllib.parse
import tldextract
from dataclasses import dataclass, field
from typing import Optional


# Common shortening services
SHORTENING_SERVICES = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly", "short.io"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "update", "secure", "account",
    "banking", "confirm", "password", "credential", "paypal",
    "amazon", "apple", "microsoft", "google", "ebay"
]


@dataclass
class URLFeatures:
    # Length-based
    url_length: int = 0
    domain_length: int = 0
    path_length: int = 0

    # Character-based
    num_dots: int = 0
    num_hyphens: int = 0
    num_underscores: int = 0
    num_slashes: int = 0
    num_at_symbols: int = 0
    num_question_marks: int = 0
    num_ampersands: int = 0
    num_digits_in_url: int = 0
    num_special_chars: int = 0

    # Domain-based
    subdomain_count: int = 0
    has_ip_address: bool = False
    is_shortened: bool = False
    tld_in_path: bool = False

    # Security
    uses_https: bool = False
    has_port: bool = False

    # Suspicion signals
    num_suspicious_keywords: int = 0
    has_double_slash_redirect: bool = False
    num_subdomains: int = 0

    def to_list(self) -> list:
        """Convert features to a flat list for ML models."""
        return [
            self.url_length,
            self.domain_length,
            self.path_length,
            self.num_dots,
            self.num_hyphens,
            self.num_underscores,
            self.num_slashes,
            self.num_at_symbols,
            self.num_question_marks,
            self.num_ampersands,
            self.num_digits_in_url,
            self.num_special_chars,
            self.subdomain_count,
            int(self.has_ip_address),
            int(self.is_shortened),
            int(self.tld_in_path),
            int(self.uses_https),
            int(self.has_port),
            self.num_suspicious_keywords,
            int(self.has_double_slash_redirect),
            self.num_subdomains,
        ]

    @staticmethod
    def feature_names() -> list:
        return [
            "url_length", "domain_length", "path_length",
            "num_dots", "num_hyphens", "num_underscores",
            "num_slashes", "num_at_symbols", "num_question_marks",
            "num_ampersands", "num_digits_in_url", "num_special_chars",
            "subdomain_count", "has_ip_address", "is_shortened",
            "tld_in_path", "uses_https", "has_port",
            "num_suspicious_keywords", "has_double_slash_redirect",
            "num_subdomains",
        ]


def _is_ip_address(hostname: str) -> bool:
    """Check if hostname is an IP address."""
    ip_pattern = re.compile(
        r"^(\d{1,3}\.){3}\d{1,3}$"
    )
    return bool(ip_pattern.match(hostname))


def extract_features(url: str) -> URLFeatures:
    """
    Extract all phishing-relevant features from a URL.

    Args:
        url: Raw URL string (with or without scheme)

    Returns:
        URLFeatures dataclass populated with extracted values
    """
    features = URLFeatures()

    # Normalize: ensure scheme present for parsing
    if not url.startswith(("http://", "https://")):
        url_to_parse = "http://" + url
    else:
        url_to_parse = url

    parsed = urllib.parse.urlparse(url_to_parse)
    extracted = tldextract.extract(url_to_parse)

    # --- Length features ---
    features.url_length = len(url)
    features.domain_length = len(parsed.netloc)
    features.path_length = len(parsed.path)

    # --- Character counts ---
    features.num_dots = url.count(".")
    features.num_hyphens = url.count("-")
    features.num_underscores = url.count("_")
    features.num_slashes = url.count("/")
    features.num_at_symbols = url.count("@")
    features.num_question_marks = url.count("?")
    features.num_ampersands = url.count("&")
    features.num_digits_in_url = sum(c.isdigit() for c in url)
    features.num_special_chars = sum(
        c in "!@#$%^&*()+=[]{}|;:,<>?" for c in url
    )

    # --- Domain features ---
    subdomain = extracted.subdomain
    features.subdomain_count = len(subdomain.split(".")) if subdomain else 0
    features.num_subdomains = features.subdomain_count

    hostname = parsed.hostname or ""
    features.has_ip_address = _is_ip_address(hostname)

    domain_root = extracted.registered_domain
    features.is_shortened = domain_root in SHORTENING_SERVICES

    # Check if a known TLD appears in the path (e.g. /paypal.com/login)
    features.tld_in_path = bool(
        re.search(r"\.(com|org|net|gov|edu|co)", parsed.path)
    )

    # --- Security features ---
    features.uses_https = url_to_parse.startswith("https://")
    features.has_port = bool(parsed.port)

    # --- Suspicion signals ---
    url_lower = url.lower()
    features.num_suspicious_keywords = sum(
        kw in url_lower for kw in SUSPICIOUS_KEYWORDS
    )
    features.has_double_slash_redirect = "//" in parsed.path

    return features


if __name__ == "__main__":
    # Quick sanity check
    test_urls = [
        "https://www.google.com/search?q=hello",
        "http://paypa1-secure-login.verify-account.com/signin",
        "http://192.168.1.1/admin/login.php",
        "https://bit.ly/3xAbc12",
    ]

    print(f"{'URL':<55} | {'HTTPS':<6} | {'IP':<5} | {'Keywords':<8} | {'Len'}")
    print("-" * 90)
    for u in test_urls:
        f = extract_features(u)
        print(
            f"{u[:54]:<55} | {str(f.uses_https):<6} | "
            f"{str(f.has_ip_address):<5} | {f.num_suspicious_keywords:<8} | {f.url_length}"
        )
