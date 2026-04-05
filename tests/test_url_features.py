"""Tests for URL feature extractor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.url_detector.feature_extractor import extract_features


def test_https_detection():
    f = extract_features("https://google.com")
    assert f.uses_https is True

    f2 = extract_features("http://google.com")
    assert f2.uses_https is False


def test_ip_detection():
    f = extract_features("http://192.168.1.1/login")
    assert f.has_ip_address is True

    f2 = extract_features("https://google.com")
    assert f2.has_ip_address is False


def test_suspicious_keywords():
    f = extract_features("http://verify-paypal-account-login.com/secure")
    assert f.num_suspicious_keywords >= 2


def test_shortened_url():
    f = extract_features("https://bit.ly/abc123")
    assert f.is_shortened is True


def test_feature_vector_length():
    f = extract_features("https://example.com")
    from src.url_detector.feature_extractor import URLFeatures
    assert len(f.to_list()) == len(URLFeatures.feature_names())


def test_url_length():
    url = "http://" + "a" * 100 + ".com"
    f = extract_features(url)
    assert f.url_length == len(url)
