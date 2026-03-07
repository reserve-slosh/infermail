"""Tests for rule engine and ML predictor logic — no DB or IMAP required."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from infermail.db.models import ClassificationMethod


# ---------------------------------------------------------------------------
# Helpers to build lightweight fake ORM objects
# ---------------------------------------------------------------------------

def _make_email(**kwargs) -> MagicMock:
    defaults = dict(
        id=1,
        subject="Test subject",
        sender="sender@example.com",
        sender_name="Sender Name",
        body_text="Hello world",
        imap_folder="INBOX",
        list_unsubscribe=None,
        classifications=[],
    )
    defaults.update(kwargs)
    email = MagicMock()
    for k, v in defaults.items():
        setattr(email, k, v)
    return email


def _make_classification(label_name: str, method: ClassificationMethod, confidence: float = 1.0) -> MagicMock:
    c = MagicMock()
    c.method = method
    c.confidence = confidence
    c.label = MagicMock()
    c.label.name = label_name
    return c


# ---------------------------------------------------------------------------
# Rule engine tests
# ---------------------------------------------------------------------------

class TestApplyRules:
    def _run(self, email, rules):
        """Run _apply_rules logic in isolation (no DB)."""
        import re
        results = []
        for rule in sorted(rules, key=lambda r: r["priority"], reverse=True):
            cond = rule["condition"]
            matched = False
            if cond.get("type") == "sender_domain":
                domain = cond.get("domain", "")
                matched = bool(email.sender and email.sender.lower().endswith(f"@{domain.lower()}"))
            elif cond.get("type") == "sender_regex":
                pattern = cond.get("pattern", "")
                field = cond.get("field", "both")
                addr = email.sender or ""
                name = email.sender_name or ""
                if field == "address":
                    matched = bool(re.search(pattern, addr, re.IGNORECASE))
                elif field == "name":
                    matched = bool(re.search(pattern, name, re.IGNORECASE))
                else:
                    matched = bool(re.search(pattern, addr, re.IGNORECASE) or re.search(pattern, name, re.IGNORECASE))
            if matched:
                results.append(rule["action"]["label"])
                break  # first match wins
        return results[0] if results else None

    def test_sender_domain_match(self):
        email = _make_email(sender="news@newsletter.com")
        rules = [{"priority": 10, "condition": {"type": "sender_domain", "domain": "newsletter.com"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) == "spam"

    def test_sender_domain_no_match(self):
        email = _make_email(sender="real@trusted.com")
        rules = [{"priority": 10, "condition": {"type": "sender_domain", "domain": "newsletter.com"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) is None

    def test_sender_domain_case_insensitive(self):
        email = _make_email(sender="news@NEWSLETTER.COM")
        rules = [{"priority": 10, "condition": {"type": "sender_domain", "domain": "newsletter.com"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) == "spam"

    def test_sender_regex_address(self):
        email = _make_email(sender="noreply+promo@shop.de", sender_name="Shop")
        rules = [{"priority": 10, "condition": {"type": "sender_regex", "pattern": r"noreply\+promo", "field": "address"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) == "spam"

    def test_sender_regex_name(self):
        email = _make_email(sender="info@legit.com", sender_name="Marketing Bot")
        rules = [{"priority": 10, "condition": {"type": "sender_regex", "pattern": r"marketing", "field": "name"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) == "spam"

    def test_sender_regex_both_fields(self):
        email = _make_email(sender="info@legit.com", sender_name="Promo Deals")
        rules = [{"priority": 10, "condition": {"type": "sender_regex", "pattern": r"promo", "field": "both"}, "action": {"label": "spam"}}]
        assert self._run(email, rules) == "spam"

    def test_higher_priority_rule_wins(self):
        email = _make_email(sender="news@example.com")
        rules = [
            {"priority": 5,  "condition": {"type": "sender_domain", "domain": "example.com"}, "action": {"label": "inbox"}},
            {"priority": 10, "condition": {"type": "sender_domain", "domain": "example.com"}, "action": {"label": "spam"}},
        ]
        assert self._run(email, rules) == "spam"

    def test_no_rules(self):
        email = _make_email(sender="anyone@example.com")
        assert self._run(email, []) is None


# ---------------------------------------------------------------------------
# Effective label priority tests (used by sync)
# ---------------------------------------------------------------------------

class TestEffectiveLabel:
    def _effective(self, classifications):
        from infermail.sync import _effective_label
        email = _make_email(classifications=classifications)
        return _effective_label(email)

    def test_manual_beats_ml(self):
        cs = [
            _make_classification("spam", ClassificationMethod.ml),
            _make_classification("inbox", ClassificationMethod.manual),
        ]
        assert self._effective(cs) == "inbox"

    def test_manual_beats_rule(self):
        cs = [
            _make_classification("spam", ClassificationMethod.rule),
            _make_classification("inbox", ClassificationMethod.manual),
        ]
        assert self._effective(cs) == "inbox"

    def test_rule_beats_ml(self):
        cs = [
            _make_classification("inbox", ClassificationMethod.ml),
            _make_classification("spam", ClassificationMethod.rule),
        ]
        assert self._effective(cs) == "spam"

    def test_no_classifications(self):
        assert self._effective([]) is None


# ---------------------------------------------------------------------------
# Predictor feature building
# ---------------------------------------------------------------------------

class TestPredictorFeatures:
    def _build(self, **kwargs):
        from infermail.classify.predictor import Predictor
        p = Predictor(Path("nonexistent.joblib"))
        email = _make_email(**kwargs)
        return p._build_features([email])[0]

    def test_spam_folder_flag(self):
        row = self._build(imap_folder="[Gmail]/Spam")
        assert row["in_spam_folder"] == 1.0

    def test_inbox_not_spam_folder(self):
        row = self._build(imap_folder="INBOX")
        assert row["in_spam_folder"] == 0.0

    def test_has_unsubscribe(self):
        row = self._build(list_unsubscribe="<mailto:unsub@list.com>")
        assert row["has_unsubscribe"] == 1.0

    def test_no_unsubscribe(self):
        row = self._build(list_unsubscribe=None)
        assert row["has_unsubscribe"] == 0.0

    def test_text_concatenation(self):
        row = self._build(subject="Hello", sender="a@b.com", sender_name="Alice", body_text="Body content")
        assert "Hello" in row["text"]
        assert "a@b.com" in row["text"]
        assert "Alice" in row["text"]
        assert "Body content" in row["text"]

    def test_body_truncated_at_2000(self):
        # Use a character that won't appear in default subject/sender/sender_name
        row = self._build(body_text="ж" * 3000, subject="", sender="a@b.com", sender_name="")
        assert row["text"].count("ж") == 2000


# ---------------------------------------------------------------------------
# Predictor.predict — integration test with real model
# ---------------------------------------------------------------------------

class TestPredictorIntegration:
    MODEL_PATH = Path("models/classifier.joblib")

    @pytest.mark.skipif(
        not Path("models/classifier.joblib").exists(),
        reason="model file not present",
    )
    def test_predict_returns_results(self):
        from infermail.classify.predictor import Predictor

        p = Predictor(self.MODEL_PATH)
        emails = [
            _make_email(subject="Invoice #123", sender="billing@company.com", body_text="Please find attached your invoice."),
            _make_email(subject="50% OFF TODAY ONLY", sender="promo@deals.com", body_text="Click here to claim your discount!", list_unsubscribe="<mailto:unsub@deals.com>"),
        ]
        results = p.predict(emails)
        assert len(results) == 2
        for label, confidence in results:
            assert label in ("inbox", "spam")
            assert 0.0 <= confidence <= 1.0

    @pytest.mark.skipif(
        not Path("models/classifier.joblib").exists(),
        reason="model file not present",
    )
    def test_obvious_spam_classified_correctly(self):
        from infermail.classify.predictor import Predictor

        p = Predictor(self.MODEL_PATH)
        spam_email = _make_email(
            subject="YOU WON A PRIZE!!!",
            sender="winner@promo-deals-xyz.com",
            body_text="Congratulations! Click here to claim $1000 now!!!",
            imap_folder="[Gmail]/Spam",
            list_unsubscribe="<mailto:unsub@promo.com>",
        )
        results = p.predict([spam_email])
        label, confidence = results[0]
        assert label == "spam"
        assert confidence > 0.7
