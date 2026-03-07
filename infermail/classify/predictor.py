"""ML inference and rule-based classification for fetched emails."""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session

from infermail.db.helpers import get_or_create_label
from infermail.db.models import ClassificationMethod, Email, EmailClassification, Rule

SPAM_FOLDERS = {"Spamverdacht", "Junk-E-Mail", "[Gmail]/Spam", "Spam", "spam"}
# Binary model output: 1 = inbox, 0 = not-inbox (spam/newsletter)
_LABEL_MAP = {1: "inbox", 0: "spam"}


class Predictor:
    """Wraps the joblib sklearn pipeline for inference."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._pipeline = None
        self._model_version: str | None = None

    def _load(self) -> bool:
        if self._pipeline is not None:
            return True
        if not self._model_path.exists():
            logger.warning(f"Model not found at {self._model_path} — ML classification skipped.")
            return False
        import joblib
        import json

        self._pipeline = joblib.load(self._model_path)
        meta_path = self._model_path.parent / "meta.json"
        if meta_path.exists():
            self._model_version = json.loads(meta_path.read_text()).get("model_version")
        logger.info(f"Model loaded (version={self._model_version})")
        return True

    def _build_features(self, emails: list[Email]) -> "list[dict]":
        import pandas as pd

        rows = []
        for e in emails:
            text = (
                (e.subject or "") + " "
                + (e.sender or "") + " "
                + (e.sender_name or "") + " "
                + (e.body_text or "")[:2000]
            )
            in_spam = float(e.imap_folder in SPAM_FOLDERS) if e.imap_folder else 0.0
            has_unsub = float(bool(e.list_unsubscribe))
            rows.append({"text": text, "in_spam_folder": in_spam, "has_unsubscribe": has_unsub})
        return rows

    def predict(self, emails: list[Email]) -> list[tuple[str, float]]:
        """Return (label_name, confidence) for each email. Empty list if model unavailable."""
        if not emails or not self._load():
            return []

        import pandas as pd

        X = pd.DataFrame(self._build_features(emails))
        try:
            proba = self._pipeline.predict_proba(X)[:, 1]  # P(inbox)
        except AttributeError:
            # LinearSVC has no predict_proba — use decision_function + sigmoid
            import math
            scores = self._pipeline.decision_function(X)
            proba = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
        results = []
        for p in proba:
            label = _LABEL_MAP[int(p >= 0.5)]
            confidence = float(p) if label == "inbox" else float(1.0 - p)
            results.append((label, round(confidence, 4)))
        return results

    @property
    def model_version(self) -> str | None:
        return self._model_version


def _apply_rules(session: Session, emails: list[Email]) -> int:
    """Apply active Rule rows to emails. Returns count of classifications written."""
    rules = session.query(Rule).filter_by(is_active=True).order_by(Rule.priority.desc()).all()
    if not rules:
        return 0

    written = 0
    for email in emails:
        for rule in rules:
            cond = rule.condition
            matched = False

            if cond.get("type") == "sender_domain":
                domain = cond.get("domain", "")
                matched = bool(email.sender and email.sender.lower().endswith(f"@{domain.lower()}"))

            elif cond.get("type") == "sender_regex":
                pattern = cond.get("pattern", "")
                field = cond.get("field", "both")
                addr = email.sender or ""
                name = email.sender_name or ""
                try:
                    if field == "address":
                        matched = bool(re.search(pattern, addr, re.IGNORECASE))
                    elif field == "name":
                        matched = bool(re.search(pattern, name, re.IGNORECASE))
                    else:
                        matched = bool(
                            re.search(pattern, addr, re.IGNORECASE)
                            or re.search(pattern, name, re.IGNORECASE)
                        )
                except re.error:
                    logger.warning(f"Invalid regex in rule {rule.id}: {pattern!r}")

            if not matched:
                continue

            # Only write if no rule classification exists yet
            existing = (
                session.query(EmailClassification)
                .filter_by(email_id=email.id, method=ClassificationMethod.rule)
                .first()
            )
            if existing:
                break  # highest-priority rule already applied

            label_name = rule.action.get("label", "spam")
            label = get_or_create_label(session, label_name)
            session.add(EmailClassification(
                email_id=email.id,
                label_id=label.id,
                method=ClassificationMethod.rule,
                confidence=1.0,
            ))
            written += 1
            break  # first matching rule wins

    session.commit()
    return written


def run_classify(
    session: Session,
    predictor: Predictor,
    account_name: str | None = None,
    limit: int = 500,
) -> dict[str, int]:
    """Classify unprocessed emails with rules + ML. Returns counts by method."""
    from datetime import datetime, timezone
    from sqlalchemy import exists

    q = (
        session.query(Email)
        .filter(
            ~exists().where(
                EmailClassification.email_id == Email.id,
                EmailClassification.method == ClassificationMethod.ml,
            )
        )
        .order_by(Email.received_at.desc())
    )
    if account_name:
        from infermail.db.models import Account
        q = q.join(Account).filter(Account.name == account_name)

    emails = q.limit(limit).all()

    if not emails:
        logger.info("No unclassified emails found.")
        return {"rule": 0, "ml": 0}

    logger.info(f"Classifying {len(emails)} emails...")

    # Rules first (fast, no model needed)
    rule_count = _apply_rules(session, emails)

    # ML classification
    ml_count = 0
    predictions = predictor.predict(emails)
    if predictions:
        now = datetime.now(timezone.utc)
        for email, (label_name, confidence) in zip(emails, predictions):
            label = get_or_create_label(session, label_name)
            session.add(EmailClassification(
                email_id=email.id,
                label_id=label.id,
                method=ClassificationMethod.ml,
                confidence=confidence,
                model_version=predictor.model_version,
            ))
            email.processed_at = now
            ml_count += 1
        session.commit()

    logger.info(f"Done — rule: {rule_count}, ml: {ml_count}")
    return {"rule": rule_count, "ml": ml_count}
