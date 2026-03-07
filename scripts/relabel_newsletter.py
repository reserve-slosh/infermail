"""One-off script: reclassify spam-labeled emails that are actually newsletters.

Usage:
    uv run python scripts/relabel_newsletter.py [--dry-run]
    uv run python scripts/relabel_newsletter.py --domains [--top 30]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

# ensure project root is importable when run directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infermail.db.models import ClassificationMethod, Email, EmailClassification, Label
from infermail.db.session import SessionLocal


NEWSLETTER_KEYWORDS_BODY = [
    "unsubscribe",
    "abbestell",
    "abmeld",
    "newsletter",
    "mailing list",
    "mailinglist",
    "digest",
]

NEWSLETTER_KEYWORDS_SUBJECT = ["newsletter"]
NEWSLETTER_KEYWORDS_SENDER = ["newsletter", "noreply", "no-reply"]


def _get_or_create_label(session: Session, name: str) -> Label:
    label = session.query(Label).filter_by(name=name).first()
    if not label:
        colors = {"newsletter": "#3b82f6", "spam": "#ef4444", "inbox": "#22c55e", "wichtig": "#eab308"}
        label = Label(name=name, is_system=True, color=colors.get(name, "#94a3b8"))
        session.add(label)
        session.commit()
    return label


def _build_newsletter_filter():
    conditions = [Email.list_unsubscribe.isnot(None)]
    for kw in NEWSLETTER_KEYWORDS_SUBJECT:
        conditions.append(Email.subject.ilike(f"%{kw}%"))
    for kw in NEWSLETTER_KEYWORDS_SENDER:
        conditions.append(Email.sender.ilike(f"%{kw}%"))
        conditions.append(Email.sender_name.ilike(f"%{kw}%"))
    for kw in NEWSLETTER_KEYWORDS_BODY:
        conditions.append(Email.body_text.ilike(f"%{kw}%"))
    return or_(*conditions)


def _extract_domain(sender: str | None) -> str | None:
    if not sender:
        return None
    match = re.search(r"@([\w.-]+)", sender)
    return match.group(1).lower() if match else None


def show_domains(session: Session, top: int) -> None:
    spam_label = session.query(Label).filter_by(name="spam").first()
    if not spam_label:
        print("No 'spam' label found in DB — nothing to do.")
        return

    candidates = (
        session.query(Email.sender)
        .join(EmailClassification, EmailClassification.email_id == Email.id)
        .filter(
            EmailClassification.method == ClassificationMethod.manual,
            EmailClassification.label_id == spam_label.id,
        )
        .filter(_build_newsletter_filter())
        .all()
    )

    counts: Counter[str] = Counter()
    for (sender,) in candidates:
        domain = _extract_domain(sender)
        if domain:
            counts[domain] += 1

    print(f"Top {top} sender domains among {len(candidates)} newsletter candidates:\n")
    width = len(str(counts.most_common(1)[0][1])) if counts else 1
    for i, (domain, count) in enumerate(counts.most_common(top), 1):
        print(f"  {i:>3}.  {count:>{width}}x  {domain}")


def relabel(session: Session, dry_run: bool, whitelist: list[str] | None = None) -> int:
    spam_label = session.query(Label).filter_by(name="spam").first()
    if not spam_label:
        print("No 'spam' label found in DB — nothing to do.")
        return 0

    # All emails manually labeled as spam that match newsletter heuristics
    candidates = (
        session.query(Email)
        .join(EmailClassification, EmailClassification.email_id == Email.id)
        .filter(
            EmailClassification.method == ClassificationMethod.manual,
            EmailClassification.label_id == spam_label.id,
        )
        .filter(_build_newsletter_filter())
        .all()
    )

    if whitelist is not None:
        whitelist_lower = {d.lower() for d in whitelist}
        spam_classified = [
            e for e in candidates
            if _extract_domain(e.sender) in whitelist_lower
        ]
        print(f"Found {len(candidates)} newsletter candidates, {len(spam_classified)} match whitelist domains.")
    else:
        spam_classified = candidates
        print(f"Found {len(spam_classified)} spam-labeled emails matching newsletter heuristics.")

    count = len(spam_classified)

    if dry_run:
        print("\nSample (up to 20):")
        for email in spam_classified[:20]:
            subject = (email.subject or "(no subject)")[:80]
            sender = email.sender or ""
            has_unsub = "unsub" if email.list_unsubscribe else "     "
            print(f"  [{has_unsub}] {sender[:35]:35} | {subject}")
        print("\nDry run — no changes made.")
        return count

    newsletter_label = _get_or_create_label(session, "newsletter")

    for email in spam_classified:
        ec = (
            session.query(EmailClassification)
            .filter_by(email_id=email.id, method=ClassificationMethod.manual)
            .first()
        )
        if ec:
            ec.label_id = newsletter_label.id
            ec.classified_at = datetime.utcnow()

    session.commit()
    print(f"Reclassified {count} emails from spam → newsletter.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify newsletter emails from spam bucket.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB.")
    parser.add_argument("--domains", action="store_true", help="Show top sender domains of newsletter candidates.")
    parser.add_argument("--top", type=int, default=30, metavar="N", help="How many domains to show (default: 30).")
    parser.add_argument("--whitelist", nargs="+", metavar="DOMAIN", help="Only reclassify emails from these domains.")
    args = parser.parse_args()

    with SessionLocal() as session:
        if args.domains:
            show_domains(session, top=args.top)
        else:
            relabel(session, dry_run=args.dry_run, whitelist=args.whitelist)


if __name__ == "__main__":
    main()
