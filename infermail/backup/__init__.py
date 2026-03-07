"""Export all emails and classifications from DB to a JSONL dump."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy.orm import Session, joinedload
from tqdm import tqdm

from infermail.db.models import Email, EmailClassification


def run_backup(session: Session, backup_dir: Path) -> Path:
    """Dump all emails + classifications to a timestamped JSONL file.

    Returns the path of the written file.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = backup_dir / f"infermail_backup_{timestamp}.jsonl"

    total = session.query(Email).count()
    logger.info(f"Backing up {total} emails to {out_path}")

    batch_size = 500
    written = 0

    with out_path.open("w", encoding="utf-8") as fh:
        with tqdm(total=total, desc="backup", unit="mail") as pbar:
            offset = 0
            while True:
                emails = (
                    session.query(Email)
                    .options(
                        joinedload(Email.account),
                        joinedload(Email.classifications).joinedload(
                            EmailClassification.label
                        ),
                    )
                    .order_by(Email.id)
                    .offset(offset)
                    .limit(batch_size)
                    .all()
                )
                if not emails:
                    break

                for email in emails:
                    row = {
                        "id": email.id,
                        "message_id": email.message_id,
                        "account": email.account.name,
                        "account_email": email.account.email_address,
                        "imap_uid": email.imap_uid,
                        "imap_folder": email.imap_folder,
                        "subject": email.subject,
                        "sender": email.sender,
                        "sender_name": email.sender_name,
                        "recipients": email.recipients,
                        "reply_to": email.reply_to,
                        "has_attachments": email.has_attachments,
                        "list_unsubscribe": email.list_unsubscribe,
                        "body_text": email.body_text,
                        "received_at": email.received_at.isoformat() if email.received_at else None,
                        "fetched_at": email.fetched_at.isoformat(),
                        "processed_at": email.processed_at.isoformat() if email.processed_at else None,
                        "classifications": [
                            {
                                "label": c.label.name,
                                "method": c.method.value,
                                "confidence": c.confidence,
                                "model_version": c.model_version,
                                "is_overridden": c.is_overridden,
                                "classified_at": c.classified_at.isoformat(),
                            }
                            for c in email.classifications
                        ],
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

                pbar.update(len(emails))
                offset += batch_size
                session.expire_all()  # free memory between batches

    logger.info(f"Backup complete — {written} emails written to {out_path}")
    return out_path
