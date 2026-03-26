"""Backup utilities: JSONL email export and PostgreSQL binary dump."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

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


# ---------------------------------------------------------------------------
# PostgreSQL binary dump
# ---------------------------------------------------------------------------

def run_pg_dump(backup_dir: Path, database_url: str, keep_count: int = 7) -> Path:
    """Create a compressed PostgreSQL binary dump (.dump) of the database.

    Uses ``pg_dump -Fc`` (custom format), which is directly restorable with
    ``pg_restore``.  Connection parameters are parsed from *database_url*
    (supports the ``postgresql+psycopg://`` and plain ``postgresql://`` schemes).

    Existing ``.dump`` files in *backup_dir* that match the naming pattern are
    pruned so that only the most recent *keep_count* files are retained.

    Returns the path of the newly written dump file.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Strip SQLAlchemy driver suffix so urlparse handles the URL correctly.
    clean_url = database_url.split("://", 1)
    clean_url = "postgresql://" + clean_url[1] if len(clean_url) == 2 else database_url
    parsed = urlparse(clean_url)

    host = parsed.hostname or "localhost"
    port = str(parsed.port or 5432)
    user = parsed.username or "postgres"
    password = parsed.password or ""
    dbname = (parsed.path or "/postgres").lstrip("/")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = backup_dir / f"infermail_{timestamp}.dump"

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    cmd = [
        "pg_dump",
        "-Fc",          # custom compressed format
        "-h", host,
        "-p", port,
        "-U", user,
        "-d", dbname,
        "-f", str(out_path),
    ]

    logger.info(f"pg_dump starting → {out_path}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dump failed (exit {result.returncode}): {result.stderr.strip()}")

    size_mb = out_path.stat().st_size / 1_048_576
    logger.info(f"pg_dump complete: {out_path} ({size_mb:.1f} MB)")

    _prune_dumps(backup_dir, keep_count)
    return out_path


def _prune_dumps(backup_dir: Path, keep_count: int) -> None:
    """Delete oldest ``infermail_*.dump`` files, retaining the last *keep_count*."""
    dumps = sorted(
        backup_dir.glob("infermail_*.dump"),
        key=lambda p: p.stat().st_mtime,
    )
    to_delete = dumps[:-keep_count] if len(dumps) > keep_count else []
    for p in to_delete:
        p.unlink()
        logger.info(f"Pruned old dump: {p}")
