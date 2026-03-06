"""IMAP fetch — download emails and persist to DB idempotently."""

from __future__ import annotations

import email as email_lib
from datetime import datetime, timezone
from email.header import decode_header as _decode_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from imapclient import IMAPClient
from loguru import logger
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from infermail.db.models import Account, Email


def _decode_str(value: str | bytes | None) -> str:
    """Decode encoded email header value to plain string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        parts = _decode_header(value.decode("utf-8", errors="replace"))
    else:
        parts = _decode_header(value)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def _parse_body(msg: email_lib.message.Message) -> tuple[str, str]:
    """Extract (text, html) body from a Message object."""
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not text:
                text = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"
                )
            elif ct == "text/html" and not html:
                html = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"
                )
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            decoded = payload.decode(
                msg.get_content_charset() or "utf-8", errors="replace"
            )
            if msg.get_content_type() == "text/html":
                html = decoded
            else:
                text = decoded
    return text, html


def _parse_received_at(msg: email_lib.message.Message) -> datetime | None:
    date_str = msg.get("Date")
    if not date_str:
        return None
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None


def _build_email_obj(
    raw: bytes,
    uid: int,
    folder: str,
    account: Account,
) -> dict[str, Any]:
    """Parse raw RFC822 bytes into a dict ready for Email model."""
    msg = email_lib.message_from_bytes(raw)

    message_id = msg.get("Message-ID", "").strip()
    subject = _decode_str(msg.get("Subject"))
    sender_raw = msg.get("From", "")
    sender_name, sender_addr = parseaddr(sender_raw)
    reply_to = msg.get("Reply-To")
    recipients_raw = msg.get("To", "")
    list_unsubscribe = msg.get("List-Unsubscribe")

    headers: dict[str, str] = dict(msg.items())
    body_text, body_html = _parse_body(msg)
    has_attachments = any(
        part.get_content_disposition() == "attachment" for part in msg.walk()
    )

    return {
        "message_id": message_id,
        "account_id": account.id,
        "imap_uid": uid,
        "imap_folder": folder,
        "subject": subject,
        "sender": sender_addr or sender_raw,
        "sender_name": _decode_str(sender_name),
        "recipients": [r.strip() for r in recipients_raw.split(",") if r.strip()],
        "reply_to": reply_to,
        "body_text": body_text,
        "body_html": body_html,
        "raw_headers": headers,
        "has_attachments": has_attachments,
        "list_unsubscribe": list_unsubscribe,
        "received_at": _parse_received_at(msg),
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _connect(host: str, port: int, username: str, password: str) -> IMAPClient:
    client = IMAPClient(host, port=port, ssl=True, timeout=30)
    client.login(username, password)
    return client


def fetch_account(
    session: Session,
    account: Account,
    password: str,
    folders: list[str],
    batch_size: int = 100,
) -> int:
    """
    Fetch unseen/new emails for one account and persist to DB.
    Returns number of new emails inserted.
    """
    inserted = 0

    try:
        client = _connect(account.imap_host, account.imap_port, account.email_address, password)
    except Exception as e:
        logger.error(f"[{account.name}] IMAP connection failed: {e}")
        return 0

    with client:
        for folder in folders:
            try:
                client.select_folder(folder, readonly=True)
            except Exception as e:
                logger.warning(f"[{account.name}] Cannot select folder '{folder}': {e}")
                continue

            # Fetch all UIDs — filter already-known via DB
            all_uids: list[int] = client.search("ALL")
            if not all_uids:
                continue

            # Check which UIDs we already have
            existing_uids = {
                row[0]
                for row in session.query(Email.imap_uid)
                .filter(
                    Email.account_id == account.id,
                    Email.imap_folder == folder,
                    Email.imap_uid.in_(all_uids),
                )
                .all()
            }
            new_uids = [u for u in all_uids if u not in existing_uids]

            logger.info(
                f"[{account.name}] {folder}: {len(new_uids)} new of {len(all_uids)} total"
            )

            for i in range(0, len(new_uids), batch_size):
                batch = new_uids[i : i + batch_size]
                try:
                    messages = client.fetch(batch, ["RFC822"])
                except Exception as e:
                    logger.error(f"[{account.name}] Fetch batch failed: {e}")
                    continue

                for uid, data in messages.items():
                    raw = data.get(b"RFC822")
                    if not raw:
                        continue
                    try:
                        obj = _build_email_obj(raw, uid, folder, account)
                        # Idempotency check via message_id + account_id
                        exists = (
                            session.query(Email)
                            .filter_by(
                                message_id=obj["message_id"],
                                account_id=account.id,
                            )
                            .first()
                        )
                        if not exists:
                            session.add(Email(**obj))
                            inserted += 1
                    except Exception as e:
                        logger.warning(f"[{account.name}] UID {uid} parse error: {e}")

                session.commit()

    # Update last_synced_at
    account.last_synced_at = datetime.now(timezone.utc)
    session.commit()

    logger.info(f"[{account.name}] Inserted {inserted} new emails")
    return inserted
