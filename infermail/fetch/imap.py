"""IMAP fetch — download emails and persist to DB idempotently."""

from __future__ import annotations

import email as email_lib
from datetime import datetime, timezone
from email.header import decode_header as _decode_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any

from imapclient import IMAPClient
from loguru import logger
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from infermail.db.models import Email, Account


def _decode_str(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        parts = _decode_header(value.decode("utf-8", errors="replace"))
    else:
        parts = _decode_header(value)
    result = []
    for part, charset in parts:
        if isinstance(part, bytes):
            try:
                result.append(part.decode(charset or "utf-8", errors="replace"))
            except (LookupError, TypeError):
                result.append(part.decode("latin-1", errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def _decode_payload(part: email_lib.message.Message) -> str:
    raw = part.get_payload(decode=True)
    if not raw:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return raw.decode("latin-1", errors="replace")


def _parse_body(msg: email_lib.message.Message) -> tuple[str, str]:
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not text:
                text = _decode_payload(part)
            elif ct == "text/html" and not html:
                html = _decode_payload(part)
    else:
        decoded = _decode_payload(msg)
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
    msg = email_lib.message_from_bytes(raw)
    message_id = msg.get("Message-ID", "").strip()
    subject = _decode_str(msg.get("Subject"))
    sender_raw = msg.get("From", "")
    sender_name, sender_addr = parseaddr(sender_raw)
    reply_to = msg.get("Reply-To")
    recipients_raw = msg.get("To", "")
    list_unsubscribe = msg.get("List-Unsubscribe")
    headers: dict[str, str] = {k: str(v) for k, v in msg.items()}
    body_text, body_html = _parse_body(msg)
    has_attachments = any(
        part.get_content_disposition() == "attachment" for part in msg.walk()
    )
    return {
        "message_id": message_id,
        "account_id": account.id,
        "imap_uid": uid,
        "imap_folder": folder,
        "source_folder": folder,
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

            all_uids: list[int] = client.search("ALL")
            if not all_uids:
                logger.info(f"[{account.name}] {folder}: empty")
                continue

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

            if not new_uids:
                logger.info(f"[{account.name}] {folder}: nothing new")
                continue

            logger.info(f"[{account.name}] {folder}: fetching {len(new_uids)} new mails")

            with tqdm(total=len(new_uids), desc=f"{account.name}/{folder}", unit="mail") as pbar:
                for i in range(0, len(new_uids), batch_size):
                    batch = new_uids[i : i + batch_size]
                    try:
                        messages = client.fetch(batch, ["RFC822"])
                    except Exception as e:
                        logger.error(f"[{account.name}] Fetch batch failed: {e}")
                        pbar.update(len(batch))
                        continue

                    rows = []
                    for uid, data in messages.items():
                        raw = data.get(b"RFC822")
                        if not raw:
                            pbar.update(1)
                            continue
                        try:
                            rows.append(_build_email_obj(raw, uid, folder, account))
                        except Exception as e:
                            logger.warning(f"[{account.name}] UID {uid} parse error: {e}")
                        pbar.update(1)

                    if rows:
                        stmt = pg_insert(Email).values(rows).on_conflict_do_nothing(
                            constraint="uq_email_message_account"
                        )
                        result = session.execute(stmt)
                        inserted += max(result.rowcount, 0)
                        session.commit()

    account.last_synced_at = datetime.now(timezone.utc)
    session.commit()
    logger.info(f"[{account.name}] Done — {inserted} new emails inserted")
    return inserted
