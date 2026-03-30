"""Move inbox emails that ML-classified as spam into the demoted folder."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from loguru import logger
from sqlalchemy.orm import Session

from infermail.db.models import Account, ClassificationMethod, Email, EmailClassification, Label
from infermail.fetch.runner import _get_password, _load_accounts_config
from infermail.sync import _BATCH, _connect, _ensure_folder, _move_batch

_DEFAULT_DEMOTED_FOLDER = "infermail/Demoted"


def _demote_account(
    session: Session,
    account: Account,
    password: str,
    inbox_folders: list[str],
    demoted_folder: str,
) -> dict[str, int]:
    counts: dict[str, int] = {"moved": 0, "skipped": 0, "errors": 0}

    emails = (
        session.query(Email)
        .join(
            EmailClassification,
            (EmailClassification.email_id == Email.id)
            & (EmailClassification.method == ClassificationMethod.ml),
        )
        .join(Label, Label.id == EmailClassification.label_id)
        .filter(
            Email.account_id == account.id,
            Email.source_folder.in_(inbox_folders),
            Email.demoted_to_spam_at.is_(None),
            Email.imap_uid.is_not(None),
            Label.name == "spam",
        )
        .all()
    )

    if not emails:
        return counts

    logger.info(f"[{account.name}] {len(emails)} inbox email(s) to demote → {demoted_folder!r}")

    # Group by current imap_folder so we select the right IMAP mailbox per batch
    by_folder: dict[str, list[Email]] = defaultdict(list)
    for email in emails:
        if email.imap_folder:
            by_folder[email.imap_folder].append(email)
        else:
            counts["skipped"] += 1

    try:
        client = _connect(account.imap_host, account.imap_port, account.email_address, password)
    except Exception as e:
        logger.error(f"[{account.name}] IMAP connection failed: {e}")
        counts["errors"] += len(emails)
        return counts

    with client:
        try:
            _ensure_folder(client, demoted_folder)
        except Exception as e:
            logger.error(f"[{account.name}] Cannot ensure demoted folder {demoted_folder!r}: {e}")
            counts["errors"] += len(emails)
            return counts

        for source_folder, folder_emails in by_folder.items():
            try:
                client.select_folder(source_folder, readonly=False)
            except Exception as e:
                logger.warning(
                    f"[{account.name}] Cannot select {source_folder!r}: {e} "
                    f"({len(folder_emails)} skipped)"
                )
                counts["errors"] += len(folder_emails)
                continue

            all_uids = [e.imap_uid for e in folder_emails]  # type: ignore[misc]
            now = datetime.now(timezone.utc)

            for i in range(0, len(all_uids), _BATCH):
                batch_uids = all_uids[i : i + _BATCH]
                batch_emails = folder_emails[i : i + _BATCH]
                try:
                    _move_batch(client, batch_uids, demoted_folder)
                    for em in batch_emails:
                        em.imap_folder = demoted_folder
                        em.imap_uid = None
                        em.demoted_to_spam_at = now
                    session.commit()
                    counts["moved"] += len(batch_uids)
                    logger.info(
                        f"[{account.name}] Demoted {len(batch_uids)} from {source_folder!r}"
                    )
                except Exception as e:
                    logger.error(
                        f"[{account.name}] Demotion batch failed ({source_folder!r}): {e}"
                    )
                    session.rollback()
                    counts["errors"] += len(batch_uids)

    return counts


def run_demotion(
    session: Session,
    account_name: str | None = None,
) -> dict[str, int]:
    """Demote inbox emails with ml=spam classification to each account's demoted_folder."""
    configs = _load_accounts_config()
    if account_name:
        configs = [c for c in configs if c["name"] == account_name]
        if not configs:
            logger.error(f"No account '{account_name}' found in config/accounts.yml")
            return {"moved": 0, "skipped": 0, "errors": 0}

    totals: dict[str, int] = {"moved": 0, "skipped": 0, "errors": 0}

    for cfg in configs:
        inbox_folders: list[str] = cfg.get("inbox_folders", cfg.get("folders", ["INBOX"]))
        demoted_folder: str = cfg.get("demoted_folder", _DEFAULT_DEMOTED_FOLDER)

        password = _get_password(cfg)
        if not password:
            logger.warning(f"[{cfg['name']}] No password set, skipping.")
            continue

        account = session.query(Account).filter_by(email_address=cfg["email_address"]).first()
        if not account:
            logger.warning(f"[{cfg['name']}] Not in DB yet — run fetch first.")
            continue

        counts = _demote_account(session, account, password, inbox_folders, demoted_folder)
        for k, v in counts.items():
            totals[k] += v
        logger.info(
            f"[{cfg['name']}] demotion — moved={counts['moved']} "
            f"skipped={counts['skipped']} errors={counts['errors']}"
        )

    return totals
