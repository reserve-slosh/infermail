"""Move emails in IMAP to folders matching their DB classifications."""

from __future__ import annotations

from collections import defaultdict

from imapclient import IMAPClient
from loguru import logger
from sqlalchemy.orm import Session, joinedload
from tenacity import retry, stop_after_attempt, wait_exponential

from infermail.db.models import Account, ClassificationMethod, Email, EmailClassification
from infermail.fetch.runner import _get_password, _load_accounts_config

_METHOD_PRIORITY = {
    ClassificationMethod.manual: 3,
    ClassificationMethod.rule: 2,
    ClassificationMethod.ml: 1,
}

_BATCH = 100


def _effective_label(email: Email) -> str | None:
    """Return the winning label name: manual > rule > ml."""
    if not email.classifications:
        return None
    best = max(email.classifications, key=lambda c: _METHOD_PRIORITY.get(c.method, 0))
    return best.label.name


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _connect(host: str, port: int, username: str, password: str) -> IMAPClient:
    client = IMAPClient(host, port=port, ssl=True, timeout=30)
    client.login(username, password)
    return client


def _ensure_folder(client: IMAPClient, folder: str) -> None:
    if not client.folder_exists(folder):
        client.create_folder(folder)
        logger.info(f"Created IMAP folder: {folder!r}")


def _move_batch(client: IMAPClient, uids: list[int], target_folder: str) -> None:
    """COPY uids to target then expunge from currently selected folder."""
    client.copy(uids, target_folder)
    client.set_flags(uids, [b"\\Deleted"])
    try:
        client.expunge(uids)  # UIDPLUS — expunges only these UIDs
    except Exception:
        logger.warning("UIDPLUS not available, falling back to full EXPUNGE")
        client.expunge()


def _sync_account(
    session: Session,
    account: Account,
    password: str,
    label_folders: dict[str, str],
    dry_run: bool,
) -> dict[str, int]:
    counts: dict[str, int] = {"moved": 0, "skipped": 0, "errors": 0}

    emails = (
        session.query(Email)
        .options(
            joinedload(Email.classifications).joinedload(EmailClassification.label)
        )
        .filter(
            Email.account_id == account.id,
            Email.imap_uid.is_not(None),
            Email.imap_folder.is_not(None),
        )
        .all()
    )

    # Determine which emails need moving
    moves: list[tuple[Email, str, str]] = []
    for email in emails:
        label = _effective_label(email)
        if not label:
            counts["skipped"] += 1
            continue
        target = label_folders.get(label)
        if not target:
            counts["skipped"] += 1
            continue
        if email.imap_folder == target:
            counts["skipped"] += 1
            continue
        moves.append((email, email.imap_folder, target))

    if not moves:
        logger.info(f"[{account.name}] Nothing to move.")
        return counts

    logger.info(f"[{account.name}] {len(moves)} emails to move (dry_run={dry_run})")

    if dry_run:
        summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for _, src, tgt in moves:
            summary[src][tgt] += 1
        for src, targets in summary.items():
            for tgt, n in targets.items():
                logger.info(f"  {src!r} → {tgt!r}: {n} emails")
        counts["moved"] += len(moves)
        return counts

    try:
        client = _connect(account.imap_host, account.imap_port, account.email_address, password)
    except Exception as e:
        logger.error(f"[{account.name}] IMAP connection failed: {e}")
        counts["errors"] += len(moves)
        return counts

    with client:
        # Ensure all target folders exist before starting
        for folder in {tgt for _, _, tgt in moves}:
            try:
                _ensure_folder(client, folder)
            except Exception as e:
                logger.error(f"[{account.name}] Cannot ensure folder {folder!r}: {e}")

        # Group: source_folder → target_folder → [(uid, email)]
        groups: dict[str, dict[str, list[tuple[int, Email]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for email, src, tgt in moves:
            groups[src][tgt].append((email.imap_uid, email))  # type: ignore[arg-type]

        for source_folder, targets in groups.items():
            try:
                client.select_folder(source_folder, readonly=False)
            except Exception as e:
                n = sum(len(v) for v in targets.values())
                logger.warning(f"[{account.name}] Cannot select {source_folder!r}: {e} ({n} skipped)")
                counts["errors"] += n
                continue

            for target_folder, uid_email_pairs in targets.items():
                all_uids = [uid for uid, _ in uid_email_pairs]
                all_emails = [em for _, em in uid_email_pairs]

                for i in range(0, len(all_uids), _BATCH):
                    batch_uids = all_uids[i : i + _BATCH]
                    batch_emails = all_emails[i : i + _BATCH]
                    try:
                        _move_batch(client, batch_uids, target_folder)
                        for em in batch_emails:
                            em.imap_folder = target_folder
                            em.imap_uid = None
                        session.commit()
                        counts["moved"] += len(batch_uids)
                        logger.info(
                            f"[{account.name}] {source_folder!r} → {target_folder!r}: "
                            f"moved {len(batch_uids)}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[{account.name}] Move batch failed "
                            f"({source_folder!r} → {target_folder!r}): {e}"
                        )
                        session.rollback()
                        counts["errors"] += len(batch_uids)

    return counts


def run_sync(
    session: Session,
    account_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Sync DB classifications to IMAP folder structure for all (or one) account."""
    configs = _load_accounts_config()
    if account_name:
        configs = [c for c in configs if c["name"] == account_name]
        if not configs:
            logger.error(f"No account '{account_name}' found in config/accounts.yml")
            return {"moved": 0, "skipped": 0, "errors": 0}

    totals: dict[str, int] = {"moved": 0, "skipped": 0, "errors": 0}

    for cfg in configs:
        label_folders: dict[str, str] = cfg.get("label_folders", {})
        if not label_folders:
            logger.warning(f"[{cfg['name']}] No label_folders configured, skipping.")
            continue

        password = _get_password(cfg)
        if not password:
            logger.warning(f"[{cfg['name']}] No password set, skipping.")
            continue

        account = session.query(Account).filter_by(email_address=cfg["email_address"]).first()
        if not account:
            logger.warning(f"[{cfg['name']}] Not in DB yet — run fetch first.")
            continue

        counts = _sync_account(session, account, password, label_folders, dry_run)
        for k, v in counts.items():
            totals[k] += v
        logger.info(
            f"[{cfg['name']}] moved={counts['moved']} "
            f"skipped={counts['skipped']} errors={counts['errors']}"
        )

    return totals
