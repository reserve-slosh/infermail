"""infermail CLI entrypoint."""

import click
from loguru import logger

from infermail.config import settings
from infermail.db.session import SessionLocal


def _setup_logging() -> None:
    import sys
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)


@click.group()
def main() -> None:
    """infermail — ML-powered email organizer."""
    _setup_logging()


@main.command()
@click.option("--account", "-a", default=None, help="Fetch single account by name.")
@click.option("--dry-run", is_flag=True, help="Parse and log, but don't write to DB.")
def fetch(account: str | None, dry_run: bool) -> None:
    """Fetch emails from IMAP and persist to DB."""
    from infermail.fetch.runner import run_fetch

    if dry_run:
        logger.warning("Dry-run mode — nothing will be written to DB.")

    with SessionLocal() as session:
        run_fetch(session, account_name=account, dry_run=dry_run)


@main.command()
def daemon() -> None:
    """Run fetch loop continuously (used by Docker)."""
    import time
    from datetime import datetime, timezone

    from infermail.backup import run_pg_dump
    from infermail.classify.predictor import Predictor, run_classify
    from infermail.fetch.runner import run_fetch
    from infermail.sync import run_sync
    from infermail.sync.demotion import run_demotion

    predictor = Predictor(settings.model_path)
    logger.info(
        f"Daemon started — fetch interval: {settings.fetch_interval_seconds}s, "
        f"backup interval: {settings.backup_interval_seconds}s, "
        f"keep: {settings.backup_keep_count} dumps"
    )

    last_backup_at: datetime | None = None

    while True:
        now = datetime.now(timezone.utc)

        try:
            with SessionLocal() as session:
                run_fetch(session)
                run_classify(session, predictor)
                run_demotion(session)
                run_sync(session)
        except Exception as e:
            logger.error(f"Fetch/classify cycle failed: {e}")

        elapsed = (now - last_backup_at).total_seconds() if last_backup_at else float("inf")
        if elapsed >= settings.backup_interval_seconds:
            try:
                run_pg_dump(settings.backup_dir, settings.database_url, settings.backup_keep_count)
                last_backup_at = now
            except Exception as e:
                logger.error(f"pg_dump failed: {e}")

        time.sleep(settings.fetch_interval_seconds)


@main.command()
def migrate() -> None:
    """Run Alembic migrations (upgrade head)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    command.upgrade(cfg, "head")
    logger.info("Migrations applied.")


@main.command()
def status() -> None:
    """Show account sync status."""
    from infermail.db.models import Account, Email
    from sqlalchemy import func

    with SessionLocal() as session:
        accounts = session.query(Account).all()
        if not accounts:
            click.echo("No accounts in DB yet.")
            return
        for acc in accounts:
            count = session.query(func.count(Email.id)).filter(
                Email.account_id == acc.id
            ).scalar()
            synced = acc.last_synced_at.strftime("%Y-%m-%d %H:%M") if acc.last_synced_at else "never"
            click.echo(f"{acc.name:20} {acc.email_address:35} emails: {count:>6}  last sync: {synced}")


@main.command()
@click.option("--account", "-a", default=None, help="Nur Mails eines Accounts labeln.")
@click.option("--batch", "-b", default=200, show_default=True, help="Anzahl Mails pro Session.")
def label(account: str | None, batch: int) -> None:
    """Interaktives Labeling-Tool für Training-Daten."""
    from infermail.classify.labeler import run_labeler
    run_labeler(account_name=account, batch=batch)


@main.command()
@click.option("--account", "-a", default=None, help="Sync single account only.")
@click.option("--dry-run", is_flag=True, help="Preview moves without touching IMAP.")
def sync(account: str | None, dry_run: bool) -> None:
    """Move emails in IMAP to match DB classifications."""
    from infermail.sync import run_sync

    with SessionLocal() as session:
        counts = run_sync(session, account_name=account, dry_run=dry_run)
    click.echo(f"moved: {counts['moved']}  skipped: {counts['skipped']}  errors: {counts['errors']}  feedback: {counts['feedback']}")


@main.command()
@click.option("--dir", "backup_dir", default=None, show_default=True, help="Directory to write the backup file (default: settings.backup_dir).")
def backup(backup_dir: str | None) -> None:
    """Dump all emails and classifications to a JSONL file."""
    from pathlib import Path

    from infermail.backup import run_backup

    target = settings.backup_dir if backup_dir is None else Path(backup_dir)
    with SessionLocal() as session:
        out = run_backup(session, target)
    click.echo(f"Backup written to {out}")


@main.command("backup-db")
@click.option("--dir", "backup_dir", default=None, help="Directory to write the .dump file (default: settings.backup_dir).")
@click.option("--keep", default=None, type=int, help="Number of dumps to retain (default: settings.backup_keep_count).")
def backup_db(backup_dir: str | None, keep: int | None) -> None:
    """Create a compressed PostgreSQL dump of the full infermail database."""
    from pathlib import Path

    from infermail.backup import run_pg_dump

    target = settings.backup_dir if backup_dir is None else Path(backup_dir)
    keep_count = settings.backup_keep_count if keep is None else keep
    out = run_pg_dump(target, settings.database_url, keep_count)
    click.echo(f"Database dump written to {out}")


@main.command("add-rule")
@click.option("--domain", default=None, help="Match emails from this sender domain (e.g. gmx.fr).")
@click.option("--regex", default=None, help="Match sender address/name against this regex.")
@click.option("--label", default="spam", show_default=True, help="Label to apply when the rule matches.")
@click.option("--name", default=None, help="Rule name (auto-generated if omitted).")
def add_rule(domain: str | None, regex: str | None, label: str, name: str | None) -> None:
    """Insert a new classification rule into the rules table."""
    from infermail.db.models import Rule

    if not domain and not regex:
        raise click.UsageError("Provide --domain or --regex.")
    if domain and regex:
        raise click.UsageError("--domain and --regex are mutually exclusive.")

    if domain:
        condition = {"type": "sender_domain", "domain": domain}
        auto_name = f"domain:{domain}"
    else:
        condition = {"type": "sender_regex", "pattern": regex, "field": "both"}
        auto_name = f"regex:{regex}"

    rule = Rule(
        name=name or auto_name,
        condition=condition,
        action={"label": label},
    )
    with SessionLocal() as session:
        session.add(rule)
        session.commit()
        session.refresh(rule)
    click.echo(f"Rule #{rule.id} created: {rule.name!r} → {label!r}")


@main.command()
@click.option("--account", "-a", default=None, help="Classify single account only.")
@click.option("--limit", "-n", default=500, show_default=True, help="Max emails to classify per run.")
def classify(account: str | None, limit: int) -> None:
    """Classify unprocessed emails with rules + ML model."""
    from infermail.classify.predictor import Predictor, run_classify

    predictor = Predictor(settings.model_path)
    with SessionLocal() as session:
        counts = run_classify(session, predictor, account_name=account, limit=limit)
    click.echo(f"rule: {counts['rule']}  ml: {counts['ml']}")
