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

    from infermail.classify.predictor import Predictor, run_classify
    from infermail.fetch.runner import run_fetch

    predictor = Predictor(settings.model_path)
    logger.info(f"Daemon started — interval: {settings.fetch_interval_seconds}s")
    while True:
        try:
            with SessionLocal() as session:
                run_fetch(session)
                run_classify(session, predictor)
        except Exception as e:
            logger.error(f"Fetch/classify cycle failed: {e}")
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
@click.option("--account", "-a", default=None, help="Classify single account only.")
@click.option("--limit", "-n", default=500, show_default=True, help="Max emails to classify per run.")
def classify(account: str | None, limit: int) -> None:
    """Classify unprocessed emails with rules + ML model."""
    from infermail.classify.predictor import Predictor, run_classify

    predictor = Predictor(settings.model_path)
    with SessionLocal() as session:
        counts = run_classify(session, predictor, account_name=account, limit=limit)
    click.echo(f"rule: {counts['rule']}  ml: {counts['ml']}")
