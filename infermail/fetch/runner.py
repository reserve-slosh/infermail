"""Orchestrates fetching across all configured accounts."""

from __future__ import annotations

from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy.orm import Session

from infermail.config import settings
from infermail.db.models import Account
from infermail.fetch.imap import fetch_account

load_dotenv()


def _load_accounts_config() -> list[dict]:
    path = Path("config/accounts.yml")
    with path.open() as f:
        return yaml.safe_load(f)["accounts"]


def _get_or_create_account(session: Session, cfg: dict) -> Account:
    account = session.query(Account).filter_by(email_address=cfg["email_address"]).first()
    if not account:
        account = Account(
            name=cfg["name"],
            email_address=cfg["email_address"],
            imap_host=cfg["imap_host"],
            imap_port=cfg.get("imap_port", 993),
            provider=cfg.get("provider"),
        )
        session.add(account)
        session.commit()
        logger.info(f"Registered new account: {account.email_address}")
    return account


def _get_password(cfg: dict) -> str:
    """Read password from settings object by env var name."""
    env_key = cfg["password_env"].lower()  # e.g. ACCOUNT_GMX_PASSWORD -> account_gmx_password
    return getattr(settings, env_key, "")


def run_fetch(
    session: Session,
    account_name: str | None = None,
    dry_run: bool = False,
) -> None:
    configs = _load_accounts_config()

    if account_name:
        configs = [c for c in configs if c["name"] == account_name]
        if not configs:
            logger.error(f"No account '{account_name}' found in config/accounts.yml")
            return

    for cfg in configs:
        password = _get_password(cfg)
        if not password:
            logger.warning(f"[{cfg['name']}] No password set ({cfg['password_env']}), skipping.")
            continue

        account = _get_or_create_account(session, cfg)
        folders = cfg.get("folders", ["INBOX"])

        if dry_run:
            logger.info(f"[{cfg['name']}] Would fetch folders: {folders}")
            continue

        fetch_account(session, account, password, folders)
