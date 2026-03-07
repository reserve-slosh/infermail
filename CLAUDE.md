# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install / sync deps
uv sync
uv sync --extra dev --extra train   # full local dev (tests + ML libs)

# Run CLI
uv run infermail fetch              # fetch all accounts
uv run infermail fetch -a gmail     # single account
uv run infermail fetch --dry-run    # no DB writes
uv run infermail daemon             # continuous fetch loop (Docker)
uv run infermail status             # account sync stats
uv run infermail label              # interactive TUI labeler
uv run infermail classify           # run rules + ML on unprocessed emails
uv run infermail classify -a gmail  # single account, up to -n 500 emails
uv run infermail sync               # move emails in IMAP to match DB classifications
uv run infermail sync --dry-run     # preview moves without touching IMAP
uv run infermail backup             # dump all emails + classifications to JSONL
uv run infermail migrate            # alembic upgrade head

# Migrations
uv run alembic upgrade head
uv run alembic revision --autogenerate -m "description"

# Tests
uv run pytest
uv run pytest tests/path/test_file.py::test_name   # single test

# Lint / format / typecheck
uv run ruff check .
uv run ruff format .
uv run mypy infermail
```

## Architecture

### Data flow

1. `config/accounts.yml` declares IMAP accounts and which folders to sync. Passwords are **not** stored there — each account has a `password_env` key (e.g. `ACCOUNT_GMAIL_PASSWORD`) that maps to a `Settings` attribute in `infermail/config.py`.
2. `infermail fetch` → `fetch/runner.py:run_fetch()` loads `accounts.yml`, resolves passwords from `settings`, upserts `Account` rows, then calls `fetch/imap.py:fetch_account()`.
3. `fetch_account()` connects via `IMAPClient`, downloads `RFC822`, parses with stdlib `email`, and bulk-upserts into `emails` using PostgreSQL `INSERT … ON CONFLICT DO NOTHING` keyed on `(message_id, account_id)`.
4. `infermail label` launches the interactive TUI (`classify/labeler.py`), writes `EmailClassification` rows with `method=manual`, and can persist domain/regex `Rule` rows for the daemon's future rule engine.

### Key modules

| Path | Role |
|---|---|
| `infermail/config.py` | `Settings` (pydantic-settings); singleton `settings` used everywhere. Reads `.env`. |
| `infermail/db/models.py` | All ORM models: `Account`, `Email`, `EmailClassification`, `Label`, `Rule`, `UnsubscribeLog`. |
| `infermail/db/session.py` | `SessionLocal` (sessionmaker) and `get_session()` generator. |
| `infermail/fetch/imap.py` | Core IMAP logic — connect, search UIDs, fetch RFC822, parse, upsert. |
| `infermail/fetch/runner.py` | Orchestrates across all configured accounts. |
| `infermail/classify/labeler.py` | Full-screen TUI labeler using `rich` + `readchar`. |
| `infermail/classify/predictor.py` | `Predictor` (joblib sklearn pipeline) + `run_classify()` — applies rules then ML to unprocessed emails. |
| `infermail/db/helpers.py` | Shared DB utilities — `get_or_create_label()`. |
| `infermail/sync/__init__.py` | `run_sync()` — moves emails in IMAP to match DB classifications; called by daemon after classify. |
| `infermail/backup/__init__.py` | `run_backup()` — streams all emails + classifications to a timestamped JSONL file. |
| `infermail/cli.py` | Click CLI entry point (`infermail.cli:main`). |
| `migrations/` | Alembic; `env.py` pulls `DATABASE_URL` from `settings`. |
| `config/accounts.yml` | IMAP account definitions (non-secret). |

### Classification methods

`ClassificationMethod` enum has three values that coexist per email:
- `rule` — applied by daemon from `Rule` table (JSONB conditions: `sender_domain`, `sender_regex`)
- `ml` — scikit-learn classifier (loaded from `model_path`, trained separately with `[train]` extras)
- `manual` — set by `infermail label`; these labels are the ground truth for retraining

### Feedback loop (not implemented)

The only source of `method=manual` classifications is `infermail label` (the interactive TUI). `infermail/sync/` and `infermail/backup/` are empty stubs — IMAP folder-move sync and Maildir backup do not exist. Do not assume they work.

## Environment

`.env` must define at minimum:

```
DATABASE_URL=postgresql+psycopg://user:pass@localhost/infermail
ACCOUNT_GMAIL_PASSWORD=...
ACCOUNT_STRATO_PASSWORD=...
ACCOUNT_GMX_PASSWORD=...
```

Database: PostgreSQL 16. Uses JSONB columns (`recipients`, `raw_headers`, `Rule.condition`, `Rule.action`) — no SQLite fallback.
