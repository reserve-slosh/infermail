# infermail

ML-powered email organizer. Fetches mail from multiple IMAP accounts, classifies with rules + a trained model, and keeps everything backed up on your homeserver.

## Stack
- Python 3.11, uv, SQLAlchemy 2.0, Alembic, PostgreSQL 16
- imapclient, pydantic-settings, loguru
- scikit-learn (training, on workstation)
- Docker Compose (deployment)

## Quickstart

```bash
cp .env.example .env
# fill in credentials

docker compose up -d db
uv run alembic upgrade head
docker compose up -d app
```

## Project Structure

```
infermail/
├── infermail/
│   ├── fetch/      # IMAP download
│   ├── db/         # SQLAlchemy models + session
│   ├── classify/   # Rules + ML inference
│   ├── backup/     # Maildir backup
│   └── sync/       # IMAP flag sync (feedback loop)
├── migrations/     # Alembic
├── config/         # accounts.yml, rules.yml
└── tests/
```

## Feedback Loop

Manual corrections (e.g. marking spam on iPhone) are picked up via IMAP folder sync → written to DB as `method=manual` classifications → used for periodic retraining on workstation.
