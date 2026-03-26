"""Central config — all settings from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    database_url: str

    # Daemon
    fetch_interval_seconds: int = 300
    log_level: str = "INFO"

    # Paths (override in .env for local dev; Docker sets /app/... via env)
    backup_dir: Path = Path("backups")
    model_path: Path = Path("models/classifier.joblib")

    # PostgreSQL dump settings
    backup_keep_count: int = 7          # number of .dump files to retain
    backup_interval_seconds: int = 86400  # how often the daemon runs pg_dump (1 day)

    # Account credentials
    account_gmail_password: str = ""
    account_strato_password: str = ""
    account_gmx_password: str = ""


settings = Settings()
