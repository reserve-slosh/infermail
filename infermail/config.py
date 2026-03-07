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

    # Paths
    backup_dir: Path = Path("/app/backup")
    model_path: Path = Path("/app/models/classifier.joblib")

    # Account credentials
    account_gmail_password: str = ""
    account_strato_password: str = ""
    account_gmx_password: str = ""


settings = Settings()
