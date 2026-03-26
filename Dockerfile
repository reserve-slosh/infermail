FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (layer-cached separately from source)
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY infermail/ ./infermail/
COPY migrations/ ./migrations/
COPY config/ ./config/
COPY alembic.ini ./

# Install the project itself (no deps, already installed above)
RUN uv sync --frozen --no-dev

# Runtime mounts (backup and models) are provided by docker-compose volumes
RUN mkdir -p /app/backup /app/models

ENTRYPOINT ["uv", "run", "infermail", "daemon"]
