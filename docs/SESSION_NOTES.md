# Session Notes

## What Is Implemented and Working

### Fetch pipeline
`infermail/fetch/imap.py` + `fetch/runner.py` are complete and production-ready.
- Connects via `IMAPClient` with TLS, retries via `tenacity` (3 attempts, exponential backoff)
- Parses RFC822: subject/sender decode, multipart body walk, `List-Unsubscribe` extraction, attachment detection
- Bulk-upserts via PostgreSQL `INSERT … ON CONFLICT DO NOTHING` keyed on `(message_id, account_id)` — idempotent
- Tracks `last_synced_at` per account
- `config/accounts.yml` defines 3 accounts (gmail, strato, gmx) with their folders and `password_env` keys

### Database
`infermail/db/models.py` — all six tables fully defined with proper indexes, constraints, and JSONB columns:
- `accounts`, `labels`, `emails`, `email_classifications`, `rules`, `unsubscribe_log`
- `ClassificationMethod` enum: `rule | ml | manual`
- Single Alembic migration (`migrations/versions/54c55c7f095f_initial.py`) covers everything — schema is stable

### CLI
`infermail/cli.py` — six commands: `fetch`, `daemon`, `migrate`, `status`, `label`, `classify`
- `daemon` loops fetch + classify with `fetch_interval_seconds` (default 300s)
- `classify` runs rules then ML inference, writes `EmailClassification` rows

### Interactive labeler
`infermail/classify/labeler.py` — full-screen TUI using `rich` + `readchar`
- Keys: inbox/newsletter/spam/wichtig/skip/back/quit
- `[g]` domain bulk-spam: labels all unlabeled mails from a domain and saves a `Rule` row
- `[r]` regex rule: matches sender address/name, labels matching mails, saves a `Rule` row
- Back navigation with label removal works correctly

### Classify / inference
`infermail/classify/predictor.py`:
- `Predictor` class lazily loads joblib pipeline; gracefully no-ops if model file missing
- `_apply_rules()` reads active `Rule` rows from DB, applies `sender_domain` and `sender_regex` conditions
- `run_classify()` queries emails with no `method=ml` classification, applies rules first, then ML

### Training script
`scripts/train.py` — full pipeline:
- Loads `method=manual` labels from DB
- Feature engineering: `text` (subject + sender + sender_name + body[:2000]), `in_spam_folder`, `has_unsubscribe`
- Binary target: spam/newsletter → 0, inbox → 1 (wichtig silently dropped via `dropna()`)
- `--benchmark` flag: 10-fold stratified CV across 5 models (logreg, linearsvc, sgd, lgbm, voting), picks winner
- Default (no flag): trains LinearSVC directly, no CV overhead
- Saves `models/classifier.joblib` (latest) + `models/classifier_{version}.joblib` (archive)
- Writes `models/meta.json` and `models/RESULTS.md` on success
- Guards with `F1_THRESHOLD = 0.85`

### Newsletter relabeling
`scripts/relabel_newsletter.py` — one-off script for cleaning mislabeled training data:
- Heuristics: `list_unsubscribe IS NOT NULL`, keyword matches in subject/sender/body
- `--dry-run`, `--domains [--top N]`, `--whitelist DOMAIN ...` flags

### Infrastructure
`docker-compose.yml` defines `db` (postgres:16-alpine) and `app` services; `app` mounts `./models` to `/app/models`

---

## What Is Stubbed or Empty

- **`infermail/backup/__init__.py`** — empty. README describes "Maildir backup" but nothing is implemented.
- **`infermail/sync/__init__.py`** — empty. README describes "IMAP flag sync (feedback loop)" for picking up manual corrections made on mobile. This is the core of the described feedback loop — it does not exist.
- **`tests/`** — contains only `tests/__init__.py`. No test files exist anywhere.
- **`.env.example`** — empty file. The required environment variables are documented only in `CLAUDE.md`.
- **`models/` directory** — not created by any setup step; `scripts/train.py` creates it on first run but Docker bind-mount assumes it exists.
- **No Dockerfile** — `docker-compose.yml` has `build: .` referencing a Dockerfile that does not exist. The app service cannot be built.
- **`rules.yml`** — README mentions `config/rules.yml` alongside `accounts.yml`, but it does not exist. Rules are persisted in the DB via the labeler, not a config file — the README is outdated.

---

## Bugs

### Critical: `predictor.py:_build_features` has an indentation bug
In `infermail/classify/predictor.py`, the `rows.append(...)` call at line 69 is **outside the `for e in emails` loop** — it is at the same indentation level as the `for` statement. This means for a batch of N emails, only the last email's features are ever appended. All ML classifications are performed on a 1-row DataFrame regardless of input size, and the results are then zipped against the full email list, producing nonsensical labels for all but the last email.

```python
# As written (broken):
for e in emails:
    text = ...
    in_spam = ...
    has_unsub = ...
rows.append(...)   # ← outside loop, only runs once
```

This bug silently produces wrong output — no exception is raised because `predict_proba` on a 1-row DataFrame still returns N predictions when broadcast through zip.

---

## Code Smells and Non-Idiomatic Patterns

### `datetime.utcnow()` used in two places
`infermail/classify/labeler.py:116` and `:150`, and `scripts/relabel_newsletter.py:150` use `datetime.utcnow()`, which is deprecated since Python 3.12 and produces a naive datetime. The rest of the codebase correctly uses `datetime.now(timezone.utc)`.

### `_get_or_create_label()` duplicated three times
Identical function (modulo color dict formatting) exists in:
- `infermail/classify/labeler.py:43`
- `infermail/classify/predictor.py:22`
- `scripts/relabel_newsletter.py:41`

Should live in `infermail/db/models.py` or a shared `infermail/db/utils.py`.

### `load_dotenv()` called at module import in `runner.py`
`infermail/fetch/runner.py:16` calls `load_dotenv()` at module level. `pydantic-settings` already loads `.env` via `SettingsConfigDict(env_file=".env")` when `Settings` is instantiated in `config.py`. The call in runner.py is redundant and runs as a side effect of importing the module.

### `model_path` default points to Docker path
`infermail/config.py:22` — `model_path: Path = Path("/app/models/classifier.joblib")`. This works in Docker but fails silently in local dev unless `MODEL_PATH` is set in `.env`. The `Predictor` handles the missing file gracefully (logs a warning and returns empty), so no error is raised — classify just silently produces no ML output.

### `MODEL_RATIONALE` is hardcoded for LinearSVC regardless of benchmark winner
`scripts/train.py:42` — `MODEL_RATIONALE` is always written verbatim to `RESULTS.md`, even when `--benchmark` selects lgbm or voting as the winner. The rationale would be wrong in those cases.

### `voting` in `_get_models()` is not a `Pipeline`
`scripts/train.py:105` — all other entries in the dict returned by `_get_models()` are `Pipeline` objects. `voting` is a bare `VotingClassifier`, which has no `named_steps` attribute and behaves differently. This is fine functionally but inconsistent and could confuse code that assumes all values are `Pipeline` instances.

### `confusion_matrix` called twice
`scripts/train.py:296` and `:337` — `confusion_matrix(y_test, y_pred)` is called once for printing and again when calling `_write_results_md`. Minor inefficiency; the result should be stored in a variable.

### IMAP fetch downloads all UIDs every sync
`infermail/fetch/imap.py:147` — `client.search("ALL")` retrieves every UID in the folder on every sync. For an inbox with 10,000 messages, this loads 10,000 UIDs into memory. The existing_uids filter then fires a `.in_(all_uids)` SQL query, which can hit parameter limits at scale. A `SINCE` date filter (e.g., last 30 days) would make this O(recent) rather than O(total).

### `UnsubscribeLog` is a dead table
`infermail/db/models.py:160` defines `UnsubscribeLog` with full status tracking (`pending`, `success`, `failed`, `skipped`). Nothing in the codebase reads from or writes to this table. The `Email.unsubscribe_log` relationship is defined but never accessed.

### `EmailClassification.is_overridden` is never set
`infermail/db/models.py:131` — the `is_overridden` column exists on `email_classifications` but is never set to `True` anywhere in the codebase. Presumably intended to track when a manual label overrides an ML prediction.

---

## What a Fresh Developer Needs to Know

1. **The critical bug in `predictor.py:_build_features` must be fixed before ML classification is used.** See above. The indentation error means only the last email in every batch gets real features.

2. **There is no Dockerfile.** `docker-compose up app` will fail. You need to write one before deployment.

3. **The feedback loop described in README does not exist.** `infermail/sync/` is empty. Manual corrections made on mobile (IMAP folder moves) are not picked up. The described architecture is aspirational.

4. **No tests.** There is no test coverage for any module. The fetch, classification, and labeling logic are untested.

5. **Local model path must be set.** Add `MODEL_PATH=models/classifier.joblib` (or absolute path) to `.env` for local dev. The default `/app/models/classifier.joblib` only works inside Docker.

6. **Three accounts are hardcoded in `config/accounts.yml`** with real email addresses. The file is checked into version control. Passwords are correctly excluded (read from env), but the addresses themselves are not private.

7. **Training requires `uv sync --extra train`** (scikit-learn, pandas, joblib, lightgbm, matplotlib, seaborn are not in the base deps). The daemon and CLI do not require these — they lazy-import joblib only when the model exists.

8. **`wichtig` label exists in the labeler TUI** (`[w]` key) and gets written to the DB, but is silently dropped in training (`_build_target` maps only spam/newsletter/inbox). Emails labeled `wichtig` contribute zero training signal.
