"""Training script for the infermail binary email classifier.

Requires [train] extras:
    uv sync --extra train
    uv run python scripts/train.py

Output:
    models/classifier.joblib
    models/meta.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier

from infermail.db.models import ClassificationMethod, Email, EmailClassification, Label
from infermail.db.session import SessionLocal

SPAM_FOLDERS = {"Spamverdacht", "Junk-E-Mail", "[Gmail]/Spam", "Spam", "spam"}
F1_THRESHOLD = 0.85
MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_RATIONALE = (
    "LinearSVC was selected for its strong performance on high-dimensional sparse "
    "TF-IDF features, fast inference, and low memory footprint suitable for "
    "automated monthly retraining on low-power hardware."
)


def _load_data(session) -> pd.DataFrame:
    rows = (
        session.query(
            Email.subject,
            Email.sender,
            Email.sender_name,
            Email.body_text,
            Email.imap_folder,
            Email.list_unsubscribe,
            Label.name.label("label"),
        )
        .join(EmailClassification, EmailClassification.email_id == Email.id)
        .join(Label, Label.id == EmailClassification.label_id)
        .filter(EmailClassification.method == ClassificationMethod.manual)
        .all()
    )
    return pd.DataFrame(rows, columns=["subject", "sender", "sender_name", "body_text", "imap_folder", "list_unsubscribe", "label"])


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = (
        (df["subject"].fillna("") + " ")
        + (df["sender"].fillna("") + " ")
        + (df["sender_name"].fillna("") + " ")
        + df["body_text"].fillna("").str[:2000]
    )
    df["in_spam_folder"] = df["imap_folder"].isin(SPAM_FOLDERS).astype(float)
    df["has_unsubscribe"] = df["list_unsubscribe"].notna() & df["list_unsubscribe"].ne("")
    df["has_unsubscribe"] = df["has_unsubscribe"].astype(float)
    return df[["text", "in_spam_folder", "has_unsubscribe"]]


def _build_target(df: pd.DataFrame) -> pd.Series:
    mapping = {"spam": 0, "newsletter": 0, "inbox": 1}
    y = df["label"].map(mapping)
    return y.dropna().astype(int)


def _get_models(preprocessor: ColumnTransformer) -> dict:
    logreg = Pipeline([
        ("features", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])
    linearsvc = Pipeline([
        ("features", preprocessor),
        ("clf", LinearSVC(max_iter=2000, class_weight="balanced", dual=True)),
    ])
    sgd = Pipeline([
        ("features", preprocessor),
        ("clf", SGDClassifier(loss="modified_huber", max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    lgbm = Pipeline([
        ("features", preprocessor),
        ("clf", LGBMClassifier(n_estimators=300, class_weight="balanced", random_state=42, verbose=-1)),
    ])
    voting = VotingClassifier(
        estimators=[
            ("logreg", clone(logreg)),
            ("linearsvc", Pipeline([
                ("features", clone(preprocessor)),
                ("clf", CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight="balanced", dual=True))),
            ])),
            ("sgd", clone(sgd)),
            ("lgbm", clone(lgbm)),
        ],
        voting="soft",
    )
    return {
        "logreg": logreg,
        "linearsvc": linearsvc,
        "sgd": sgd,
        "lgbm": lgbm,
        "voting": voting,
    }


def _run_benchmark(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, dict]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=30_000, sublinear_tf=True, ngram_range=(1, 2)), "text"),
            ("flags", "passthrough", ["in_spam_folder", "has_unsubscribe"]),
        ]
    )
    models = _get_models(preprocessor)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    results = {}
    for name, model in models.items():
        print(f"  [{name}] running 10-fold CV...", flush=True)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        results[name] = {"f1_mean": round(float(scores.mean()), 4), "f1_std": round(float(scores.std()), 4)}

    sorted_results = sorted(results.items(), key=lambda kv: kv[1]["f1_mean"], reverse=True)

    col_w = max(len(n) for n in results) + 2
    print(f"\n{'Model':<{col_w}}  {'F1 mean':>8}  {'± std':>7}")
    print("-" * (col_w + 20))
    for name, metrics in sorted_results:
        print(f"{name:<{col_w}}  {metrics['f1_mean']:>8.4f}  {metrics['f1_std']:>7.4f}")

    return dict(sorted_results)


def _write_results_md(
    *,
    n_samples: int,
    label_counts: dict,
    benchmark: dict | None,
    best_name: str,
    report: dict,
    cm,
    trained_at: str,
    model_version: str,
) -> Path:
    lines = ["# infermail — Training Results", ""]

    # Dataset
    lines += ["## Dataset", ""]
    lines.append(f"- **Total samples:** {n_samples}")
    lines.append("- **Label distribution:**")
    for label, count in sorted(label_counts.items()):
        lines.append(f"  - `{label}`: {count}")
    lines.append("")

    # Features
    lines += [
        "## Features",
        "",
        "- **`text`** — Concatenation of subject, sender address, sender name, and the first 2000 characters of the body; provides the primary lexical signal for distinguishing inbox from spam/newsletter content.",
        "- **`in_spam_folder`** — Binary flag set when the email's IMAP folder matches known spam folder names; acts as a strong prior that the server's own filter has already flagged the message.",
        "- **`has_unsubscribe`** — Binary flag set when the `List-Unsubscribe` header is present; reliably identifies mailing-list traffic that the binary target maps to the not-inbox class.",
        "",
    ]

    # Benchmark table (only when --benchmark was used)
    if benchmark is not None:
        lines += ["## Benchmark (10-fold stratified CV on train split)", ""]
        col_w = max(len(n) for n in benchmark) + 2
        lines.append(f"| {'Model':<{col_w}} | {'CV F1 mean':>10} | {'± std':>7} |")
        lines.append(f"| {'-' * col_w} | {'-' * 10} | {'-' * 7} |")
        for name, metrics in benchmark.items():
            lines.append(f"| {name:<{col_w}} | {metrics['f1_mean']:>10.4f} | {metrics['f1_std']:>7.4f} |")
        lines.append("")

    # Winning model
    lines += [
        "## Winning Model",
        "",
        f"**{best_name}**",
        "",
        MODEL_RATIONALE,
        "",
    ]

    # Test set results
    cm_arr = cm.tolist()
    lines += [
        "## Test Set Results",
        "",
        f"**F1 macro:** {report['macro avg']['f1-score']:.4f}",
        "",
        "| Class | Precision | Recall | F1 |",
        "| --- | --- | --- | --- |",
    ]
    for cls in ["not-inbox", "inbox"]:
        r = report[cls]
        lines.append(f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} |")
    lines += [
        "",
        "**Confusion matrix** (rows = true, cols = predicted):",
        "",
        "| | pred not-inbox | pred inbox |",
        "| --- | --- | --- |",
        f"| true not-inbox | {cm_arr[0][0]} | {cm_arr[0][1]} |",
        f"| true inbox     | {cm_arr[1][0]} | {cm_arr[1][1]} |",
        "",
    ]

    # Footer
    lines += [
        "---",
        "",
        f"- `trained_at`: {trained_at}",
        f"- `model_version`: {model_version}",
    ]

    results_path = MODELS_DIR / "RESULTS.md"
    results_path.write_text("\n".join(lines) + "\n")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run cross-validation benchmark before training.")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading data from DB...")
    with SessionLocal() as session:
        df = _load_data(session)

    print(f"Loaded {len(df)} labeled emails.")
    print(df["label"].value_counts().to_string())

    y = _build_target(df)
    df = df.loc[y.index]
    X = _build_features(df)

    label_counts = df["label"].value_counts().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # Benchmark all models via cross-validation, or default to LinearSVC
    benchmark = None
    if args.benchmark:
        print("\nBenchmarking models...")
        benchmark = _run_benchmark(X_train, y_train)
        best_name = next(iter(benchmark))  # already sorted best-first
        print(f"\nBest model: {best_name}  (CV F1 macro: {benchmark[best_name]['f1_mean']:.4f})")
    else:
        best_name = "linearsvc"

    # Refit best model on full training set
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(max_features=30_000, sublinear_tf=True, ngram_range=(1, 2)),
                "text",
            ),
            ("flags", "passthrough", ["in_spam_folder", "has_unsubscribe"]),
        ]
    )
    pipeline = _get_models(preprocessor)[best_name]

    print(f"\nTraining [{best_name}]...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["not-inbox", "inbox"], output_dict=True)
    print("\n" + classification_report(y_test, y_pred, target_names=["not-inbox", "inbox"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    f1_macro = report["macro avg"]["f1-score"]
    print(f"\nF1 macro: {f1_macro:.4f}  (threshold: {F1_THRESHOLD})")

    if f1_macro < F1_THRESHOLD:
        print(f"F1 {f1_macro:.4f} below threshold {F1_THRESHOLD} — model NOT saved.")
        sys.exit(1)

    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_file = f"classifier_{model_version}.joblib"
    model_path = MODELS_DIR / "classifier.joblib"
    model_archive_path = MODELS_DIR / model_file
    meta_path = MODELS_DIR / "meta.json"

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, model_archive_path)

    meta = {
        "model_version": model_version,
        "model_name": best_name,
        "model_file": model_file,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(df),
        "label_counts": label_counts,
        "metrics": {
            "f1_macro": round(f1_macro, 4),
            "f1_not_inbox": round(report["not-inbox"]["f1-score"], 4),
            "f1_inbox": round(report["inbox"]["f1-score"], 4),
            "precision_inbox": round(report["inbox"]["precision"], 4),
            "recall_inbox": round(report["inbox"]["recall"], 4),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    results_path = _write_results_md(
        n_samples=len(df),
        label_counts=label_counts,
        benchmark=benchmark if args.benchmark else None,
        best_name=best_name,
        report=report,
        cm=confusion_matrix(y_test, y_pred),
        trained_at=meta["trained_at"],
        model_version=model_version,
    )

    print(f"\nModel saved → {model_path}")
    print(f"Archive     → {model_archive_path}")
    print(f"Meta  saved → {meta_path}")
    print(f"Results     → {results_path}")
    print(f"model_version: {model_version}")


if __name__ == "__main__":
    main()
