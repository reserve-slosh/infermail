"""Interactive email labeling tool for building training data."""

from __future__ import annotations

import re
from datetime import datetime

import readchar
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from sqlalchemy import func, exists
from sqlalchemy.orm import Session

from infermail.db.helpers import get_or_create_label
from infermail.db.models import Email, EmailClassification, Label, ClassificationMethod, Rule
from infermail.db.session import SessionLocal

console = Console()

SKIP_FOLDERS = {"Gesendet", "Sent", "Sent Items", "Gesendete Elemente", "OUTBOX", "Drafts", "Entwürfe"}

LABELS = {
    "i": "inbox",
    "n": "newsletter",
    "s": "spam",
    "w": "wichtig",
    " ": None,  # skip
    "b": None,  # back
    "g": None,  # domain bulk-spam
    "r": None,  # regex rule
    "q": None,  # quit
}

LABEL_COLORS = {
    "inbox": "green",
    "newsletter": "blue",
    "spam": "red",
    "wichtig": "yellow",
}



def _extract_domain(sender: str | None) -> str | None:
    """Extract domain from email address."""
    if not sender:
        return None
    match = re.search(r"@([\w.-]+)", sender)
    return match.group(1).lower() if match else None


def _get_unlabeled(session: Session, account_name: str | None, limit: int) -> list[Email]:
    q = (
        session.query(Email)
        .filter(
            ~exists().where(
                EmailClassification.email_id == Email.id,
                EmailClassification.method == ClassificationMethod.manual,
            )
        )
        .filter(~Email.imap_folder.in_(SKIP_FOLDERS))
        .order_by(Email.received_at.desc())
    )
    if account_name:
        from infermail.db.models import Account
        q = q.join(Account).filter(Account.name == account_name)
    return q.limit(limit).all()


def _count_unlabeled(session: Session, account_name: str | None) -> int:
    q = session.query(func.count(Email.id)).filter(
        ~exists().where(
            EmailClassification.email_id == Email.id,
            EmailClassification.method == ClassificationMethod.manual,
        )
    ).filter(~Email.imap_folder.in_(SKIP_FOLDERS))
    if account_name:
        from infermail.db.models import Account
        q = q.join(Account).filter(Account.name == account_name)
    return q.scalar()


def _count_labeled(session: Session) -> dict[str, int]:
    rows = (
        session.query(Label.name, func.count(EmailClassification.id))
        .join(EmailClassification)
        .filter(EmailClassification.method == ClassificationMethod.manual)
        .group_by(Label.name)
        .all()
    )
    return dict(rows)


def _apply_label(session: Session, email: Email, label_name: str) -> None:
    label = get_or_create_label(session, label_name)
    existing = (
        session.query(EmailClassification)
        .filter_by(email_id=email.id, method=ClassificationMethod.manual)
        .first()
    )
    if existing:
        existing.label_id = label.id
        existing.classified_at = datetime.utcnow()
    else:
        session.add(EmailClassification(
            email_id=email.id,
            label_id=label.id,
            method=ClassificationMethod.manual,
            confidence=1.0,
        ))
    session.commit()


def _remove_label(session: Session, email: Email) -> None:
    session.query(EmailClassification).filter_by(
        email_id=email.id,
        method=ClassificationMethod.manual,
    ).delete()
    session.commit()


def _bulk_spam_domain(session: Session, domain: str) -> int:
    """Label all unlabeled emails from domain as spam. Returns count."""
    label = get_or_create_label(session, "spam")
    matching = (
        session.query(Email)
        .filter(
            Email.sender.ilike(f"%@{domain}"),
            ~exists().where(
                EmailClassification.email_id == Email.id,
                EmailClassification.method == ClassificationMethod.manual,
            ),
        )
        .all()
    )
    for email in matching:
        session.add(EmailClassification(
            email_id=email.id,
            label_id=label.id,
            method=ClassificationMethod.manual,
            confidence=1.0,
        ))
    session.commit()
    return len(matching)


def _save_domain_rule(session: Session, domain: str) -> None:
    """Persist a sender-domain spam rule for future use by daemon."""
    existing = session.query(Rule).filter(
        Rule.condition["domain"].astext == domain
    ).first()
    if not existing:
        session.add(Rule(
            name=f"spam-domain:{domain}",
            priority=10,
            condition={"type": "sender_domain", "domain": domain},
            action={"label": "spam"},
            is_active=True,
        ))
        session.commit()


def _handle_domain_spam(session: Session, email: Email) -> int:
    """Interactive domain bulk-spam flow. Returns number of emails labeled."""
    domain = _extract_domain(email.sender)
    if not domain:
        console.print("[red]Keine Domain erkennbar.[/red]")
        return 0

    # Count affected
    total = session.query(func.count(Email.id)).filter(
        Email.sender.ilike(f"%@{domain}"),
        ~exists().where(
            EmailClassification.email_id == Email.id,
            EmailClassification.method == ClassificationMethod.manual,
        ),
    ).scalar()

    console.print(f"\n[bold red]Domain-Filter: @{domain}[/bold red]")
    console.print(f"{total} unlabeled Mails von dieser Domain werden als [red]spam[/red] markiert.")
    console.print("Bestätigen? [bold][y/n][/bold] ", end="")

    while True:
        confirm = readchar.readchar()
        if confirm in ("y", "n"):
            break

    if confirm == "n":
        console.print("\n[dim]Abgebrochen.[/dim]")
        return 0

    count = _bulk_spam_domain(session, domain)
    _save_domain_rule(session, domain)
    console.print(f"\n[green]{count} Mails markiert. Regel gespeichert.[/green]")
    return count


def _render_email(email: Email, remaining: int, total: int, labeled_counts: dict) -> None:
    console.clear()

    stats = Table.grid(padding=(0, 2))
    stats.add_row(
        Text(f"📬 {remaining}/{total} verbleibend", style="bold"),
        *[
            Text(f"{name}: {count}", style=LABEL_COLORS.get(name, "white"))
            for name, count in labeled_counts.items()
        ],
    )
    console.print(stats)
    console.rule()

    meta = Table.grid(padding=(0, 1))
    meta.add_column(style="dim", width=12)
    meta.add_column()
    meta.add_row("Von:", Text(f"{email.sender_name} <{email.sender}>", style="cyan"))
    meta.add_row("Betreff:", Text(email.subject or "(kein Betreff)", style="bold white"))
    meta.add_row("Datum:", Text(str(email.received_at)[:16] if email.received_at else "?", style="dim"))
    meta.add_row("Ordner:", Text(email.imap_folder or "?", style="dim"))
    console.print(meta)
    console.rule()

    body = (email.body_text or "").strip()[:400]
    if body:
        console.print(Panel(body, title="Vorschau", border_style="dim"))

    console.rule()
    keys = Text()
    keys.append(" [i] ", style="bold green"); keys.append("inbox  ")
    keys.append(" [n] ", style="bold blue"); keys.append("newsletter  ")
    keys.append(" [s] ", style="bold red"); keys.append("spam  ")
    keys.append(" [w] ", style="bold yellow"); keys.append("wichtig  ")
    keys.append(" [g] ", style="bold red"); keys.append("domain-spam  ")
    keys.append(" [r] ", style="bold cyan"); keys.append("regex-regel  ")
    keys.append(" [space] ", style="bold dim"); keys.append("skip  ")
    keys.append(" [b] ", style="bold magenta"); keys.append("zurück  ")
    keys.append(" [q] ", style="bold dim"); keys.append("beenden")
    console.print(keys, justify="center")



def _read_pattern(prompt: str) -> str:
    """Read a regex pattern from stdin with backspace support."""
    console.print(prompt, end="")
    pattern = ""
    while True:
        ch = readchar.readchar()
        if ch in (readchar.key.ENTER, "\n", "\r"):
            break
        if ch in (readchar.key.BACKSPACE, "\x7f"):
            pattern = pattern[:-1]
            console.print(f"\r{prompt}{pattern}  \r{prompt}{pattern}", end="")
        else:
            pattern += ch
            console.print(ch, end="")
    console.print()
    return pattern


def _handle_regex_rule(session: Session) -> int:
    """Interactive regex rule flow — matches sender address and/or name."""
    console.print("\n[bold cyan]Regex-Regel[/bold cyan]")
    console.print("Felder: [bold][a][/bold] Adresse  [bold][n][/bold] Name  [bold][b][/bold] Beide (Standard: beide)")
    console.print("Auswahl: ", end="")

    while True:
        field_key = readchar.readchar()
        if field_key in ("a", "n", "b", "\r", "\n"):
            break

    field_map = {"a": "address", "n": "name", "b": "both"}
    field = field_map.get(field_key, "both")
    field_label = {"address": "Adresse", "name": "Name", "both": "Adresse + Name"}[field]
    console.print(f"\r→ Filter auf: [cyan]{field_label}[/cyan]")

    pattern = _read_pattern("Pattern eingeben: ")

    if not pattern:
        console.print("[dim]Abgebrochen.[/dim]")
        return 0

    try:
        re.compile(pattern)
    except re.error as e:
        console.print(f"[red]Ungültiges Regex: {e}[/red]")
        return 0

    all_unlabeled = (
        session.query(Email)
        .filter(
            ~exists().where(
                EmailClassification.email_id == Email.id,
                EmailClassification.method == ClassificationMethod.manual,
            )
        )
        .all()
    )

    def _matches(email: Email) -> bool:
        addr = email.sender or ""
        name = email.sender_name or ""
        if field == "address":
            return bool(re.search(pattern, addr, re.IGNORECASE))
        if field == "name":
            return bool(re.search(pattern, name, re.IGNORECASE))
        return bool(re.search(pattern, addr, re.IGNORECASE) or re.search(pattern, name, re.IGNORECASE))

    matching = [e for e in all_unlabeled if _matches(e)]

    if not matching:
        console.print("[yellow]Keine unlabeled Mails gefunden.[/yellow]")
        return 0

    console.print(f"{len(matching)} unlabeled Mails matchen [cyan]{pattern}[/cyan] → [red]spam[/red]")
    console.print("Bestätigen? [bold][y/n][/bold] ", end="")

    while True:
        confirm = readchar.readchar()
        if confirm in ("y", "n"):
            break

    if confirm == "n":
        console.print("\n[dim]Abgebrochen.[/dim]")
        return 0

    label = get_or_create_label(session, "spam")
    for email in matching:
        session.add(EmailClassification(
            email_id=email.id,
            label_id=label.id,
            method=ClassificationMethod.manual,
            confidence=1.0,
        ))

    # Save rule
    existing = session.query(Rule).filter(
        Rule.condition["pattern"].astext == pattern
    ).first()
    if not existing:
        session.add(Rule(
            name=f"spam-pattern:{pattern}",
            priority=10,
            condition={"type": "sender_regex", "pattern": pattern, "field": field},
            action={"label": "spam"},
            is_active=True,
        ))

    session.commit()
    console.print(f"\n[green]{len(matching)} Mails markiert. Regel gespeichert.[/green]")
    return len(matching)


def run_labeler(account_name: str | None = None, batch: int = 200) -> None:
    with SessionLocal() as session:
        total_unlabeled = _count_unlabeled(session, account_name)
        if total_unlabeled == 0:
            console.print("[green]Alle Mails sind bereits gelabelt![/green]")
            return

        console.print(f"[bold]{total_unlabeled} unlabeled Mails[/bold] — lade nächste {min(batch, total_unlabeled)}...")
        emails = _get_unlabeled(session, account_name, batch)

        skipped = 0
        labeled = 0
        history: list[tuple[Email, bool]] = []

        i = 0
        while i < len(emails):
            email = emails[i]
            remaining = total_unlabeled - labeled
            labeled_counts = _count_labeled(session)
            _render_email(email, remaining, total_unlabeled, labeled_counts)

            while True:
                key = readchar.readchar()
                if key in LABELS:
                    break

            if key == "q":
                console.print(f"\n[yellow]Beendet. {labeled} gelabelt, {skipped} übersprungen.[/yellow]")
                return

            if key == "b":
                if history:
                    prev_email, was_labeled = history.pop()
                    if was_labeled:
                        _remove_label(session, prev_email)
                        labeled -= 1
                    else:
                        skipped -= 1
                    i -= 1
                else:
                    console.print("[dim]Kein vorheriger Eintrag.[/dim]")
                continue

            if key == "g":
                count = _handle_domain_spam(session, email)
                if count > 0:
                    labeled += count
                    emails = _get_unlabeled(session, account_name, batch)
                    i = 0
                continue

            if key == "r":
                count = _handle_regex_rule(session)
                if count > 0:
                    labeled += count
                    emails = _get_unlabeled(session, account_name, batch)
                    i = 0
                continue

            if key == " ":
                history.append((email, False))
                skipped += 1
                i += 1
                continue

            label_name = LABELS[key]
            _apply_label(session, email, label_name)
            history.append((email, True))
            labeled += 1
            i += 1

        console.print(f"\n[green]Batch fertig. {labeled} gelabelt, {skipped} übersprungen.[/green]")
        remaining = _count_unlabeled(session, account_name)
        if remaining > 0:
            console.print(f"[dim]{remaining} Mails noch offen. Nochmal starten zum Fortfahren.[/dim]")