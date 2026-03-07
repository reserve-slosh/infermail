"""Shared DB utility functions."""

from __future__ import annotations

from sqlalchemy.orm import Session

from infermail.db.models import Label

_LABEL_COLORS = {
    "inbox": "#22c55e",
    "spam": "#ef4444",
    "newsletter": "#3b82f6",
    "wichtig": "#eab308",
}


def get_or_create_label(session: Session, name: str) -> Label:
    """Return existing Label by name, or create and flush it."""
    label = session.query(Label).filter_by(name=name).first()
    if not label:
        label = Label(name=name, is_system=True, color=_LABEL_COLORS.get(name, "#94a3b8"))
        session.add(label)
        session.flush()
    return label
