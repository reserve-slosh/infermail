"""SQLAlchemy ORM models for infermail."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ClassificationMethod(str, enum.Enum):
    rule = "rule"
    ml = "ml"
    manual = "manual"


class UnsubscribeMethod(str, enum.Enum):
    mailto = "mailto"
    https = "https"


class UnsubscribeStatus(str, enum.Enum):
    pending = "pending"
    success = "success"
    failed = "failed"
    skipped = "skipped"


class Account(Base):
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    email_address: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    imap_host: Mapped[str] = mapped_column(String(255), nullable=False)
    imap_port: Mapped[int] = mapped_column(Integer, default=993)
    provider: Mapped[str | None] = mapped_column(String(50))  # gmail, gmx, strato, ...
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    emails: Mapped[list[Email]] = relationship(back_populates="account")


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    color: Mapped[str | None] = mapped_column(String(7))  # hex color
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)  # spam, inbox, etc.

    classifications: Mapped[list[EmailClassification]] = relationship(
        back_populates="label"
    )


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    message_id: Mapped[str] = mapped_column(String(512), nullable=False)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), nullable=False)
    imap_uid: Mapped[int | None] = mapped_column(BigInteger)  # UID in source mailbox
    imap_folder: Mapped[str | None] = mapped_column(String(255))
    source_folder: Mapped[str | None] = mapped_column(String(255))
    demoted_to_spam_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    subject: Mapped[str | None] = mapped_column(Text)
    sender: Mapped[str | None] = mapped_column(String(512))
    sender_name: Mapped[str | None] = mapped_column(String(255))
    recipients: Mapped[list | None] = mapped_column(JSONB)  # list of addresses
    reply_to: Mapped[str | None] = mapped_column(String(512))

    body_text: Mapped[str | None] = mapped_column(Text)
    body_html: Mapped[str | None] = mapped_column(Text)
    raw_headers: Mapped[dict | None] = mapped_column(JSONB)

    has_attachments: Mapped[bool] = mapped_column(Boolean, default=False)
    list_unsubscribe: Mapped[str | None] = mapped_column(Text)

    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    account: Mapped[Account] = relationship(back_populates="emails")
    classifications: Mapped[list[EmailClassification]] = relationship(
        back_populates="email"
    )
    unsubscribe_log: Mapped[list[UnsubscribeLog]] = relationship(back_populates="email")

    __table_args__ = (
        UniqueConstraint("message_id", "account_id", name="uq_email_message_account"),
        Index("ix_emails_received_at", "received_at"),
        Index("ix_emails_sender", "sender"),
        Index("ix_emails_account_id", "account_id"),
    )


class EmailClassification(Base):
    __tablename__ = "email_classifications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), nullable=False)
    label_id: Mapped[int] = mapped_column(ForeignKey("labels.id"), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    method: Mapped[ClassificationMethod] = mapped_column(
        Enum(ClassificationMethod), nullable=False
    )
    is_overridden: Mapped[bool] = mapped_column(Boolean, default=False)
    model_version: Mapped[str | None] = mapped_column(String(50))
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    email: Mapped[Email] = relationship(back_populates="classifications")
    label: Mapped[Label] = relationship(back_populates="classifications")

    __table_args__ = (
        UniqueConstraint("email_id", "method", name="uq_classification_email_method"),
        Index("ix_classifications_label_id", "label_id"),
    )


class Rule(Base):
    __tablename__ = "rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    condition: Mapped[dict] = mapped_column(JSONB, nullable=False)
    action: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UnsubscribeLog(Base):
    __tablename__ = "unsubscribe_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), nullable=False)
    method: Mapped[UnsubscribeMethod] = mapped_column(
        Enum(UnsubscribeMethod), nullable=False
    )
    url_or_address: Mapped[str | None] = mapped_column(Text)
    status: Mapped[UnsubscribeStatus] = mapped_column(
        Enum(UnsubscribeStatus), default=UnsubscribeStatus.pending
    )
    attempted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    response_code: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)

    email: Mapped[Email] = relationship(back_populates="unsubscribe_log")
