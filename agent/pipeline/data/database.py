"""Database configuration and session management."""

from __future__ import annotations

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("GOSP_DATABASE_URL", "sqlite:///./gosp.db")

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def init_db() -> None:
    """Create database tables if they do not exist."""
    from gosp.data import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
