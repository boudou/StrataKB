"""Shared data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID


@dataclass
class User:
    """Current MCP session user."""

    external_id: str
    role: str  # viewer, operator, admin
    display_name: str | None = None


@dataclass
class KnowledgeLayer:
    """Knowledge layer configuration."""

    id: UUID
    name: str
    display_name: str
    description: str | None
    metadata_schema: dict
    chunk_config: dict
    status: str
    doc_count: int
    chunk_count: int
    last_ingested: datetime | None
    created_at: datetime
    updated_at: datetime
    origin: str | None = None
    owner: str | None = None
    root_path: str | None = None
