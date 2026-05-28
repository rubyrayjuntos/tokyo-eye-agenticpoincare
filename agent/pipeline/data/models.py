"""SQLAlchemy models for ingestion persistence."""

from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, DateTime, Float, Integer, String, JSON, ForeignKey
from gosp.data.database import Base


class IngestionRequest(Base):
    __tablename__ = "ingestion_requests"

    request_id = Column(String, primary_key=True)
    input_type = Column(String, nullable=False)
    input_value = Column(String, nullable=False)
    source_used = Column(String, nullable=False)
    status = Column(String, nullable=False)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StructureMetadata(Base):
    __tablename__ = "structure_metadata"

    structure_id = Column(String, primary_key=True)
    pdb_id = Column(String, nullable=True, unique=True)
    uniprot_id = Column(String, nullable=True)
    title = Column(String, nullable=True)
    method = Column(String, nullable=True)
    resolution = Column(Float, nullable=True)
    source = Column(String, nullable=False)
    organism = Column(String, nullable=True)
    total_residues = Column(Integer, nullable=True)
    total_atoms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class StructureFile(Base):
    __tablename__ = "structure_files"

    file_id = Column(String, primary_key=True)
    structure_id = Column(String, ForeignKey("structure_metadata.structure_id"), nullable=False)
    file_type = Column(String, nullable=False)
    uri = Column(String, nullable=False)
    size_bytes = Column(Integer, nullable=True)
    checksum = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class IngestionProvenance(Base):
    __tablename__ = "ingestion_provenance"

    provenance_id = Column(String, primary_key=True)
    request_id = Column(String, ForeignKey("ingestion_requests.request_id"), nullable=False)
    structure_id = Column(String, ForeignKey("structure_metadata.structure_id"), nullable=False)
    source_chain = Column(JSON, nullable=True)
    retries = Column(JSON, nullable=True)
    file_hash = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
