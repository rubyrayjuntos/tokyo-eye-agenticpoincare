"""Structure ingestion pipeline — BinaryCIF download, parse, enrich, score.

This package handles the full lifecycle of PDB structure ingestion:
  1. Download BinaryCIF from RCSB (downloader.py)
  2. Parse with biotite into structured dataclasses (parser.py)
  3. Enrich metadata via RCSB Data API (metadata.py)
  4. Score chains and select computation scope (chain_scorer.py)

Usage:
    from science.dtie.ingest.downloader import download_bcif
    from science.dtie.ingest.parser import parse_bcif
"""

from __future__ import annotations
