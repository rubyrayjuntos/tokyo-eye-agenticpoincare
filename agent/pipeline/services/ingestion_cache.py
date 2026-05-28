"""Ephemeral cache for ingestion payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class CacheEntry:
    request_id: str
    path: Path


class IngestionCache:
    def __init__(self, root: str = ".cache/gosp/ingestion") -> None:
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def write_payload(self, request_id: str, payload: Dict[str, Any]) -> CacheEntry:
        path = self.root / f"broker_{request_id}.json"
        path.write_text(json.dumps(payload, indent=2))
        self._append_index(request_id=request_id, path=path)
        return CacheEntry(request_id=request_id, path=path)

    def read_payload(self, request_id: str) -> Dict[str, Any]:
        path = self.root / f"broker_{request_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Cached payload not found for request {request_id}")
        return json.loads(path.read_text())

    def list_cached(self) -> list[Dict[str, str]]:
        index = self._read_index()
        summaries: list[Dict[str, str]] = []
        for entry in index:
            try:
                payload = json.loads(Path(entry["path"]).read_text())
                metadata = payload.get("metadata", {})
                summaries.append({
                    "request_id": entry["request_id"],
                    "created_at": entry.get("created_at", ""),
                    "pdb_id": str(metadata.get("pdb_id", "")),
                    "source": str(metadata.get("source", "")),
                })
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
        return summaries

    def _append_index(self, request_id: str, path: Path) -> None:
        index = self._read_index()
        index.append({
            "request_id": request_id,
            "path": str(path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        self.index_path.write_text(json.dumps(index, indent=2))

    def _read_index(self) -> list[Dict[str, str]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text())
