"""Read JSONL or CSV DNS query logs and normalize common field names."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

ALIASES = {
    "timestamp": ("timestamp", "time", "date", "querytime", "datetime"),
    "client_ip": ("client_ip", "client", "clientaddress", "clientipaddress", "source", "source_ip"),
    "domain": ("domain", "query", "qname", "name", "question"),
    "query_type": ("query_type", "type", "qtype", "record_type"),
    "response_code": ("response_code", "rcode", "status", "result"),
    "label": ("label", "class", "ground_truth"),
}

def _key(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())

def normalize(row: dict[str, Any]) -> dict[str, Any]:
    by_key = {_key(str(k)): v for k, v in row.items()}
    out: dict[str, Any] = {}
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if _key(alias) in by_key:
                out[target] = by_key[_key(alias)]
                break
    out.setdefault("timestamp", "")
    out.setdefault("client_ip", "unknown")
    out.setdefault("domain", "")
    out.setdefault("query_type", "A")
    out.setdefault("response_code", "NOERROR")
    return out

def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Read .jsonl/.ndjson, JSON arrays, or CSV. Unsupported raw logs fail clearly."""
    path = Path(path)
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        if suffix in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".json":
            raw = json.load(f)
            rows = raw if isinstance(raw, list) else raw.get("events", raw.get("queries", []))
        elif suffix == ".csv":
            rows = list(csv.DictReader(f))
        else:
            raise ValueError("Use JSONL, JSON, or CSV. Convert the Technitium export first if it is another format.")
    events = [normalize(dict(row)) for row in rows if isinstance(row, dict)]
    if not events:
        raise ValueError(f"No DNS events found in {path}")
    missing = sum(not e["domain"] for e in events)
    if missing:
        raise ValueError(f"{missing} rows have no recognized domain/query field; inspect the Technitium export headers.")
    # API/log exports can be newest-first. Sort ISO timestamps when every row
    # has one so per-client rates and next-query sequences follow time order.
    stamps = []
    for event in events:
        try:
            stamp = datetime.fromisoformat(str(event["timestamp"]).replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            stamps.append(stamp.timestamp())
        except (ValueError, TypeError):
            stamps = []
            break
    if stamps:
        events = [event for _, event in sorted(zip(stamps, events), key=lambda pair: pair[0])]
    return events

