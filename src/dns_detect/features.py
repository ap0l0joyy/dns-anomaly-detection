"""DNS query and per-client time-window feature extraction."""
from __future__ import annotations

import math
from collections import deque
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

FEATURES = ["domain_length", "label_length", "subdomain_depth", "entropy", "first_label_entropy",
            "numeric_ratio", "digit_count", "hyphen_count", "query_type_code", "is_nxdomain",
            "client_rate_60s", "client_unique_domains_60s"]

def entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text.lower())
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def _time(value: Any) -> float:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError):
        return 0.0

def make_features(events: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    parsed = []
    for i, event in enumerate(events):
        domain = str(event.get("domain", "")).strip().rstrip(".").lower()
        labels = [x for x in domain.split(".") if x]
        first = labels[0] if labels else ""
        parsed.append((i, _time(event.get("timestamp")), str(event.get("client_ip", "unknown")), domain, labels, first))
    rows = []
    query_types = {"A": 1, "AAAA": 2, "TXT": 3, "MX": 4, "NS": 5, "CNAME": 6, "PTR": 7, "SRV": 8}
    window: dict[str, deque] = {}
    for i, ts, client, domain, labels, first in parsed:
        # Events should be timestamp-sorted. Fall back to input order for missing timestamps.
        event_time = ts if ts else i * 2.0
        history = window.setdefault(client, deque())
        while history and event_time - history[0][0] > 60:
            history.popleft()
        history.append((event_time, domain))
        peers = list(history)
        numeric = sum(c.isdigit() for c in domain)
        rows.append([
            len(domain), len(first), max(0, len(labels) - 1), entropy(domain), entropy(first),
            numeric / max(1, len(domain)), numeric, domain.count("-"), query_types.get(str(events[i].get("query_type", "A")).upper(), 0),
            int("NXDOMAIN" in str(events[i].get("response_code", "")).upper()), len(peers), len({p[1] for p in peers}),
        ])
    return np.asarray(rows, dtype=np.float32), FEATURES.copy()

def sequences(x: np.ndarray, clients: list[str] | None = None, size: int = 8) -> np.ndarray:
    """Make per-client rolling windows; each output still maps to one event."""
    out = np.zeros((len(x), size, x.shape[1]), dtype=np.float32)
    history: dict[str, list[np.ndarray]] = {}
    for i in range(len(x)):
        client = clients[i] if clients else "all"
        previous = history.setdefault(client, [])
        previous.append(x[i])
        chunk = previous[-size:]
        out[i, size - len(chunk):] = np.asarray(chunk)
    return out

def sequence_context(x: np.ndarray, clients: list[str], size: int = 8) -> np.ndarray:
    """For each event, return up to `size` earlier events from the same client."""
    out = np.zeros((len(x), size, x.shape[1]), dtype=np.float32)
    history: dict[str, list[np.ndarray]] = {}
    for i, client in enumerate(clients):
        previous = history.setdefault(client, [])
        chunk = previous[-size:]
        if chunk:
            out[i, size - len(chunk):] = np.asarray(chunk)
        previous.append(x[i])
    return out

