"""Read-only CSV export client for Technitium DNS query-log apps."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def export_query_logs(
    server_url: str,
    token: str,
    app_name: str,
    class_path: str,
    output_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    timeout: int = 30,
) -> Path:
    """Download CSV query logs using Technitium's /api/logs/export endpoint.

    The query log app must already be installed and collecting queries on the
    Technitium server. This method only reads/export logs; it changes no DNS
    settings and sends no DNS queries.
    """
    if not token.strip():
        raise ValueError("Technitium API token is empty. Set TECHNITIUM_TOKEN first.")
    if not app_name.strip() or not class_path.strip():
        raise ValueError("Provide the installed query-log app name and its class path.")

    params = {"name": app_name, "classPath": class_path}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    url = f"{server_url.rstrip('/')}/api/logs/export?{urlencode(params)}"
    request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "text/csv"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Technitium returned HTTP {exc.code} for query-log export.") from None
    except URLError as exc:
        raise RuntimeError(f"Could not reach Technitium query-log API: {exc.reason}") from None

    # The API returns CSV on success and a JSON status/error object on failure.
    text = payload.decode("utf-8-sig", errors="replace")
    if text.lstrip().startswith("{"):
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = {}
        if body.get("status") != "ok":
            message = body.get("errorMessage", "API did not return CSV; check token and app details")
            raise RuntimeError(f"Technitium query-log export failed: {message}")

    if not payload.strip():
        raise RuntimeError("Technitium returned an empty query-log export.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(payload)
    return output_path
