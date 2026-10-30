"""Command-line entry point. Commands: demo-data, train, detect."""
from __future__ import annotations

import argparse
import csv
import json
import random
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .parser import read_events
from .pipeline import detect, train
from .collector import export_query_logs

def make_demo(path: Path, count: int = 1200) -> None:
    """Create explainable synthetic events; labels come from the generator scenario."""
    if count < 100:
        raise ValueError("Use at least 100 demo events so the chronological train/test split has enough examples.")
    random.seed(24)
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = []
    normal_names = ["portal.lab.test", "updates.lab.test", "docs.lab.test", "printer.lab.test", "time.lab.test"]
    for i in range(count):
        attack = i % 5 == 0
        if attack:
            # Simulated tunneling-like query names for a controlled local demo.
            token = "".join(random.choice("abcdef0123456789") for _ in range(random.randint(35, 55)))
            domain = f"{token}.chunk.lab.test"
            rcode = "NOERROR"
        else:
            domain = random.choice(normal_names)
            rcode = "NXDOMAIN" if random.random() < 0.015 else "NOERROR"
        rows.append({"timestamp": (start + timedelta(seconds=i * 2)).isoformat(),
                     "client_ip": f"192.168.20.{random.randint(20, 40)}", "domain": domain,
                     "query_type": random.choice(["A", "A", "AAAA", "TXT"]),
                     "response_code": rcode, "label": "attack" if attack else "benign"})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser(prog="dns-detect", description="Hybrid DNS anomaly detection lab")
    sub = ap.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-data", help="generate labeled synthetic practice events")
    demo.add_argument("--out", default="data/raw/demo.csv")
    demo.add_argument("--count", type=int, default=1200)
    collect = sub.add_parser("collect-technitium", help="download CSV query logs from Technitium's API")
    collect.add_argument("--server-url", default=os.environ.get("TECHNITIUM_URL", "http://127.0.0.1:5380"))
    collect.add_argument("--app-name", required=True, help="name of the installed query-log DNS app")
    collect.add_argument("--class-path", required=True, help="class path shown for that query-log app")
    collect.add_argument("--start", help="optional ISO-8601 start timestamp")
    collect.add_argument("--end", help="optional ISO-8601 end timestamp")
    collect.add_argument("--out", default="data/raw/technitium.csv")
    fit = sub.add_parser("train", help="train Random Forest and benign-only Transformer")
    fit.add_argument("--input", required=True); fit.add_argument("--models", default="models"); fit.add_argument("--reports", default="reports")
    fit.add_argument("--epochs", type=int, default=25, help="Transformer training epochs (lower this for a quick smoke run)")
    fit.add_argument("--rf-threshold", type=float, default=0.65, help="Random Forest attack-probability threshold")
    run = sub.add_parser("detect", help="score DNS events and write JSON/CSV alerts")
    run.add_argument("--input", required=True); run.add_argument("--models", default="models"); run.add_argument("--reports", default="reports")
    run.add_argument("--rf-threshold", type=float, default=0.65)
    args = ap.parse_args()
    if args.command == "demo-data":
        make_demo(Path(args.out), args.count)
        print(f"Generated synthetic DNS data: {args.out}")
    elif args.command == "collect-technitium":
        token = os.environ.get("TECHNITIUM_TOKEN", "")
        result = export_query_logs(args.server_url, token, args.app_name, args.class_path,
                                   args.out, args.start, args.end)
        print(f"Downloaded Technitium query logs: {result}")
    elif args.command == "train":
        report = train(read_events(args.input), args.models, args.reports, epochs=args.epochs,
                       rf_threshold=args.rf_threshold)
        print(json.dumps({"trained_samples": report["samples"], "transformer_threshold": report["transformer_threshold"],
                          "report": str(Path(args.reports) / "training_report.json")}, indent=2))
    else:
        alerts = detect(read_events(args.input), args.models, args.reports, args.rf_threshold)
        print(f"Scored {len(alerts)} DNS events; alerts: {sum(x['is_alert'] for x in alerts)}")
        print(f"Results: {Path(args.reports) / 'alerts.json'} and alerts.csv")

if __name__ == "__main__":
    main()

