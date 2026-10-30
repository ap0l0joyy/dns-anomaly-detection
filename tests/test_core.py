"""Unit and end-to-end tests for the DNS anomaly lab."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlparse, parse_qs

import numpy as np

from dns_detect.cli import make_demo
from dns_detect.collector import export_query_logs
from dns_detect.features import entropy, make_features, sequence_context, sequences
from dns_detect.parser import normalize, read_events
from dns_detect.pipeline import detect, train


class ParserTests(unittest.TestCase):
    def test_common_technitium_aliases_are_normalized(self):
        event = normalize({"Time": "2026-10-27T10:00:00Z", "Client": "192.0.2.5",
                           "Query": "portal.lab.test.", "Type": "A", "Status": "NOERROR"})
        self.assertEqual(event["domain"], "portal.lab.test.")
        self.assertEqual(event["client_ip"], "192.0.2.5")
        self.assertEqual(event["query_type"], "A")

    def test_technitium_api_export_headers_parse(self):
        event = normalize({"timestamp": "2026-10-27T10:00:00Z", "clientIpAddress": "192.0.2.5",
                           "qname": "portal.lab.test", "qtype": "A", "rcode": "NoError"})
        self.assertEqual(event["client_ip"], "192.0.2.5")
        self.assertEqual(event["domain"], "portal.lab.test")

    def test_csv_requires_a_domain_field(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            path.write_text("time,client\nnow,192.0.2.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no recognized domain"):
                read_events(path)

    def test_csv_events_are_sorted_oldest_to_newest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reverse.csv"
            path.write_text("timestamp,client_ip,qname\n2026-10-27T10:00:02Z,192.0.2.5,b.lab.test\n2026-10-27T10:00:00Z,192.0.2.5,a.lab.test\n", encoding="utf-8")
            events = read_events(path)
            self.assertEqual([row["domain"] for row in events], ["a.lab.test", "b.lab.test"])


class FeatureTests(unittest.TestCase):
    def test_entropy_and_feature_shape(self):
        self.assertAlmostEqual(entropy("aaaa"), 0.0)
        self.assertAlmostEqual(entropy("abcd"), 2.0)
        rows, names = make_features([
            {"domain": "portal.lab.test", "client_ip": "one", "query_type": "A", "response_code": "NOERROR"},
            {"domain": "x9ab.lab.test", "client_ip": "two", "query_type": "TXT", "response_code": "NXDOMAIN"},
        ])
        self.assertEqual(rows.shape, (2, len(names)))
        self.assertEqual(rows[0, names.index("client_rate_60s")], 1)
        self.assertEqual(rows[1, names.index("client_rate_60s")], 1)

    def test_sequence_history_does_not_mix_clients(self):
        x = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        result = sequences(x, ["A", "B", "A"], size=2)
        np.testing.assert_array_equal(result[1, :, 0], [0.0, 2.0])
        np.testing.assert_array_equal(result[2, :, 0], [1.0, 3.0])
        contexts = sequence_context(x, ["A", "B", "A"], size=2)
        np.testing.assert_array_equal(contexts[0, :, 0], [0.0, 0.0])
        np.testing.assert_array_equal(contexts[1, :, 0], [0.0, 0.0])
        np.testing.assert_array_equal(contexts[2, :, 0], [0.0, 1.0])


class EndToEndTests(unittest.TestCase):
    def test_demo_train_detect_writes_models_and_alerts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "demo.csv"
            models = root / "models"
            reports = root / "reports"
            make_demo(data, count=240)
            events = read_events(data)
            report = train(events, models, reports, epochs=2)
            self.assertEqual(report["samples"], 240)
            self.assertIn("hybrid", report)
            self.assertIn("transformer", report)
            self.assertTrue((models / "rf_bundle.joblib").exists())
            self.assertTrue((models / "transformer.pt").exists())
            alerts = detect(events, models, reports)
            self.assertEqual(len(alerts), 240)
            self.assertTrue(any(row["is_alert"] for row in alerts))
            for row in alerts:
                self.assertEqual(row["is_alert"], bool(
                    row["random_forest_attack_probability"] >= 0.65 or
                    row["transformer_anomaly_score"] > row["transformer_threshold"]))
            self.assertTrue((reports / "alerts.json").exists())
            self.assertTrue((reports / "alerts.csv").exists())
            self.assertEqual(len(json.loads((reports / "alerts.json").read_text())), 240)


class CollectorTests(unittest.TestCase):
    def test_collector_sends_bearer_token_and_saves_csv(self):
        seen = {}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["authorization"] = self.headers.get("Authorization")
                seen["query"] = parse_qs(urlparse(self.path).query)
                data = b"timestamp,clientIpAddress,qname,qtype,rcode\n2026-10-27T10:00:00Z,192.0.2.5,portal.lab.test,A,NoError\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *_args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                output = export_query_logs(f"http://127.0.0.1:{server.server_port}", "lab-token",
                                           "Query Logs", "Technitium.DnsServerApp.QueryLogs.QueryLogsApp",
                                           Path(temp) / "logs.csv", start="2026-10-27T00:00:00Z")
                self.assertTrue(output.exists())
                self.assertEqual(seen["authorization"], "Bearer lab-token")
                self.assertEqual(seen["query"]["name"], ["Query Logs"])
                self.assertEqual(len(read_events(output)), 1)
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
