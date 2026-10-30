"""Train and run the two-path hybrid detector."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from .features import make_features, sequence_context
from .models import TransformerPredictor, fit_transformer, transformer_scores

DEFAULT_RF_THRESHOLD = 0.65

def train(events: list[dict], model_dir: str | Path, report_dir: str | Path, epochs: int = 25,
          rf_threshold: float = DEFAULT_RF_THRESHOLD) -> dict:
    labels = [str(e.get("label", "")).strip().lower() for e in events]
    if any(label not in {"benign", "attack"} for label in labels):
        raise ValueError("Training data must have a label column containing only benign or attack.")
    y = np.asarray([int(label == "attack") for label in labels])
    if len(set(y)) != 2:
        raise ValueError("Training data needs both benign and attack examples.")
    x, feature_names = make_features(events)
    # Keep time order intact so sequence contexts never join shuffled rows.
    cutoff = max(1, int(len(y) * 0.75))
    scaler = StandardScaler().fit(x[:cutoff])
    xs = scaler.transform(x).astype(np.float32)
    x_train, x_test = xs[:cutoff], xs[cutoff:]
    y_train, y_test = y[:cutoff], y[cutoff:]
    if len(set(y_train)) != 2:
        raise ValueError("The first 75% of training data must contain both benign and attack labels; mix scenarios over time.")
    client_ids = [str(e.get("client_ip", "unknown")) for e in events]

    rf = RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", min_samples_leaf=2,
                                random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf_scores = rf.predict_proba(x_test)[:, 1]
    rf_pred = rf_scores >= rf_threshold

    # Predict each query from that client's earlier DNS behavior. Train only on
    # benign targets whose preceding context is also benign.
    contexts = sequence_context(xs, client_ids)
    seq_train_all, seq_test = contexts[:cutoff], contexts[cutoff:]
    benign_window = np.zeros(cutoff, dtype=bool)
    seen: dict[str, list[int]] = {}
    for i, client in enumerate(client_ids[:cutoff]):
        history = seen.setdefault(client, [])
        benign_window[i] = bool(y[i] == 0 and all(history[-8:]))
        history.append(int(y[i] == 0))
    seq_normal = seq_train_all[benign_window]
    target_normal = x_train[benign_window]
    if len(seq_normal) < 30:
        raise ValueError("Need at least 30 benign training sequences for Transformer training and threshold calibration.")
    fit_count = max(16, int(len(seq_normal) * 0.8))
    transformer, _ = fit_transformer(seq_normal[:fit_count], target_normal[:fit_count], epochs=epochs)
    calibration_scores = transformer_scores(transformer, seq_normal[fit_count:], target_normal[fit_count:])
    threshold = float(np.quantile(calibration_scores, 0.995))
    anomaly_scores = transformer_scores(transformer, seq_test, x_test)
    transformer_pred = anomaly_scores > threshold
    hybrid_pred = rf_pred | transformer_pred
    report = {
        "samples": len(events), "features": feature_names,
        "random_forest_threshold": rf_threshold,
        "random_forest": classification_report(y_test, rf_pred, labels=[0, 1], target_names=["benign", "attack"], output_dict=True, zero_division=0),
        "random_forest_confusion_matrix": confusion_matrix(y_test, rf_pred, labels=[0, 1]).tolist(),
        "transformer_threshold": threshold,
        "transformer_test_flagged": int(transformer_pred.sum()),
        "transformer_test_samples": len(y_test),
        "transformer": classification_report(y_test, transformer_pred, labels=[0, 1], target_names=["benign", "attack"], output_dict=True, zero_division=0),
        "transformer_confusion_matrix": confusion_matrix(y_test, transformer_pred, labels=[0, 1]).tolist(),
        "hybrid": classification_report(y_test, hybrid_pred, labels=[0, 1], target_names=["benign", "attack"], output_dict=True, zero_division=0),
        "hybrid_confusion_matrix": confusion_matrix(y_test, hybrid_pred, labels=[0, 1]).tolist(),
        "note": "All metrics use the time-ordered held-out 25%. The Transformer predicts the next query from earlier benign queries; its threshold is calibrated on reserved benign examples.",
    }
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"rf": rf, "scaler": scaler, "features": feature_names, "transformer_threshold": threshold}, model_dir / "rf_bundle.joblib")
    torch.save({"state_dict": transformer.state_dict(), "n_features": len(feature_names), "seq_len": 8}, model_dir / "transformer.pt")
    (report_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def detect(events: list[dict], model_dir: str | Path, report_dir: str | Path,
           rf_threshold: float = DEFAULT_RF_THRESHOLD) -> list[dict]:
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    bundle = joblib.load(model_dir / "rf_bundle.joblib")
    x, _ = make_features(events)
    xs = bundle["scaler"].transform(x).astype(np.float32)
    rf_scores = bundle["rf"].predict_proba(xs)[:, 1]
    model_state = torch.load(model_dir / "transformer.pt", map_location="cpu", weights_only=True)
    transformer = TransformerPredictor(model_state["n_features"], seq_len=model_state["seq_len"])
    transformer.load_state_dict(model_state["state_dict"])
    clients = [str(e.get("client_ip", "unknown")) for e in events]
    anomaly_scores = transformer_scores(transformer, sequence_context(xs, clients), xs)
    threshold = float(bundle["transformer_threshold"])
    alerts = []
    for i, event in enumerate(events):
        rf_flag = float(rf_scores[i]) >= rf_threshold
        transformer_flag = float(anomaly_scores[i]) > threshold
        reasons = []
        if rf_flag:
            reasons.append("Random Forest matched learned attack patterns")
        if transformer_flag:
            reasons.append("Transformer found behavior unlike its benign baseline")
        alerts.append({**event,
                       "random_forest_attack_probability": round(float(rf_scores[i]), 6),
                       "transformer_anomaly_score": round(float(anomaly_scores[i]), 6),
                       "transformer_threshold": round(threshold, 6),
                       "is_alert": rf_flag or transformer_flag,
                       "reasons": reasons})
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    import csv
    with (report_dir / "alerts.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(alerts[0].keys()) if alerts else []
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            for row in alerts:
                row = row.copy()
                row["reasons"] = "; ".join(row["reasons"])
                writer.writerow(row)
    return alerts

