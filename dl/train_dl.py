"""Entraînement, comparaison (LSTM vs GRU vs Random Forest) et évaluation (Membre 4)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASELINE_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    LABEL_TO_INDEX,
    RANDOM_STATE,
    RISK_LEVELS,
    TEMPORAL_FEATURE_COLUMNS,
)
from .data_generation import load_sequences_from_csv
from .model import TemporalRiskNet
from .preprocessing import (
    SequenceScaler,
    extract_last_step_features,
    prepare_tensors,
    split_sequences_by_patient,
)


def _set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(RISK_LEVELS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    crit_idx = LABEL_TO_INDEX["Critical"]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "critical_recall": round(float(per_class_recall[crit_idx]), 4),
        "critical_precision": round(float(per_class_prec[crit_idx]), 4),
        "per_class": {
            lvl: {
                "precision": round(float(per_class_prec[i]), 4),
                "recall": round(float(per_class_recall[i]), 4),
                "f1": round(float(per_class_f1[i]), 4),
            }
            for i, lvl in enumerate(RISK_LEVELS)
        },
        "confusion_matrix": cm.tolist(),
    }


def _train_rnn(
    cell_type: str,
    X_tr: np.ndarray,
    l_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    l_va: np.ndarray,
    y_va: np.ndarray,
    class_weights: torch.Tensor,
    epochs: int = 80,
    patience: int = 12,
    batch_size: int = 32,
    lr: float = 1.5e-3,
) -> tuple[TemporalRiskNet, dict[str, list[float]], dict[str, Any]]:
    _set_seed(RANDOM_STATE)
    model = TemporalRiskNet(
        input_size=X_tr.shape[2],
        hidden_size=48,
        num_layers=2,
        dropout=0.25,
        num_classes=len(RISK_LEVELS),
        cell_type=cell_type,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(l_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
    )
    X_va_t, l_va_t, y_va_t = torch.from_numpy(X_va), torch.from_numpy(l_va), torch.from_numpy(y_va)

    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_critical_recall": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, bl, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx, bl)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(by)
        train_loss = total_loss / len(y_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t, l_va_t)
            val_loss = float(criterion(val_logits, y_va_t).item())
            val_preds = val_logits.argmax(dim=1).numpy()

        val_metrics = _compute_metrics(y_va, val_preds)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_critical_recall"].append(val_metrics["critical_recall"])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_va_t, l_va_t).argmax(dim=1).numpy()
    best_val_metrics = _compute_metrics(y_va, val_preds)
    best_val_metrics["best_epoch"] = best_epoch
    best_val_metrics["best_val_loss"] = round(best_val_loss, 4)
    return model, history, best_val_metrics


def _predict_rnn(model: TemporalRiskNet, X: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X), torch.from_numpy(lengths))
        probs = torch.softmax(logits, dim=1).numpy()
    return probs.argmax(axis=1), probs


def _save_confusion_svg(cm: list[list[int]], title: str, path: Path) -> None:
    cell, pad_l, pad_t = 62, 95, 55
    n = len(RISK_LEVELS)
    w, h = pad_l + n * cell + 20, pad_t + n * cell + 30
    max_v = max(max(row) for row in cm) or 1
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{w // 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">{title}</text>',
    ]
    for j, lbl in enumerate(RISK_LEVELS):
        lines.append(f'<text x="{pad_l + j * cell + cell // 2}" y="{pad_t - 10}" text-anchor="middle" font-family="sans-serif" font-size="11">{lbl}</text>')
    for i, row in enumerate(cm):
        lines.append(f'<text x="{pad_l - 8}" y="{pad_t + i * cell + cell // 2 + 4}" text-anchor="end" font-family="sans-serif" font-size="11">{RISK_LEVELS[i]}</text>')
        for j, val in enumerate(row):
            intensity = int(245 - 175 * (val / max_v))
            color = f"rgb({intensity},{intensity + 5 if intensity < 250 else 255},255)"
            x, y = pad_l + j * cell, pad_t + i * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ccc"/>')
            lines.append(f'<text x="{x + cell // 2}" y="{y + cell // 2 + 5}" text-anchor="middle" font-family="sans-serif" font-size="13">{val}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_learning_curves_svg(lstm_hist: dict[str, list[float]], gru_hist: dict[str, list[float]], path: Path) -> None:
    w, h, ml, mr, mt, mb = 640, 320, 55, 25, 35, 45
    pw, ph = w - ml - mr, h - mt - mb
    all_vals = lstm_hist["train_loss"] + lstm_hist["val_loss"] + gru_hist["train_loss"] + gru_hist["val_loss"]
    max_loss = max(all_vals) * 1.05
    max_ep = max(len(lstm_hist["train_loss"]), len(gru_hist["train_loss"]), 2)

    def poly(vals: list[float], color: str, dash: str = "") -> str:
        pts = []
        for idx, v in enumerate(vals):
            x = ml + (idx / max(max_ep - 1, 1)) * pw
            y = mt + ph - (v / max_loss) * ph
            pts.append(f"{x:.1f},{y:.1f}")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash_attr} points="{" ".join(pts)}"/>'

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{w // 2}" y="20" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">Courbes perte (train / validation) - LSTM vs GRU</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>',
        poly(lstm_hist["train_loss"], "#1f77b4", "4,3"),
        poly(lstm_hist["val_loss"], "#1f77b4"),
        poly(gru_hist["train_loss"], "#d62728", "4,3"),
        poly(gru_hist["val_loss"], "#d62728"),
        f'<text x="{ml + 10}" y="{mt + 18}" font-family="sans-serif" font-size="11" fill="#1f77b4">--- LSTM train | — LSTM val</text>',
        f'<text x="{ml + 10}" y="{mt + 34}" font-family="sans-serif" font-size="11" fill="#d62728">--- GRU train  | — GRU val</text>',
        f'<text x="{w // 2}" y="{h - 10}" text-anchor="middle" font-family="sans-serif" font-size="11">Époque (max {max_ep})</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_and_evaluate(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    _set_seed(RANDOM_STATE)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences_from_csv(dataset_path)
    train_s, val_s, test_s = split_sequences_by_patient(sequences)

    scaler = SequenceScaler.fit(train_s)
    scaler.save(DEFAULT_SCALER_PATH if Path(artifact_dir) == DEFAULT_ARTIFACT_DIR else out_dir / "scaler_stats.json")

    X_tr, l_tr, y_tr, _ = prepare_tensors(train_s, scaler)
    X_va, l_va, y_va, _ = prepare_tensors(val_s, scaler)
    X_te, l_te, y_te, traj_te = prepare_tensors(test_s, scaler)

    weights = compute_class_weight("balanced", classes=np.arange(len(RISK_LEVELS)), y=y_tr)
    # Légère priorité supplémentaire à la classe Critical pour limiter les faux négatifs
    weights[LABEL_TO_INDEX["Critical"]] *= 1.15
    class_weights = torch.tensor(weights, dtype=torch.float32)

    lstm_model, lstm_hist, lstm_val = _train_rnn("lstm", X_tr, l_tr, y_tr, X_va, l_va, y_va, class_weights)
    gru_model, gru_hist, gru_val = _train_rnn("gru", X_tr, l_tr, y_tr, X_va, l_va, y_va, class_weights)

    # Baseline statique : Random Forest sur le dernier pas uniquement (hr, bp, spo2)
    X_tr_last = extract_last_step_features(train_s)
    X_te_last = extract_last_step_features(test_s)
    baseline = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    baseline.fit(X_tr_last, y_tr)
    base_preds = baseline.predict(X_te_last)

    lstm_preds, _ = _predict_rnn(lstm_model, X_te, l_te)
    gru_preds, _ = _predict_rnn(gru_model, X_te, l_te)

    # Évaluation globale + focus spécifique sur les trajectoires de détérioration
    det_mask = np.array([t in {"deterioration_lente", "deterioration_rapide"} for t in traj_te])

    results = {
        "split_sizes": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        "validation": {"lstm": lstm_val, "gru": gru_val},
        "test": {
            "lstm": _compute_metrics(y_te, lstm_preds),
            "gru": _compute_metrics(y_te, gru_preds),
            "baseline_rf_last_step": _compute_metrics(y_te, base_preds),
        },
        "test_deterioration_only": {
            "count": int(det_mask.sum()),
            "lstm": _compute_metrics(y_te[det_mask], lstm_preds[det_mask]),
            "gru": _compute_metrics(y_te[det_mask], gru_preds[det_mask]),
            "baseline_rf_last_step": _compute_metrics(y_te[det_mask], base_preds[det_mask]),
        },
        "learning_curves": {"lstm": lstm_hist, "gru": gru_hist},
    }

    # Sélection entre LSTM et GRU sur la validation : rappel Critical en priorité, puis F1 macro
    lstm_key = (lstm_val["critical_recall"], lstm_val["f1_macro"], -lstm_val["best_val_loss"])
    gru_key = (gru_val["critical_recall"], gru_val["f1_macro"], -gru_val["best_val_loss"])
    selected_name = "lstm" if lstm_key >= gru_key else "gru"
    selected_model = lstm_model if selected_name == "lstm" else gru_model
    results["selected_model"] = selected_name

    model_path = DEFAULT_MODEL_PATH if Path(artifact_dir) == DEFAULT_ARTIFACT_DIR else out_dir / "lstm_risk.pt"
    torch.save(
        {
            "cell_type": selected_name,
            "input_size": len(TEMPORAL_FEATURE_COLUMNS),
            "hidden_size": selected_model.hidden_size,
            "num_layers": selected_model.num_layers,
            "dropout": selected_model.dropout_rate,
            "num_classes": len(RISK_LEVELS),
            "state_dict": selected_model.state_dict(),
        },
        model_path,
    )
    baseline_path = DEFAULT_BASELINE_PATH if Path(artifact_dir) == DEFAULT_ARTIFACT_DIR else out_dir / "baseline_last_step.joblib"
    joblib.dump(baseline, baseline_path)

    # Tableau comparatif CSV + figures SVG + JSON
    rows = []
    for name in ["lstm", "gru", "baseline_rf_last_step"]:
        m = results["test"][name]
        d = results["test_deterioration_only"][name]
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "critical_recall": m["critical_recall"],
                "critical_precision": m["critical_precision"],
                "det_accuracy": d["accuracy"],
                "det_f1_macro": d["f1_macro"],
                "det_critical_recall": d["critical_recall"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)
    (out_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _save_learning_curves_svg(lstm_hist, gru_hist, out_dir / "learning_curves.svg")
    _save_confusion_svg(results["test"][selected_name]["confusion_matrix"], f"Matrice confusion ({selected_name.upper()})", out_dir / "confusion_dl.svg")
    _save_confusion_svg(results["test"]["baseline_rf_last_step"]["confusion_matrix"], "Matrice confusion (Baseline RF dernier pas)", out_dir / "confusion_baseline.svg")

    return results


if __name__ == "__main__":
    res = train_and_evaluate()
    print(f"Modèle retenu : {res['selected_model'].upper()}")
    print("\n--- Comparaison globale (Test) ---")
    print(pd.read_csv(DEFAULT_ARTIFACT_DIR / "model_comparison.csv").to_string(index=False))
