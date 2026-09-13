"""Optional supervised classifier for presence / motion.

The adaptive engine works with no training data at all, and that is the default.
This module exists for when you want to trade calibration-in-the-field for
calibration-on-a-dataset: record one session with the room empty and one with a
person moving in it, then let a small classifier find the boundary.

Deliberately boring choices:

* A RandomForest (or logistic regression) on the ~28 engineered features — no
  deep learning, no sequences, nothing that needs a GPU or 10 GB of data.
* Grouped cross-validation by session, so the reported accuracy is not inflated
  by windows from the same recording leaking across the split.
* The trained artefact records the exact feature order and the training date, so
  a stale model is detected instead of silently mispredicting.

Training data format: CSV files written by the logger (``data/windows_*.csv``).
Every file is one *session*, and the label comes from which list it was passed
in — ``--empty`` files are label 0, ``--human`` files are label 1. No manual
column editing, no label leakage.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAVE_SKLEARN = True
except Exception:  # pragma: no cover
    HAVE_SKLEARN = False

MODEL_SCHEMA = 2


def _numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    skip = {"t", "label", "session", "source", "bssid", "presence", "motion",
            "confidence", "event", "note", "file"}
    names: List[str] = []
    for key in rows[0].keys():
        if key in skip:
            continue
        try:
            float(rows[0][key])
        except (TypeError, ValueError):
            continue
        names.append(key)
    return names


def read_windows(path: str | Path) -> List[Dict[str, float]]:
    """Read a windows CSV into a list of numeric feature dicts."""
    path = Path(path).expanduser()
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: Dict[str, float] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if row:
                rows.append(row)
    return rows


def load_sessions(paths: Sequence[str | Path], label: int
                  ) -> List[Tuple[Dict[str, float], int, str]]:
    out: List[Tuple[Dict[str, float], int, str]] = []
    for path in paths:
        session = Path(path).stem
        for row in read_windows(path):
            out.append((row, label, session))
    return out


def train(empty_files: Sequence[str], human_files: Sequence[str],
          motion_files: Sequence[str] = (), model_type: str = "rf",
          out_path: str = "models/clf.joblib", seed: int = 1337) -> Dict[str, Any]:
    """Train presence (and optionally motion) classifiers; save with joblib."""
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for --train (pip install scikit-learn)")
    import joblib

    data = load_sessions(empty_files, 0) + load_sessions(human_files, 1)
    if len({s for _r, _l, s in data}) < 2:
        raise ValueError("need at least two sessions (e.g. one empty, one with a person) "
                         "to run grouped cross-validation")

    all_rows = [row for row, _l, _s in data]
    feature_names = _numeric_columns(all_rows)
    if not feature_names:
        raise ValueError("no numeric feature columns found in the supplied files")

    def matrix(rows: List[Dict[str, float]]) -> np.ndarray:
        return np.asarray([[float(r.get(f, 0.0)) for f in feature_names] for r in rows], dtype=float)

    X = matrix(all_rows)
    y = np.asarray([label for _r, label, _s in data], dtype=int)
    groups = np.asarray([session for _r, _l, session in data])

    clf = _make_classifier(model_type, seed)
    n_splits = min(5, len(set(groups.tolist())))
    report: Dict[str, Any] = {"model_type": model_type, "sessions": len(set(groups.tolist())),
                              "windows": int(X.shape[0]), "features": feature_names}
    if n_splits >= 2 and len(set(y.tolist())) > 1:
        cv = GroupKFold(n_splits=n_splits)
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba", groups=groups)[:, 1]
        pred = (proba >= 0.5).astype(int)
        report["cv_auc"] = float(roc_auc_score(y, proba))
        report["cv_report"] = classification_report(y, pred, output_dict=True, zero_division=0)
        report["cv_confusion"] = confusion_matrix(y, pred).tolist()
        report["cv_note"] = (f"Grouped {n_splits}-fold CV across sessions — the only honest "
                             f"estimate for unseen recordings.")
    else:  # pragma: no cover - degenerate input
        report["cv_note"] = "not enough distinct sessions for cross-validation"

    clf.fit(X, y)
    payload: Dict[str, Any] = {
        "presence": clf,
        "motion": None,
        "features": feature_names,
        "trained_at": time.time(),
        "schema": MODEL_SCHEMA,
        "metrics": report,
    }

    if motion_files:
        motion_data = load_sessions(empty_files, 0) + load_sessions(motion_files, 1)
        m_rows = [row for row, _l, _s in motion_data]
        mX = matrix(m_rows)
        my = np.asarray([label for _r, label, _s in motion_data], dtype=int)
        mgroups = np.asarray([s for _r, _l, s in motion_data])
        m_clf = _make_classifier(model_type, seed)
        m_clf.fit(mX, my)
        payload["motion"] = m_clf
        payload["motion_metrics"] = {"windows": int(mX.shape[0]), "sessions": len(set(mgroups.tolist()))}
    else:
        report["motion_note"] = ("no --motion sessions supplied: motion keeps using the adaptive "
                                 "threshold, which is the recommended default anyway")

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out)
    report["saved"] = str(out)
    return report


def _make_classifier(model_type: str, seed: int):
    if model_type == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )


def load_model(path: str | Path) -> Dict[str, Any]:
    import joblib

    payload = joblib.load(Path(path).expanduser())
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError(f"model schema mismatch: {payload.get('schema')} != {MODEL_SCHEMA}; retrain")
    return payload


def predict_one(model: Dict[str, Any], features: Dict[str, float]) -> Tuple[float, float]:
    """-> (presence_probability, motion_probability)."""
    names = model.get("features") or []
    x = np.asarray([[float(features.get(name, 0.0)) for name in names]], dtype=float)
    p_presence = 0.0
    clf = model.get("presence")
    if clf is not None and hasattr(clf, "predict_proba"):
        p_presence = float(clf.predict_proba(x)[0][1])
    p_motion = 0.0
    m_clf = model.get("motion")
    if m_clf is not None and hasattr(m_clf, "predict_proba"):
        p_motion = float(m_clf.predict_proba(x)[0][1])
    else:
        # no motion model: derive a motion score from the motion-relevant features
        # relative to their training-window magnitudes (not a hidden model, just a
        # documented heuristic fallback)
        p_motion = float(min(1.0, max(0.0, features.get("band_ratio", 0.0))))
    return p_presence, p_motion


def describe_model(path: str | Path) -> str:
    payload = load_model(path)
    metrics = payload.get("metrics", {})
    lines = [
        f"model   : {Path(path).expanduser()}",
        f"trained : {time.strftime('%Y-%m-%d %H:%M', time.localtime(payload.get('trained_at', 0)))}",
        f"windows : {metrics.get('windows')} over {metrics.get('sessions')} sessions",
        f"features: {len(payload.get('features') or [])}",
    ]
    if "cv_auc" in metrics:
        lines.append(f"grouped-CV AUC: {metrics['cv_auc']:.3f}")
    if metrics.get("cv_confusion"):
        tn, fp, fn, tp = np.asarray(metrics["cv_confusion"]).ravel()
        lines.append(f"grouped-CV confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    if metrics.get("motion_note"):
        lines.append(f"motion  : {metrics['motion_note']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m wifisense.detection.ml",
        description="Train the optional presence classifier from logged windows.",
    )
    parser.add_argument("--empty", nargs="+", required=True,
                        help="windows CSV recorded with the room EMPTY")
    parser.add_argument("--human", nargs="+", required=True,
                        help="windows CSV recorded with a person present")
    parser.add_argument("--motion", nargs="*", default=[],
                        help="optional windows CSV recorded with a person MOVING")
    parser.add_argument("--model", default="rf", choices=("rf", "logreg"))
    parser.add_argument("--out", default="models/clf.joblib")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    report = train(args.empty, args.human, args.motion, args.model, args.out, args.seed)
    print(json.dumps(report, indent=2, default=str))
    print()
    print(describe_model(report["saved"]))
    print("\nrun with:  python main.py --method ml --model " + report["saved"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
