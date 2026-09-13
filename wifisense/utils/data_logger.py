"""Data logging: every measurement, every decision, every event — on disk.

Three files per session plus a metadata sidecar, all plain CSV/JSON so pandas,
Excel, or two lines of awk can open them:

* ``samples_<session>.csv`` — one row per physical sample (raw RSSI/CSI summary).
* ``windows_<session>.csv`` — one row per analysis window: the full feature
  vector **and** the decision. This is the training set for ``--method ml`` and
  the file you label by recording an empty session and a "someone in the room"
  session.
* ``events_<session>.csv`` — the alarm timeline (armed / triggered / cleared).
* ``session_<session>.json`` — capabilities, config, baseline description,
  counters. Without this, a CSV from three weeks ago is uninterpretable.

CSI windows carry per-subcarrier vectors, which do not belong in a CSV; they are
appended to ``csi_<session>_part<N>.npz`` in batches (amplitude and phase arrays
plus timestamps) so nothing is lost and nothing is inflated.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..collectors.base import Sample


class DataLogger:
    """Session-scoped CSV/JSONL/ npz writer with buffered flushes."""

    def __init__(self, directory: str | Path = "data", enabled: bool = True,
                 fmt: str = "csv", flush_every: int = 1, log_events: bool = True,
                 session_id: Optional[str] = None, max_rows_per_file: int = 500_000):
        self.enabled = bool(enabled)
        self.dir = Path(directory).expanduser()
        self.fmt = fmt if fmt in ("csv", "jsonl") else "csv"
        self.flush_every = max(1, int(flush_every))
        self.log_events = bool(log_events)
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.max_rows_per_file = int(max_rows_per_file)

        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        self._headers: Dict[str, List[str]] = {}
        self._buffers: Dict[str, List[Dict[str, Any]]] = {}
        self._counts: Dict[str, int] = {}
        self._part: Dict[str, int] = {}
        self._csi_buffer: List[Dict[str, Any]] = []
        self.started_at = time.time()
        self.paths: Dict[str, str] = {}

        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # file plumbing
    # ------------------------------------------------------------------ #
    def _path_for(self, kind: str) -> Path:
        suffix = "csv" if self.fmt == "csv" else "jsonl"
        part = self._part.get(kind, 0)
        tail = f"_part{part}" if part else ""
        return self.dir / f"{kind}_{self.session_id}{tail}.{suffix}"

    def _ensure(self, kind: str, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        if kind in self._writers:
            extra = [k for k in row if k not in self._headers[kind]]
            if extra:
                self._headers[kind].extend(extra)
                self._reopen(kind)
            return
        self._headers[kind] = list(row.keys())
        self._buffers[kind] = []
        self._counts[kind] = 0
        self._reopen(kind)

    def _reopen(self, kind: str) -> None:
        path = self._path_for(kind)
        existing = path.exists()
        if self.fmt == "csv":
            handle = path.open("a", newline="")
            writer = csv.DictWriter(handle, fieldnames=self._headers[kind], extrasaction="ignore")
            if not existing:
                writer.writeheader()
            self._files[kind] = handle
            self._writers[kind] = writer
        else:
            self._files[kind] = path.open("a")
        self.paths[kind] = str(path)

    def _write_row(self, kind: str, row: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._ensure(kind, row)
        if self.fmt == "csv":
            self._writers[kind].writerow(row)
        else:
            self._files[kind].write(json.dumps(row) + "\n")
        self._counts[kind] = self._counts.get(kind, 0) + 1

        if self._counts[kind] % self.flush_every == 0:
            self._files[kind].flush()

        # rotate very long sessions so no single file becomes unopenable
        if self._counts[kind] >= self.max_rows_per_file:
            self._files[kind].close()
            self._part[kind] = self._part.get(kind, 0) + 1
            self._counts[kind] = 0
            self._writers.pop(kind, None)
            self._ensure(kind, row)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def log_sample(self, sample: Sample, extra: Optional[Dict[str, Any]] = None) -> None:
        row = sample.to_row()
        if extra:
            row.update(extra)
        self._write_row("samples", row)
        if sample.is_csi:
            self.append_csi(sample)

    def log_window(self, features: Dict[str, float], state: Any = None,
                   extra: Optional[Dict[str, Any]] = None) -> None:
        row: Dict[str, Any] = {"t": round(time.time(), 6)}
        row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in features.items()})
        if state is not None:
            row.update({
                "presence": int(bool(state.presence)),
                "motion": int(bool(state.motion)),
                "confidence": round(float(state.confidence), 2),
                "presence_score": round(float(state.presence_score), 4),
                "motion_score": round(float(state.motion_score), 4),
                "presence_label": state.presence_label,
                "motion_label": state.motion_label,
                "presence_ratio": round(float(state.presence_ratio), 4),
                "motion_ratio": round(float(state.motion_ratio), 4),
                "calibrated": int(bool(state.calibrated)),
                "method": state.method,
                "reason": state.reason,
                "label": "",   # fill in later to build a supervised dataset
            })
        if extra:
            row.update(extra)
        self._write_row("windows", row)

    def log_event(self, row: Dict[str, Any]) -> None:
        if not self.log_events:
            return
        self._write_row("events", row)

    def append_csi(self, sample: Sample, batch: int = 2000) -> None:
        if not self.enabled or not sample.amplitude:
            return
        self._csi_buffer.append({
            "t": sample.t,
            "amplitude": np.asarray(sample.amplitude, dtype=np.float32),
            "phase": np.asarray(sample.phase, dtype=np.float32) if sample.phase else None,
            "rssi": sample.rssi,
            "subcarriers": sample.n_subcarriers,
        })
        if len(self._csi_buffer) >= batch:
            self.flush_csi()

    def flush_csi(self) -> Optional[Path]:
        if not self._csi_buffer or not self.enabled:
            return None
        part = self._part.get("csi", 0)
        path = self.dir / f"csi_{self.session_id}_part{part}.npz"
        widths = {len(item["amplitude"]) for item in self._csi_buffer}
        if len(widths) != 1:
            # heterogeneous frame lengths: store a flat concatenation with offsets
            amplitude = np.concatenate([item["amplitude"] for item in self._csi_buffer])
            offsets = np.cumsum([0] + [item["amplitude"].size for item in self._csi_buffer])
            phases = [item["phase"] for item in self._csi_buffer if item["phase"] is not None]
            np.savez_compressed(
                path,
                t=np.asarray([item["t"] for item in self._csi_buffer], dtype=np.float64),
                rssi=np.asarray([item["rssi"] if item["rssi"] is not None else np.nan
                                 for item in self._csi_buffer], dtype=np.float32),
                amplitude=amplitude, offsets=offsets.astype(np.int64),
                phase=np.concatenate(phases) if phases else np.asarray([], dtype=np.float32),
                layout=np.asarray(["ragged"]),
            )
        else:
            amplitude = np.vstack([item["amplitude"] for item in self._csi_buffer])
            phases = [item["phase"] for item in self._csi_buffer if item["phase"] is not None]
            phase = np.vstack(phases) if phases else np.asarray([], dtype=np.float32)
            np.savez_compressed(
                path,
                t=np.asarray([item["t"] for item in self._csi_buffer], dtype=np.float64),
                rssi=np.asarray([item["rssi"] if item["rssi"] is not None else np.nan
                                 for item in self._csi_buffer], dtype=np.float32),
                amplitude=amplitude, phase=phase, layout=np.asarray(["rect"]),
            )
        self._csi_buffer.clear()
        self._part["csi"] = part + 1
        self.paths["csi"] = str(path)
        return path

    def write_meta(self, meta: Dict[str, Any]) -> Path:
        path = self.dir / f"session_{self.session_id}.json"
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "written_at": time.time(),
            "counts": dict(self._counts),
            "files": dict(self.paths),
            **meta,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def counters(self) -> Dict[str, int]:
        return dict(self._counts)

    def close(self) -> Dict[str, Any]:
        self.flush_csi()
        for handle in self._files.values():
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        self._files.clear()
        self._writers.clear()
        return {"files": dict(self.paths), "counts": dict(self._counts),
                "duration_s": time.time() - self.started_at}


# --------------------------------------------------------------------------- #
# label helper
# --------------------------------------------------------------------------- #

def label_windows(path: str | Path, label: str = "human") -> int:
    """Write a label into the ``label`` column of a windows CSV (in place).

    Recording happens faster than labelling, so the logger writes an empty
    ``label`` column and you stamp it afterwards: the whole file is one session,
    so the label is a property of the file. Accepted labels: ``empty`` (no
    person), ``human`` (present but still), ``motion`` (person moving).
    """
    path = Path(path).expanduser()
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            row["label"] = label
            rows.append(row)
    if "label" not in fieldnames:
        fieldnames.append("label")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
