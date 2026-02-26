import os
import re
import json
import math
import traceback
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLineEdit, QMessageBox,
    QFrame
)

# -----------------------------
# FIXED BSOR IMPORT (DO NOT TOUCH)
# -----------------------------
from py_bsor import make_bsor as MAKE_BSOR


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v1.4 (full-report)"
OUTPUT_ROOT = "BeatSaberReplayAnalysis"

# -----------------------------
# Helpers
# -----------------------------
def documents_dir() -> Path:
    return Path.home() / "Documents"

def ensure_output_root() -> Path:
    base = documents_dir() / OUTPUT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base

def make_output_folder() -> Path:
    base = ensure_output_root()
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def stdev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def extract_score_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"scoreId\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{6,})\b", text)
    if m:
        return m.group(1)
    return None

# -----------------------------
# BeatLeader download
# -----------------------------
def download_bsor(score_id: str) -> bytes:
    url = f"https://cdn.beatleader.xyz/replays/{score_id}.bsor"
    r = requests.get(url, timeout=30)
    if not r.ok:
        raise RuntimeError(f"Failed to download BSOR (HTTP {r.status_code})")
    return r.content

# -----------------------------
# Cut data
# -----------------------------
@dataclass
class Cut:
    t: float
    hand: str
    pre: Optional[float]
    post: Optional[float]
    acc_dist: Optional[float]
    x: Optional[float]
    y: Optional[float]

def extract_cuts(bsor) -> List[Cut]:
    cuts: List[Cut] = []
    notes = getattr(bsor, "noteCuts", None) or []
    for n in notes:
        hand = "left" if n.saberType == 0 else "right"
        c = n.cut
        cuts.append(
            Cut(
                t=n.time,
                hand=hand,
                pre=c.beforeCutAngle,
                post=c.afterCutAngle,
                acc_dist=c.cutDistanceToCenter,
                x=c.cutPointX,
                y=c.cutPointY,
            )
        )
    return cuts

def summarize(cuts: List[Cut]) -> Dict[str, Any]:
    pres = [c.pre for c in cuts if c.pre is not None]
    posts = [c.post for c in cuts if c.post is not None]
    dists = [c.acc_dist for c in cuts if c.acc_dist is not None]
    xs = [c.x for c in cuts if c.x is not None]
    ys = [c.y for c in cuts if c.y is not None]

    return {
        "count": len(cuts),
        "pre_avg": mean(pres),
        "post_avg": mean(posts),
        "acc_dist_avg": mean(dists),
        "bias_x_m": mean(xs),
        "bias_y_m": mean(ys),
    }

def build_report(bsor) -> Dict[str, Any]:
    cuts = extract_cuts(bsor)
    left = [c for c in cuts if c.hand == "left"]
    right = [c for c in cuts if c.hand == "right"]

    return {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "summary": {
            "all": summarize(cuts),
            "left": summarize(left),
            "right": summarize(right),
        },
    }

def save_report(report: Dict[str, Any], label: str):
    out = make_output_folder()
    safe = re.sub(r"[^\w\- ]+", "_", label)
    path = out / f"{safe} - {APP_NAME}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path

# -----------------------------
# Worker
# -----------------------------
class Worker(QThread):
    log = Signal(str)
    done = Signal(str)
    fail = Signal(str)

    def __init__(self, path: Optional[str], score: Optional[str]):
        super().__init__()
        self.path = path
        self.score = score

    def run(self):
        try:
            if self.path:
                with open(self.path, "rb") as f:
                    bsor = MAKE_BSOR(f)
                label = Path(self.path).stem
            else:
                sid = extract_score_id(self.score or "")
                if not sid:
                    raise RuntimeError("No BSOR or scoreId provided")
                data = download_bsor(sid)
                import io
                bsor = MAKE_BSOR(io.BytesIO(data))
                label = f"scoreId_{sid}"

            report = build_report(bsor)
            out = save_report(report, label)
            self.done.emit(str(out))

        except Exception:
            self.fail.emit(traceback.format_exc())

# -----------------------------
# UI
# -----------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(900, 520)
        self.setStyleSheet("background:#111;color:white")

        lay = QVBoxLayout(self)
        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        lay.addWidget(title)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Paste BeatLeader scoreId or replay link")
        lay.addWidget(self.input)

        btns = QHBoxLayout()
        self.choose = QPushButton("Choose .bsor")
        self.run = QPushButton("Analyse")
        btns.addWidget(self.choose)
        btns.addWidget(self.run)
        lay.addLayout(btns)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log)

        self.choose.clicked.connect(self.pick)
        self.run.clicked.connect(self.analyse)
        self.bsor_path = None

    def pick(self):
        p, _ = QFileDialog.getOpenFileName(self, "BSOR", "", "*.bsor")
        if p:
            self.bsor_path = p
            self.log.append(f"Loaded {p}")

    def analyse(self):
        self.log.append("Running...")
        self.worker = Worker(self.bsor_path, self.input.text())
        self.worker.done.connect(lambda p: QMessageBox.information(self, "Done", f"Saved:\n{p}"))
        self.worker.fail.connect(lambda e: QMessageBox.critical(self, "Error", e))
        self.worker.start()

def main():
    app = QApplication([])
    w = App()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
