import os
import re
import json
import math
import traceback
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLineEdit, QMessageBox,
    QFrame
)

# -----------------------------
# LOCAL BSOR PARSER (VENDORED)
# -----------------------------
from bsor.parser import make_bsor as MAKE_BSOR


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v1.4 (stable)"
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

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def extract_score_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"scoreId\s*=\s*(\d+)", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{6,})\b", text)
    return m.group(1) if m else None

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
# Cut model
# -----------------------------
@dataclass
class Cut:
    time: float
    hand: str
    pre: float
    post: float
    acc_dist: float

def extract_cuts(bsor) -> List[Cut]:
    cuts: List[Cut] = []
    for n in getattr(bsor, "notes", []):
        cuts.append(
            Cut(
                time=n["time"],
                hand="left" if n["saberType"] == 0 else "right",
                pre=n["preSwing"],
                post=n["postSwing"],
                acc_dist=n["cutDistanceToCenter"],
            )
        )
    return cuts

def summarize(cuts: List[Cut]) -> Dict[str, Any]:
    return {
        "count": len(cuts),
        "pre_avg": mean([c.pre for c in cuts]),
        "post_avg": mean([c.post for c in cuts]),
        "acc_dist_avg": mean([c.acc_dist for c in cuts]),
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

def save_report(report: Dict[str, Any], label: str) -> Path:
    out = make_output_folder()
    safe = re.sub(r"[^\w\- ]+", "_", label)
    path = out / f"{safe} - {APP_NAME}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path

# -----------------------------
# Worker thread
# -----------------------------
class Worker(QThread):
    done = Signal(str)
    fail = Signal(str)

    def __init__(self, bsor_path: Optional[str], score_input: str):
        super().__init__()
        self.bsor_path = bsor_path
        self.score_input = score_input

    def run(self):
        try:
            if self.bsor_path:
                with open(self.bsor_path, "rb") as f:
                    bsor = MAKE_BSOR(f)
                label = Path(self.bsor_path).stem
            else:
                sid = extract_score_id(self.score_input)
                if not sid:
                    raise RuntimeError("No BSOR file or valid scoreId provided.")
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

        layout = QVBoxLayout(self)

        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        layout.addWidget(title)

        self.input = QLineEdit()
        self.input.setPlaceholderText("Paste BeatLeader scoreId or replay link")
        layout.addWidget(self.input)

        buttons = QHBoxLayout()
        self.choose_btn = QPushButton("Choose .bsor")
        self.analyse_btn = QPushButton("Analyse")
        buttons.addWidget(self.choose_btn)
        buttons.addWidget(self.analyse_btn)
        layout.addLayout(buttons)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.bsor_path = None
        self.choose_btn.clicked.connect(self.pick)
        self.analyse_btn.clicked.connect(self.run_analysis)

    def pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose BSOR", "", "*.bsor")
        if path:
            self.bsor_path = path
            self.log.append(f"Loaded {path}")

    def run_analysis(self):
    self.log.append("Analysing...")
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
