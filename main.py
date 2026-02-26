import re
import io
import json
import traceback
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLineEdit, QMessageBox, QFrame
)

# py-bsor (PyPI)
from bsor.Bsor import make_bsor
from bsor.Scoring import calc_stats


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v2.0 (py-bsor)"
OUTPUT_ROOT = "BeatSaberReplayAnalysis"


# -----------------------------
# Paths / output
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

def safe_name(label: str) -> str:
    return re.sub(r"[^\w\-. ]+", "_", label).strip()[:180]

def write_crash_log(text: str) -> Path:
    out = ensure_output_root()
    p = out / "crash_log.txt"
    p.write_text(text, encoding="utf-8")
    return p


# -----------------------------
# BeatLeader helpers
# -----------------------------
def extract_score_id(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"scoreId\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{6,})\b", text)
    return m.group(1) if m else None

def download_bsor_bytes(score_id: str) -> bytes:
    # BeatLeader CDN direct replay download
    url = f"https://cdn.beatleader.xyz/replays/{score_id}.bsor"
    r = requests.get(url, timeout=30)
    if not r.ok or not r.content:
        raise RuntimeError(f"Failed to download BSOR from BeatLeader CDN (HTTP {r.status_code})")
    return r.content


# -----------------------------
# Analysis helpers
# -----------------------------
def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None

def stdev(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5

def pct(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    if part is None or whole in (None, 0):
        return None
    return (part / whole) * 100.0

def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except:
        return None

def get_attr(obj: Any, *names: str) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return None

def get_dict(d: Any, *keys: str) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


# -----------------------------
# Build detailed report
# -----------------------------
def build_report(bsor_obj: Any, label: str) -> Dict[str, Any]:
    """
    Uses calc_stats from py-bsor plus extra breakdowns we can compute safely from notes.
    """
    # calc_stats returns a stats object/dict-like. We'll convert to JSON safely.
    stats = calc_stats(bsor_obj)

    # Notes list
    notes = getattr(bsor_obj, "notes", None) or []
    note_count = len(notes)

    # We try to pull common per-note cut fields if present
    # (py-bsor supports multiple schemas / versions)
    left_dists, right_dists = [], []
    left_pre, right_pre = [], []
    left_post, right_post = [], []
    left_dev, right_dev = [], []

    # 4x3 grid (Beat Saber is 4 columns x 3 rows)
    grid = [[{"n": 0, "dist_avg": None} for _ in range(4)] for _ in range(3)]
    grid_acc = [[[] for _ in range(4)] for _ in range(3)]

    for n in notes:
        # py-bsor note can be dict-like or object-like depending on version
        # Normalize:
        if isinstance(n, dict):
            saber = get_dict(n, "saberType", "hand")
            line = get_dict(n, "lineIndex", "x")
            layer = get_dict(n, "lineLayer", "y")
            cut = get_dict(n, "cut")
            pre = get_dict(n, "preSwing", "beforeCutAngle")
            post = get_dict(n, "postSwing", "afterCutAngle")
            dist = get_dict(n, "cutDistanceToCenter", "distanceToCenter")
            dev = get_dict(n, "cutDirDeviation", "directionDeviation")
        else:
            saber = get_attr(n, "saberType", "hand")
            line = get_attr(n, "lineIndex", "x")
            layer = get_attr(n, "lineLayer", "y")
            cut = get_attr(n, "cut")
            pre = get_attr(n, "preSwing", "beforeCutAngle")
            post = get_attr(n, "postSwing", "afterCutAngle")
            dist = get_attr(n, "cutDistanceToCenter", "distanceToCenter")
            dev = get_attr(n, "cutDirDeviation", "directionDeviation")

        saber = int(saber) if saber is not None else None
        pre = as_float(pre)
        post = as_float(post)
        dist = as_float(dist)
        dev = as_float(dev)

        if saber == 0:  # left
            if dist is not None: left_dists.append(dist)
            if pre is not None: left_pre.append(pre)
            if post is not None: left_post.append(post)
            if dev is not None: left_dev.append(dev)
        elif saber == 1:  # right
            if dist is not None: right_dists.append(dist)
            if pre is not None: right_pre.append(pre)
            if post is not None: right_post.append(post)
            if dev is not None: right_dev.append(dev)

        # Grid accumulation if we have line/layer and dist
        if line is not None and layer is not None and dist is not None:
            try:
                x = int(line)
                y = int(layer)
                # layer usually 0..2 (bottom..top)
                if 0 <= x <= 3 and 0 <= y <= 2:
                    grid_acc[y][x].append(dist)
            except:
                pass

    # finalize grid
    for y in range(3):
        for x in range(4):
            arr = grid_acc[y][x]
            grid[y][x]["n"] = len(arr)
            grid[y][x]["dist_avg"] = mean(arr)

    # Conservative controller setting suggestion from consistent centre bias:
    # NOTE: Requires cutPointX/Y or similar; many replays don't expose it cleanly in calc_stats.
    # We keep placeholder fields and we’ll extend once we confirm where py-bsor exposes cutpoint.
    recommendations = {
        "note": "Calibration recommendations need cut-point bias vectors. This version outputs full swing/accuracy stats + grid; next step adds bias → exact X/Y/Z adjustments.",
        "left": {"pos_x_cm": None, "pos_y_cm": None, "rot_y_deg": None},
        "right": {"pos_x_cm": None, "pos_y_cm": None, "rot_y_deg": None},
        "axis_key": {
            "PositionX": "left/right (cm)",
            "PositionY": "up/down (cm)",
            "PositionZ": "forward/back (cm)",
            "RotationX": "tilt up/down (deg)",
            "RotationY": "yaw turn (deg)",
            "RotationZ": "roll twist (deg)",
        }
    }

    # Convert stats to JSON-safe form
    # py-bsor stats may be a dataclass-like object; we try to serialize robustly
    def to_jsonable(o: Any) -> Any:
        if o is None:
            return None
        if isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, dict):
            return {k: to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_jsonable(v) for v in o]
        if hasattr(o, "__dict__"):
            return {k: to_jsonable(v) for k, v in o.__dict__.items() if not k.startswith("_")}
        return str(o)

    report = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "meta": {
            "label": label,
            "parsed_at": datetime.datetime.now().isoformat(),
            "notes_in_replay": note_count,
            "source": "py-bsor calc_stats + note breakdown",
        },
        "calc_stats": to_jsonable(stats),
        "breakdown": {
            "left": {
                "dist_avg_m": mean(left_dists),
                "dist_std_m": stdev(left_dists),
                "pre_avg_deg": mean(left_pre),
                "post_avg_deg": mean(left_post),
                "dir_dev_avg_deg": mean(left_dev),
            },
            "right": {
                "dist_avg_m": mean(right_dists),
                "dist_std_m": stdev(right_dists),
                "pre_avg_deg": mean(right_pre),
                "post_avg_deg": mean(right_post),
                "dir_dev_avg_deg": mean(right_dev),
            },
            "grid_4x3_dist_avg_m": grid,  # [layer][column]
        },
        "recommendations": recommendations,
        "glossary": {
            "pre_swing_deg": "Your angle BEFORE the cut. Low pre = underswing (missed points).",
            "post_swing_deg": "Your follow-through angle AFTER the cut. Low post = underswing (missed points).",
            "dist_to_center_m": "How far from block center you cut (meters). Closer = better.",
            "dir_dev_deg": "Deviation between ideal cut direction and your cut direction.",
            "grid_4x3": "Beat Saber lane grid: 4 columns (0-3) x 3 layers (0 bottom, 2 top).",
        }
    }

    return report


def build_human_txt(report: Dict[str, Any]) -> str:
    bL = report["breakdown"]["left"]
    bR = report["breakdown"]["right"]

    def f(v, unit=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}{unit}"
        return f"{v}{unit}"

    lines = []
    lines.append(f"{APP_NAME} {APP_VERSION}")
    lines.append("")
    lines.append("PLAIN ENGLISH KEY")
    lines.append("- Pre-swing: backswing before contact (more = better, up to scoring cap)")
    lines.append("- Post-swing: follow-through after contact (more = better, up to scoring cap)")
    lines.append("- Dist-to-center: how far from center you hit (smaller = better)")
    lines.append("- Dir deviation: how off-angle your cut is (smaller = better)")
    lines.append("")
    lines.append("BREAKDOWN")
    lines.append("")
    lines.append("LEFT HAND")
    lines.append(f"  Dist avg: {f(bL['dist_avg_m'], ' m')}   (lower is better)")
    lines.append(f"  Dist std: {f(bL['dist_std_m'], ' m')}")
    lines.append(f"  Pre avg:  {f(bL['pre_avg_deg'], ' °')}")
    lines.append(f"  Post avg: {f(bL['post_avg_deg'], ' °')}")
    lines.append(f"  Dir dev:  {f(bL['dir_dev_avg_deg'], ' °')}")
    lines.append("")
    lines.append("RIGHT HAND")
    lines.append(f"  Dist avg: {f(bR['dist_avg_m'], ' m')}   (lower is better)")
    lines.append(f"  Dist std: {f(bR['dist_std_m'], ' m')}")
    lines.append(f"  Pre avg:  {f(bR['pre_avg_deg'], ' °')}")
    lines.append(f"  Post avg: {f(bR['post_avg_deg'], ' °')}")
    lines.append(f"  Dir dev:  {f(bR['dir_dev_avg_deg'], ' °')}")
    lines.append("")
    lines.append("GRID (4x3) dist-to-center averages (meters)")
    lines.append("Rows: 2 top, 1 mid, 0 bottom | Cols: 0..3 left→right")
    grid = report["breakdown"]["grid_4x3_dist_avg_m"]
    for y in (2, 1, 0):
        row = []
        for x in range(4):
            row.append(f(grid[y][x]["dist_avg"], ""))
        lines.append(f"  Layer {y}: " + " | ".join(row))
    lines.append("")
    lines.append("NOTE")
    lines.append(report["recommendations"]["note"])
    lines.append("")
    lines.append("Outputs saved in Documents\\" + OUTPUT_ROOT)

    return "\n".join(lines)


def save_outputs(report: Dict[str, Any], label: str) -> (Path, Path):
    out_dir = make_output_folder()
    base = safe_name(label)
    json_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.json"
    txt_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.txt"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(build_human_txt(report), encoding="utf-8")

    return json_path, txt_path


# -----------------------------
# Worker
# -----------------------------
class Worker(QThread):
    log = Signal(str)
    success = Signal(str)
    error = Signal(str)

    def __init__(self, bsor_path: Optional[str], score_input: str):
        super().__init__()
        self.bsor_path = bsor_path
        self.score_input = score_input

    def run(self):
        try:
            label = ""
            if self.bsor_path:
                self.log.emit(f"Loading file: {self.bsor_path}")
                label = Path(self.bsor_path).stem
                with open(self.bsor_path, "rb") as f:
                    bsor_obj = make_bsor(f)
            else:
                sid = extract_score_id(self.score_input.strip())
                if not sid:
                    raise RuntimeError("Paste a BeatLeader replay link/scoreId OR choose a .bsor file.")
                self.log.emit(f"ScoreId detected: {sid}")
                self.log.emit("Downloading BSOR from BeatLeader CDN...")
                data = download_bsor_bytes(sid)
                self.log.emit(f"Downloaded {len(data)} bytes")
                bsor_obj = make_bsor(io.BytesIO(data))
                label = f"scoreId_{sid}"

            self.log.emit("Calculating stats (py-bsor calc_stats)...")
            report = build_report(bsor_obj, label)

            self.log.emit("Writing outputs...")
            jp, tp = save_outputs(report, label)

            self.success.emit(f"Saved:\n{jp}\n{tp}")

        except Exception as e:
            tb = traceback.format_exc()
            crash_path = write_crash_log(tb)
            self.error.emit(f"{e}\n\nFull crash log saved to:\n{crash_path}\n\n{tb}")


# -----------------------------
# UI
# -----------------------------
class DropBox(QFrame):
    file_dropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed rgba(255,255,255,0.25);
                border-radius: 12px;
                background: rgba(255,255,255,0.03);
            }
        """)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self.label = QLabel("Drop a .bsor replay here\n(or click 'Choose File')")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: rgba(255,255,255,0.85);")
        lay.addWidget(self.label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path.lower().endswith(".bsor"):
            self.file_dropped.emit(path)
        else:
            QMessageBox.warning(self, "Not a BSOR", "Please drop a .bsor file.")


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(980, 560)
        self.setStyleSheet("background:#0f0f12; color:white;")

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        subtitle = QLabel(
            "Drag & drop a .bsor, OR paste a BeatLeader replay link / scoreId.\n"
            f"Outputs JSON + TXT in Documents\\{OUTPUT_ROOT}."
        )
        subtitle.setStyleSheet("color: rgba(255,255,255,0.7);")

        left.addWidget(title)
        left.addWidget(subtitle)

        self.drop = DropBox()
        self.drop.setFixedHeight(240)
        self.drop.file_dropped.connect(self.on_file)
        left.addWidget(self.drop)

        self.status = QLabel("")
        self.status.setStyleSheet("color: rgba(255,255,255,0.7);")
        left.addWidget(self.status)

        left.addWidget(QLabel("Log"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 12px;")
        self.log.setMinimumHeight(200)
        left.addWidget(self.log)

        right.addWidget(QLabel("Replay link / scoreId"))
        self.replay_input = QLineEdit()
        self.replay_input.setPlaceholderText("https://replay.beatleader.com/?scoreId=...  OR just the scoreId")
        self.replay_input.setStyleSheet("padding:10px; border-radius:10px; background: rgba(255,255,255,0.04);")
        right.addWidget(self.replay_input)

        self.choose_btn = QPushButton("Choose File...")
        self.choose_btn.clicked.connect(self.choose_file)
        self.analyse_btn = QPushButton("Analyse")
        self.analyse_btn.clicked.connect(self.analyse)

        for b in (self.choose_btn, self.analyse_btn):
            b.setStyleSheet("""
                QPushButton {
                    padding: 12px;
                    border-radius: 10px;
                    background: rgba(255,255,255,0.08);
                }
                QPushButton:hover { background: rgba(255,255,255,0.12); }
                QPushButton:disabled { background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.35); }
            """)

        right.addStretch(1)
        right.addWidget(self.choose_btn)
        right.addWidget(self.analyse_btn)
        right.addStretch(6)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        self.bsor_path: Optional[str] = None
        self.worker: Optional[Worker] = None

    def append_log(self, msg: str):
        self.log.append(msg)

    def on_file(self, path: str):
        self.bsor_path = path
        self.status.setText(f"Loaded: {Path(path).name}")
        self.append_log(f"Loaded file: {path}")

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose .bsor file", str(Path.home()), "BSOR files (*.bsor)")
        if path:
            self.on_file(path)

    def analyse(self):
        self.analyse_btn.setEnabled(False)
        self.choose_btn.setEnabled(False)
        self.append_log("---- Analyse clicked ----")

        score_input = self.replay_input.text().strip()
        self.worker = Worker(self.bsor_path, score_input)
        self.worker.log.connect(self.append_log)
        self.worker.success.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_success(self, msg: str):
        self.analyse_btn.setEnabled(True)
        self.choose_btn.setEnabled(True)
        self.status.setText("Done.")
        QMessageBox.information(self, "Success", msg)

    def on_error(self, msg: str):
        self.analyse_btn.setEnabled(True)
        self.choose_btn.setEnabled(True)
        self.status.setText("Failed.")
        self.append_log(msg)
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication([])
    w = App()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
