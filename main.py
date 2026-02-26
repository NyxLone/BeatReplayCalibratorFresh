import re
import io
import json
import math
import traceback
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import requests

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QLineEdit, QMessageBox, QFrame
)

# py-bsor (PyPI package provides "bsor")
# IMPORTANT: we only use parsing here (no Scoring.py), so no wall event_time crashes.
from bsor.Bsor import make_bsor


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v2.2 (full-detail)"
OUTPUT_ROOT = "BeatSaberReplayAnalysis"


# -----------------------------
# Output paths
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
    url = f"https://cdn.beatleader.xyz/replays/{score_id}.bsor"
    r = requests.get(url, timeout=30)
    if not r.ok or not r.content:
        raise RuntimeError(f"Failed to download BSOR from BeatLeader CDN (HTTP {r.status_code})")
    return r.content


# -----------------------------
# Maths helpers
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
    return math.sqrt(var)

def percentile(xs: List[float], p: float) -> Optional[float]:
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return None
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

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

def detect_hand(v: Any) -> str:
    # Common: 0=left, 1=right
    if isinstance(v, (int, float)):
        return "left" if int(v) == 0 else "right"
    if isinstance(v, str):
        s = v.lower()
        if "left" in s or "red" in s:
            return "left"
        if "right" in s or "blue" in s:
            return "right"
    return "unknown"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Cut:
    t: Optional[float]
    hand: str
    line: Optional[int]
    layer: Optional[int]
    pre: Optional[float]
    post: Optional[float]
    dist: Optional[float]
    x: Optional[float]
    y: Optional[float]
    dir_dev: Optional[float]


# -----------------------------
# Extractor (schema-hunting, robust)
# -----------------------------
def iter_note_like(bsor_obj: Any) -> List[Any]:
    # Most common in py-bsor: bsor_obj.noteCuts
    nl = getattr(bsor_obj, "noteCuts", None)
    if isinstance(nl, list) and nl:
        return nl

    # Sometimes: bsor_obj.notes
    nl = getattr(bsor_obj, "notes", None)
    if isinstance(nl, list) and nl:
        return nl

    # Sometimes: dict container
    if isinstance(bsor_obj, dict):
        for k in ("noteCuts", "notes", "cuts"):
            v = bsor_obj.get(k)
            if isinstance(v, list) and v:
                return v

    return []


def extract_cuts(bsor_obj: Any) -> Tuple[List[Cut], Dict[str, Any]]:
    items = iter_note_like(bsor_obj)

    # Schema sample for debugging (first few items)
    sample = []
    for i, it in enumerate(items[:5]):
        if isinstance(it, dict):
            sample.append({
                "type": "dict",
                "keys": sorted(list(it.keys()))[:80]
            })
        else:
            # object: list public attrs
            attrs = [a for a in dir(it) if not a.startswith("_")]
            # keep it sane
            sample.append({
                "type": type(it).__name__,
                "attrs": attrs[:80]
            })

    cuts: List[Cut] = []
    for it in items:
        if isinstance(it, dict):
            # direct note dict
            t = as_float(get_dict(it, "time", "t", "songTime", "noteTime"))
            saber = get_dict(it, "saberType", "hand", "colorType")
            hand = detect_hand(saber)

            line = get_dict(it, "lineIndex", "line", "x")
            layer = get_dict(it, "lineLayer", "layer", "y")

            # cut subobject or flattened
            cut = get_dict(it, "cut", "noteCut", "cutInfo") or it

            pre = as_float(get_dict(cut, "beforeCutAngle", "preSwing", "preSwingAngle"))
            post = as_float(get_dict(cut, "afterCutAngle", "postSwing", "postSwingAngle"))

            dist = as_float(get_dict(cut, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
            x = as_float(get_dict(cut, "cutPointX", "x", "offsetX"))
            y = as_float(get_dict(cut, "cutPointY", "y", "offsetY"))
            dir_dev = as_float(get_dict(cut, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))

        else:
            # object style
            t = as_float(get_attr(it, "time", "t", "songTime", "noteTime"))

            saber = get_attr(it, "saberType", "hand", "colorType")
            hand = detect_hand(saber)

            line = get_attr(it, "lineIndex", "line", "x")
            layer = get_attr(it, "lineLayer", "layer", "y")

            cut = get_attr(it, "cut", "noteCut", "cutInfo") or it

            pre = as_float(get_attr(cut, "beforeCutAngle", "preSwing", "preSwingAngle"))
            post = as_float(get_attr(cut, "afterCutAngle", "postSwing", "postSwingAngle"))

            dist = as_float(get_attr(cut, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
            x = as_float(get_attr(cut, "cutPointX", "x", "offsetX"))
            y = as_float(get_attr(cut, "cutPointY", "y", "offsetY"))
            dir_dev = as_float(get_attr(cut, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))

        try:
            line_i = int(line) if line is not None else None
        except:
            line_i = None
        try:
            layer_i = int(layer) if layer is not None else None
        except:
            layer_i = None

        cuts.append(Cut(
            t=t,
            hand=hand,
            line=line_i,
            layer=layer_i,
            pre=pre,
            post=post,
            dist=dist,
            x=x,
            y=y,
            dir_dev=dir_dev
        ))

    schema_info = {
        "note_items_found": len(items),
        "sample": sample
    }
    return cuts, schema_info


# -----------------------------
# Summaries
# -----------------------------
def summarize_hand(cuts: List[Cut]) -> Dict[str, Any]:
    pres = [c.pre for c in cuts if c.pre is not None]
    posts = [c.post for c in cuts if c.post is not None]
    dists = [c.dist for c in cuts if c.dist is not None]
    devs = [c.dir_dev for c in cuts if c.dir_dev is not None]

    # underswing: basic definitions used commonly
    under_pre = [p for p in pres if p < 100.0]
    under_post = [p for p in posts if p < 60.0]

    out = {
        "n": len(cuts),
        "time_min": min([c.t for c in cuts if c.t is not None], default=None),
        "time_max": max([c.t for c in cuts if c.t is not None], default=None),

        "pre_avg_deg": mean(pres),
        "pre_std_deg": stdev(pres),
        "pre_p10_deg": percentile(pres, 10),
        "pre_p50_deg": percentile(pres, 50),
        "pre_p90_deg": percentile(pres, 90),

        "post_avg_deg": mean(posts),
        "post_std_deg": stdev(posts),
        "post_p10_deg": percentile(posts, 10),
        "post_p50_deg": percentile(posts, 50),
        "post_p90_deg": percentile(posts, 90),

        "dist_avg_m": mean(dists),
        "dist_std_m": stdev(dists),
        "dist_p10_m": percentile(dists, 10),
        "dist_p50_m": percentile(dists, 50),
        "dist_p90_m": percentile(dists, 90),

        "dir_dev_avg_deg": mean(devs),
        "dir_dev_std_deg": stdev(devs),

        "underswing_rate_pre": (len(under_pre) / len(pres)) if pres else None,
        "underswing_rate_post": (len(under_post) / len(posts)) if posts else None,
    }
    return out


def build_grid_4x3(cuts: List[Cut]) -> List[List[Dict[str, Any]]]:
    # grid[layer][line] where layer 0 bottom..2 top, line 0..3 left->right
    acc = [[[] for _ in range(4)] for _ in range(3)]
    for c in cuts:
        if c.layer is None or c.line is None or c.dist is None:
            continue
        if 0 <= c.layer <= 2 and 0 <= c.line <= 3:
            acc[c.layer][c.line].append(c.dist)

    grid = [[{"n": 0, "dist_avg_m": None, "dist_p50_m": None} for _ in range(4)] for _ in range(3)]
    for y in range(3):
        for x in range(4):
            arr = acc[y][x]
            grid[y][x]["n"] = len(arr)
            grid[y][x]["dist_avg_m"] = mean(arr)
            grid[y][x]["dist_p50_m"] = percentile(arr, 50)
    return grid


def build_time_windows(cuts: List[Cut], window_s: float = 10.0) -> List[Dict[str, Any]]:
    ts = [c.t for c in cuts if c.t is not None]
    if not ts:
        return []
    tmax = max(ts)
    bins = int(math.ceil(tmax / window_s))
    out = []
    for i in range(bins):
        a = i * window_s
        b = (i + 1) * window_s
        seg = [c for c in cuts if c.t is not None and a <= c.t < b]
        out.append({
            "t_start": a,
            "t_end": b,
            "all": summarize_hand(seg),
            "left": summarize_hand([c for c in seg if c.hand == "left"]),
            "right": summarize_hand([c for c in seg if c.hand == "right"]),
        })
    return out


def build_report(bsor_obj: Any, label: str) -> Dict[str, Any]:
    cuts, schema = extract_cuts(bsor_obj)
    left = [c for c in cuts if c.hand == "left"]
    right = [c for c in cuts if c.hand == "right"]
    unk = [c for c in cuts if c.hand == "unknown"]

    report = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "meta": {
            "label": label,
            "parsed_at": datetime.datetime.now().isoformat(),
        },
        "schema_debug": schema,
        "counts": {
            "all": len(cuts),
            "left": len(left),
            "right": len(right),
            "unknown": len(unk),
        },
        "summary": {
            "all": summarize_hand(cuts),
            "left": summarize_hand(left),
            "right": summarize_hand(right),
        },
        "grid_4x3_dist": build_grid_4x3(cuts),
        "time_windows_10s": build_time_windows(cuts, 10.0),
        "glossary": {
            "pre_swing_deg": "Angle BEFORE the cut. Low pre = underswing (lost points).",
            "post_swing_deg": "Angle AFTER the cut (follow-through). Low post = underswing (lost points).",
            "dist_to_center_m": "Distance from center of block when cut. Lower = better accuracy points.",
            "dir_dev_deg": "How far off your cut direction is vs ideal. Lower = better.",
            "grid_4x3": "Beat Saber lanes: 4 columns (0-3) x 3 layers (0 bottom, 2 top).",
            "time_windows_10s": "Performance by time slice to show fatigue/consistency issues.",
        }
    }
    return report


def build_human_txt(report: Dict[str, Any]) -> str:
    def f(v, unit=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}{unit}"
        return f"{v}{unit}"

    sA = report["summary"]["all"]
    sL = report["summary"]["left"]
    sR = report["summary"]["right"]

    lines = []
    lines.append(f"{APP_NAME} {APP_VERSION}")
    lines.append("")
    lines.append(f"Replay label: {report['meta']['label']}")
    lines.append(f"Parsed at: {report['meta']['parsed_at']}")
    lines.append("")
    lines.append("COUNTS")
    lines.append(f"  all:     {report['counts']['all']}")
    lines.append(f"  left:    {report['counts']['left']}")
    lines.append(f"  right:   {report['counts']['right']}")
    lines.append(f"  unknown: {report['counts']['unknown']}")
    lines.append("")
    lines.append("ALL (combined)")
    lines.append(f"  Pre avg:  {f(sA['pre_avg_deg'], '°')}   p10/p50/p90: {f(sA['pre_p10_deg'],'°')} / {f(sA['pre_p50_deg'],'°')} / {f(sA['pre_p90_deg'],'°')}")
    lines.append(f"  Post avg: {f(sA['post_avg_deg'], '°')}  p10/p50/p90: {f(sA['post_p10_deg'],'°')} / {f(sA['post_p50_deg'],'°')} / {f(sA['post_p90_deg'],'°')}")
    lines.append(f"  Dist avg: {f(sA['dist_avg_m'], 'm')}    p10/p50/p90: {f(sA['dist_p10_m'],'m')} / {f(sA['dist_p50_m'],'m')} / {f(sA['dist_p90_m'],'m')}")
    lines.append(f"  Dir dev:  {f(sA['dir_dev_avg_deg'], '°')}")
    lines.append(f"  Underswing pre rate:  {f(sA['underswing_rate_pre'])}")
    lines.append(f"  Underswing post rate: {f(sA['underswing_rate_post'])}")
    lines.append("")
    lines.append("LEFT")
    lines.append(f"  Pre avg:  {f(sL['pre_avg_deg'], '°')}")
    lines.append(f"  Post avg: {f(sL['post_avg_deg'], '°')}")
    lines.append(f"  Dist avg: {f(sL['dist_avg_m'], 'm')}")
    lines.append(f"  Dir dev:  {f(sL['dir_dev_avg_deg'], '°')}")
    lines.append("")
    lines.append("RIGHT")
    lines.append(f"  Pre avg:  {f(sR['pre_avg_deg'], '°')}")
    lines.append(f"  Post avg: {f(sR['post_avg_deg'], '°')}")
    lines.append(f"  Dist avg: {f(sR['dist_avg_m'], 'm')}")
    lines.append(f"  Dir dev:  {f(sR['dir_dev_avg_deg'], '°')}")
    lines.append("")
    lines.append("GRID 4x3 (dist avg meters)  layer 2(top)->0(bottom), col 0..3")
    grid = report["grid_4x3_dist"]
    for y in (2, 1, 0):
        row = []
        for x in range(4):
            row.append(f(grid[y][x]["dist_avg_m"], "m"))
        lines.append(f"  Layer {y}: " + " | ".join(row))
    lines.append("")
    lines.append("SCHEMA DEBUG (first few note items)")
    for i, s in enumerate(report["schema_debug"]["sample"]):
        lines.append(f"  item {i+1}: {s}")
    lines.append("")
    lines.append(f"Outputs saved in Documents\\{OUTPUT_ROOT}")
    return "\n".join(lines)


def save_outputs(report: Dict[str, Any], label: str) -> Tuple[Path, Path]:
    out_dir = make_output_folder()
    base = safe_name(label)
    json_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.json"
    txt_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    txt_path.write_text(build_human_txt(report), encoding="utf-8")
    return json_path, txt_path


# -----------------------------
# Worker thread
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

            self.log.emit("Building full-detail report...")
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
