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

# py-bsor (PyPI)  ✅ NO Scoring import (calc_stats removed)
from bsor.Bsor import make_bsor


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v2.1 (stable-no-scoring)"
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
    url = f"https://cdn.beatleader.xyz/replays/{score_id}.bsor"
    r = requests.get(url, timeout=30)
    if not r.ok or not r.content:
        raise RuntimeError(f"Failed to download BSOR from BeatLeader CDN (HTTP {r.status_code})")
    return r.content


# -----------------------------
# Math helpers
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

def as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except:
        return None

def as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
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
# Data model
# -----------------------------
@dataclass
class Cut:
    t: float
    hand: str                      # "left" / "right"
    pre: Optional[float]           # degrees
    post: Optional[float]          # degrees
    dist: Optional[float]          # meters (distance to center)
    cut_x: Optional[float]         # cut point x (if available)
    cut_y: Optional[float]         # cut point y (if available)
    dir_dev: Optional[float]       # degrees
    grid_x: Optional[int]          # 0..3
    grid_y: Optional[int]          # 0..2


def iter_notes(bsor_obj: Any) -> List[Any]:
    """
    py-bsor objects differ by version.
    Prefer bsor_obj.notes if present, fallback to bsor_obj.noteCuts.
    """
    notes = getattr(bsor_obj, "notes", None)
    if isinstance(notes, list) and notes:
        return notes
    note_cuts = getattr(bsor_obj, "noteCuts", None)
    if isinstance(note_cuts, list) and note_cuts:
        return note_cuts
    return []


def extract_cuts(bsor_obj: Any) -> List[Cut]:
    cuts: List[Cut] = []
    notes = iter_notes(bsor_obj)

    for n in notes:
        if isinstance(n, dict):
            t = as_float(get_dict(n, "time", "t", "songTime")) or 0.0
            saber = as_int(get_dict(n, "saberType", "hand"))
            pre = as_float(get_dict(n, "preSwing", "beforeCutAngle", "beforeAngle"))
            post = as_float(get_dict(n, "postSwing", "afterCutAngle", "afterAngle"))
            dist = as_float(get_dict(n, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
            cut_x = as_float(get_dict(n, "cutPointX", "x", "offsetX", "cutX"))
            cut_y = as_float(get_dict(n, "cutPointY", "y", "offsetY", "cutY"))
            dev = as_float(get_dict(n, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))
            gx = as_int(get_dict(n, "lineIndex", "line", "gridX", "x"))
            gy = as_int(get_dict(n, "lineLayer", "layer", "gridY", "y"))
        else:
            t = as_float(get_attr(n, "time", "t", "songTime")) or 0.0
            saber = as_int(get_attr(n, "saberType", "hand"))
            # some versions store a cut object with before/after
            cut_obj = get_attr(n, "cut", "noteCut", "cutInfo")

            pre = as_float(get_attr(n, "preSwing", "beforeCutAngle", "beforeAngle"))
            post = as_float(get_attr(n, "postSwing", "afterCutAngle", "afterAngle"))
            dist = as_float(get_attr(n, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
            cut_x = as_float(get_attr(n, "cutPointX", "x", "offsetX", "cutX"))
            cut_y = as_float(get_attr(n, "cutPointY", "y", "offsetY", "cutY"))
            dev = as_float(get_attr(n, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))
            gx = as_int(get_attr(n, "lineIndex", "line", "gridX", "x"))
            gy = as_int(get_attr(n, "lineLayer", "layer", "gridY", "y"))

            # if missing, try the cut object
            if cut_obj is not None:
                if pre is None:
                    pre = as_float(get_attr(cut_obj, "preSwing", "beforeCutAngle", "beforeAngle"))
                if post is None:
                    post = as_float(get_attr(cut_obj, "postSwing", "afterCutAngle", "afterAngle"))
                if dist is None:
                    dist = as_float(get_attr(cut_obj, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
                if cut_x is None:
                    cut_x = as_float(get_attr(cut_obj, "cutPointX", "x", "offsetX", "cutX"))
                if cut_y is None:
                    cut_y = as_float(get_attr(cut_obj, "cutPointY", "y", "offsetY", "cutY"))
                if dev is None:
                    dev = as_float(get_attr(cut_obj, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))

        hand = "left" if saber == 0 else ("right" if saber == 1 else "unknown")

        # sanity clamp grid
        if gx is not None and not (0 <= gx <= 3):
            gx = None
        if gy is not None and not (0 <= gy <= 2):
            gy = None

        cuts.append(Cut(
            t=t, hand=hand, pre=pre, post=post, dist=dist,
            cut_x=cut_x, cut_y=cut_y, dir_dev=dev,
            grid_x=gx, grid_y=gy
        ))

    return cuts


# -----------------------------
# Summaries
# -----------------------------
def summarize(cuts: List[Cut]) -> Dict[str, Any]:
    pres = [c.pre for c in cuts if c.pre is not None]
    posts = [c.post for c in cuts if c.post is not None]
    dists = [c.dist for c in cuts if c.dist is not None]
    xs = [c.cut_x for c in cuts if c.cut_x is not None]
    ys = [c.cut_y for c in cuts if c.cut_y is not None]
    devs = [c.dir_dev for c in cuts if c.dir_dev is not None]

    under_pre = [c for c in cuts if c.pre is not None and c.pre < 100.0]
    under_post = [c for c in cuts if c.post is not None and c.post < 60.0]

    return {
        "n": len(cuts),
        "pre_avg_deg": mean(pres),
        "pre_std_deg": stdev(pres),
        "post_avg_deg": mean(posts),
        "post_std_deg": stdev(posts),
        "dist_avg_m": mean(dists),
        "dist_std_m": stdev(dists),
        "cut_bias_x": mean(xs),          # only if replay provides cut points
        "cut_bias_y": mean(ys),
        "cut_std_x": stdev(xs),
        "cut_std_y": stdev(ys),
        "dir_dev_avg_deg": mean(devs),
        "dir_dev_std_deg": stdev(devs),
        "underswing_rate_pre": (len(under_pre) / len(pres)) if pres else None,
        "underswing_rate_post": (len(under_post) / len(posts)) if posts else None,
    }


def grid_4x3(cuts: List[Cut]) -> List[List[Dict[str, Any]]]:
    # [layer][col] (layer 0 bottom, 2 top)
    acc: List[List[List[float]]] = [[[] for _ in range(4)] for _ in range(3)]
    for c in cuts:
        if c.grid_x is None or c.grid_y is None or c.dist is None:
            continue
        acc[c.grid_y][c.grid_x].append(c.dist)

    out: List[List[Dict[str, Any]]] = []
    for y in range(3):
        row = []
        for x in range(4):
            arr = acc[y][x]
            row.append({"n": len(arr), "dist_avg_m": mean(arr), "dist_std_m": stdev(arr)})
        out.append(row)
    return out


def time_windows(cuts: List[Cut], window_s: float = 10.0) -> List[Dict[str, Any]]:
    if not cuts:
        return []
    max_t = max(c.t for c in cuts)
    bins = int(math.ceil(max_t / window_s))
    out: List[Dict[str, Any]] = []

    for i in range(bins):
        a = i * window_s
        b = (i + 1) * window_s
        seg = [c for c in cuts if a <= c.t < b]
        out.append({
            "t_start": a,
            "t_end": b,
            "all": summarize(seg),
            "left": summarize([c for c in seg if c.hand == "left"]),
            "right": summarize([c for c in seg if c.hand == "right"]),
        })
    return out


def recommend_from_bias(summary_hand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only works if replay exposes cut_x/cut_y.
    Convention:
      +X means you hit to the right of center -> move controller PositionX negative (left)
      +Y means you hit above center -> move PositionY negative (down)
    We apply 50% gain so you don’t overcorrect.
    """
    bx = summary_hand.get("cut_bias_x")
    by = summary_hand.get("cut_bias_y")
    dev = summary_hand.get("dir_dev_avg_deg")

    rec = {
        "pos_x_cm": None,
        "pos_y_cm": None,
        "rot_y_deg": None,
        "notes": []
    }

    if bx is not None:
        dx = -bx * 100.0 * 0.5
        rec["pos_x_cm"] = round(dx, 2)
        rec["notes"].append(f"cut_bias_x={bx:+.4f} → suggest PositionX {dx:+.2f} cm")
    else:
        rec["notes"].append("No cut_bias_x in replay (can’t suggest PositionX).")

    if by is not None:
        dy = -by * 100.0 * 0.5
        rec["pos_y_cm"] = round(dy, 2)
        rec["notes"].append(f"cut_bias_y={by:+.4f} → suggest PositionY {dy:+.2f} cm")
    else:
        rec["notes"].append("No cut_bias_y in replay (can’t suggest PositionY).")

    if dev is not None:
        ry = -dev * 0.5
        rec["rot_y_deg"] = round(ry, 2)
        rec["notes"].append(f"dir_dev_avg={dev:+.2f}° → suggest RotationY {ry:+.2f}°")
    else:
        rec["notes"].append("No dir_dev_avg in replay (can’t suggest RotationY).")

    return rec


def build_report(bsor_obj: Any, label: str) -> Dict[str, Any]:
    cuts = extract_cuts(bsor_obj)
    L = [c for c in cuts if c.hand == "left"]
    R = [c for c in cuts if c.hand == "right"]

    sum_all = summarize(cuts)
    sum_L = summarize(L)
    sum_R = summarize(R)

    report = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "meta": {
            "label": label,
            "parsed_at": datetime.datetime.now().isoformat(),
            "cuts_parsed": len(cuts),
            "source": "py-bsor make_bsor (no Scoring/calc_stats)",
        },
        "summary": {
            "all": sum_all,
            "left": sum_L,
            "right": sum_R,
        },
        "grid_4x3": grid_4x3(cuts),
        "time_windows_10s": time_windows(cuts, window_s=10.0),
        "recommendations": {
            "note": "Recommendations only appear when replay provides cut-point bias. Apply HALF first, re-test, then iterate.",
            "left": recommend_from_bias(sum_L),
            "right": recommend_from_bias(sum_R),
            "axis_key": {
                "PositionX": "left/right (cm)",
                "PositionY": "up/down (cm)",
                "PositionZ": "forward/back (cm)",
                "RotationX": "tilt up/down (deg)",
                "RotationY": "yaw turn (deg)",
                "RotationZ": "roll twist (deg)",
            }
        },
        "glossary": {
            "pre_avg_deg": "Backswing angle before hit. Higher is better until scoring cap (~100°).",
            "post_avg_deg": "Follow-through after hit. Higher is better until scoring cap (~60°).",
            "dist_avg_m": "Avg distance from block center (meters). Lower is better.",
            "underswing_rate_pre": "Fraction of notes with pre < 100° (simple underswing measure).",
            "underswing_rate_post": "Fraction of notes with post < 60° (simple underswing measure).",
            "cut_bias_x/y": "Average cutpoint bias if replay provides it (used for calibration).",
            "dir_dev_avg_deg": "Average cut direction deviation if replay provides it.",
        }
    }

    return report


def build_human_txt(report: Dict[str, Any]) -> str:
    L = report["summary"]["left"]
    R = report["summary"]["right"]

    def f(v, unit=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}{unit}"
        return f"{v}{unit}"

    lines = []
    lines.append(f"{APP_NAME} {APP_VERSION}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append(f"- Cuts parsed: {report['meta']['cuts_parsed']}")
    lines.append("")
    lines.append("LEFT HAND")
    lines.append(f"- Pre avg: {f(L.get('pre_avg_deg'), '°')}  | Post avg: {f(L.get('post_avg_deg'), '°')}")
    lines.append(f"- Dist avg: {f(L.get('dist_avg_m'), ' m')} (lower=better)")
    lines.append(f"- Underswing pre: {f(L.get('underswing_rate_pre'))} | Underswing post: {f(L.get('underswing_rate_post'))}")
    lines.append(f"- Bias X/Y: {f(L.get('cut_bias_x'))} / {f(L.get('cut_bias_y'))}")
    lines.append("")
    lines.append("RIGHT HAND")
    lines.append(f"- Pre avg: {f(R.get('pre_avg_deg'), '°')}  | Post avg: {f(R.get('post_avg_deg'), '°')}")
    lines.append(f"- Dist avg: {f(R.get('dist_avg_m'), ' m')} (lower=better)")
    lines.append(f"- Underswing pre: {f(R.get('underswing_rate_pre'))} | Underswing post: {f(R.get('underswing_rate_post'))}")
    lines.append(f"- Bias X/Y: {f(R.get('cut_bias_x'))} / {f(R.get('cut_bias_y'))}")
    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append(report["recommendations"]["note"])
    for side in ("left", "right"):
        rec = report["recommendations"][side]
        lines.append(f"{side.upper()}:")
        lines.append(f"  PositionX: {rec.get('pos_x_cm')}")
        lines.append(f"  PositionY: {rec.get('pos_y_cm')}")
        lines.append(f"  RotationY: {rec.get('rot_y_deg')}")
        for n in rec.get("notes", []):
            lines.append(f"   - {n}")
    lines.append("")
    lines.append("Saved in: Documents\\" + OUTPUT_ROOT)

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

            self.log.emit("Building report (stable-no-scoring)...")
            report = build_report(bsor_obj, label)

            self.log.emit("Writing outputs...")
            jp, tp = save_outputs(report, label)

            self.success.emit(f"Saved:\n{jp}\n{tp}")

        except Exception as e:
            tb = traceback.format_exc()
            crash_path = write_crash_log(tb)
            self.error.emit(f"{e}\n\nCrash log saved to:\n{crash_path}\n\n{tb}")


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
