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

APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v1.4 (full-report)"
OUTPUT_ROOT = "BeatSaberReplayAnalysis"

# -----------------------------
# Safe import of BSOR parser
# -----------------------------
MAKE_BSOR = None
_make_bsor_import_error = None
for modname in ("py_bsor", "pybsor", "bsor"):
    try:
        mod = __import__(modname, fromlist=["make_bsor"])
        if hasattr(mod, "make_bsor"):
            MAKE_BSOR = getattr(mod, "make_bsor")
            break
        # common pattern: from bsor.Bsor import make_bsor
        if hasattr(mod, "Bsor"):
            Bsor = getattr(mod, "Bsor")
            if hasattr(Bsor, "make_bsor"):
                MAKE_BSOR = getattr(Bsor, "make_bsor")
                break
    except Exception as e:
        _make_bsor_import_error = e


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
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except:
        return None

def obj_get(o: Any, *keys: str) -> Any:
    """
    Try multiple possible keys/attrs (replay schemas vary).
    Returns first match that isn't None.
    """
    for k in keys:
        if o is None:
            continue
        if isinstance(o, dict) and k in o and o[k] is not None:
            return o[k]
        if hasattr(o, k):
            v = getattr(o, k)
            if v is not None:
                return v
    return None

def extract_score_id(text: str) -> Optional[str]:
    if not text:
        return None
    # accepts:
    # https://replay.beatleader.com/?scoreId=30204803
    # scoreId=30204803
    # 30204803
    m = re.search(r"scoreId\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{6,})\b", text)
    if m:
        return m.group(1)
    return None


# -----------------------------
# BeatLeader replay download (best-effort)
# -----------------------------
def download_bsor_to_bytes_from_scoreid(score_id: str) -> bytes:
    """
    BeatLeader is a moving target. We try multiple approaches:
    1) Query score endpoint for replay/cdn link
    2) Try common replay endpoints
    3) As a last resort, hit replay.beatleader.com page and scrape bsor link
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": f"{APP_NAME} {APP_VERSION}"})

    # 1) score endpoint (if available)
    candidates: List[str] = []
    try:
        r = sess.get(f"https://api.beatleader.xyz/score/{score_id}", timeout=20)
        if r.ok:
            j = r.json()
            # hunt for replay-like fields
            for k in ("replay", "replayLink", "replayUrl", "replayURL", "cdn", "cdnLink", "cdnUrl"):
                v = j.get(k) if isinstance(j, dict) else None
                if isinstance(v, str) and v.startswith("http"):
                    candidates.append(v)
            # sometimes nested
            replay_obj = j.get("replay") if isinstance(j, dict) else None
            if isinstance(replay_obj, dict):
                for k in ("url", "link", "cdn", "cdnLink", "cdnUrl"):
                    v = replay_obj.get(k)
                    if isinstance(v, str) and v.startswith("http"):
                        candidates.append(v)
    except:
        pass

    # 2) common direct endpoints
    candidates += [
        f"https://api.beatleader.xyz/score/{score_id}/replay",
        f"https://api.beatleader.xyz/replay/{score_id}",
        f"https://cdn.beatleader.xyz/replays/{score_id}.bsor",
        f"https://replay.beatleader.com/replay/{score_id}.bsor",
    ]

    # 3) scrape replay page for any .bsor link
    try:
        page = sess.get(f"https://replay.beatleader.com/?scoreId={score_id}", timeout=20)
        if page.ok:
            m = re.search(r"https?://[^\s\"']+?\.bsor", page.text, re.IGNORECASE)
            if m:
                candidates.append(m.group(0))
    except:
        pass

    last_err = None
    for url in candidates:
        try:
            rr = sess.get(url, timeout=30, allow_redirects=True)
            if rr.status_code == 200 and rr.content and rr.content[:4] != b"<htm":
                return rr.content
            last_err = f"{url} -> HTTP {rr.status_code}"
        except Exception as e:
            last_err = f"{url} -> {e}"

    raise RuntimeError(f"Could not download BSOR for scoreId {score_id}. Last error: {last_err}")


# -----------------------------
# Scoring maths (Beat Saber rules)
# -----------------------------
def score_pre_swing(pre_angle_deg: float) -> float:
    # 0..70, max at 100 degrees
    return 70.0 * clamp(pre_angle_deg / 100.0, 0.0, 1.0)

def score_post_swing(post_angle_deg: float) -> float:
    # 0..30, max at 60 degrees
    return 30.0 * clamp(post_angle_deg / 60.0, 0.0, 1.0)

def score_accuracy(cut_dist_m: float) -> float:
    """
    0..15: perfect = 15 at 0m from centre, falls off toward ~0.
    The exact curve is game-specific; we keep a simple linear proxy:
    0.1m (~10cm) is "very off". Clamp.
    """
    d = abs(cut_dist_m)
    return 15.0 * clamp(1.0 - (d / 0.10), 0.0, 1.0)

@dataclass
class Cut:
    t: float
    hand: str               # "left" or "right"
    pre: Optional[float]
    post: Optional[float]
    acc_dist: Optional[float]   # meters (distance from centre) if available
    x: Optional[float]          # metres offset on cut plane (if available)
    y: Optional[float]
    dir_dev_deg: Optional[float]  # cut direction deviation (if available)


def extract_cuts(bsor_obj: Any) -> List[Cut]:
    """
    Try to pull note cut data out of whatever schema we get.
    We will harvest anything we can find and stay resilient.
    """
    cuts: List[Cut] = []

    # Common containers in BSOR decoders:
    # - "noteCuts"
    # - "notes" where each note has a "cut" subobject
    # - "cuts" list
    note_list = obj_get(bsor_obj, "noteCuts", "notes", "cuts") or []
    if not isinstance(note_list, list):
        return cuts

    for n in note_list:
        # time
        t = safe_float(obj_get(n, "time", "t", "songTime", "noteTime")) or 0.0

        # hand
        h = obj_get(n, "saberType", "hand", "color", "saber", "saber")  # varies
        hand = None
        if isinstance(h, str):
            if "left" in h.lower() or "red" in h.lower():
                hand = "left"
            elif "right" in h.lower() or "blue" in h.lower():
                hand = "right"
        if hand is None:
            # numeric codes sometimes: 0 = left, 1 = right
            hn = obj_get(n, "saberType", "hand", "saber")
            if isinstance(hn, (int, float)):
                hand = "left" if int(hn) == 0 else "right"
        if hand is None:
            hand = "unknown"

        cut_obj = obj_get(n, "cut", "noteCut", "cutInfo") or n

        pre = safe_float(obj_get(cut_obj, "preSwing", "preSwingAngle", "beforeCutAngle", "beforeAngle"))
        post = safe_float(obj_get(cut_obj, "postSwing", "postSwingAngle", "afterCutAngle", "afterAngle"))

        # distance from centre (meters) and/or x/y offsets
        dist = safe_float(obj_get(cut_obj, "cutDistanceToCenter", "distanceToCenter", "accuracyDistance"))
        x = safe_float(obj_get(cut_obj, "cutPointX", "x", "offsetX", "cutX"))
        y = safe_float(obj_get(cut_obj, "cutPointY", "y", "offsetY", "cutY"))

        # direction deviation (degrees)
        dir_dev = safe_float(obj_get(cut_obj, "cutDirDeviation", "directionDeviation", "dirDeviationDeg"))

        cuts.append(Cut(t=t, hand=hand, pre=pre, post=post, acc_dist=dist, x=x, y=y, dir_dev_deg=dir_dev))

    return cuts


def bin_by_time(cuts: List[Cut], window_s: float = 10.0) -> List[Dict[str, Any]]:
    if not cuts:
        return []
    max_t = max(c.t for c in cuts)
    bins = int(math.ceil(max_t / window_s))
    out: List[Dict[str, Any]] = []
    for i in range(bins):
        a = i * window_s
        b = (i + 1) * window_s
        seg = [c for c in cuts if a <= c.t < b]
        out.append(summarize_segment(seg, a, b))
    return out


def summarize_segment(cuts: List[Cut], t0: float, t1: float) -> Dict[str, Any]:
    def hand_filter(h: str) -> List[Cut]:
        return [c for c in cuts if c.hand == h]

    return {
        "t_start": t0,
        "t_end": t1,
        "all": summarize_cuts(cuts),
        "left": summarize_cuts(hand_filter("left")),
        "right": summarize_cuts(hand_filter("right")),
    }


def summarize_cuts(cuts: List[Cut]) -> Dict[str, Any]:
    pres = [c.pre for c in cuts if c.pre is not None]
    posts = [c.post for c in cuts if c.post is not None]
    dists = [c.acc_dist for c in cuts if c.acc_dist is not None]
    xs = [c.x for c in cuts if c.x is not None]
    ys = [c.y for c in cuts if c.y is not None]
    devs = [c.dir_dev_deg for c in cuts if c.dir_dev_deg is not None]

    # underswing: pre < 100deg or post < 60deg (simple definition)
    under_pre = [c for c in cuts if c.pre is not None and c.pre < 100.0]
    under_post = [c for c in cuts if c.post is not None and c.post < 60.0]

    out: Dict[str, Any] = {
        "n": len(cuts),
        "pre_avg_deg": mean(pres) if pres else None,
        "pre_std_deg": stdev(pres) if pres else None,
        "post_avg_deg": mean(posts) if posts else None,
        "post_std_deg": stdev(posts) if posts else None,
        "dir_dev_avg_deg": mean(devs) if devs else None,
        "dir_dev_std_deg": stdev(devs) if devs else None,

        "center_bias_x_m": mean(xs) if xs else None,
        "center_bias_y_m": mean(ys) if ys else None,
        "center_std_x_m": stdev(xs) if xs else None,
        "center_std_y_m": stdev(ys) if ys else None,

        "acc_dist_avg_m": mean(dists) if dists else None,
        "acc_dist_std_m": stdev(dists) if dists else None,

        "underswing_rate_pre": (len(under_pre) / len([c for c in cuts if c.pre is not None])) if any(c.pre is not None for c in cuts) else None,
        "underswing_rate_post": (len(under_post) / len([c for c in cuts if c.post is not None])) if any(c.post is not None for c in cuts) else None,
    }

    # Score proxies if data exists
    if pres and posts:
        pre_scores = [score_pre_swing(p) for p in pres]
        post_scores = [score_post_swing(p) for p in posts]
        out["pre_score_avg"] = mean(pre_scores)
        out["post_score_avg"] = mean(post_scores)
        out["pre_percent_of_max"] = (out["pre_score_avg"] / 70.0) * 100.0 if out["pre_score_avg"] is not None else None
        out["post_percent_of_max"] = (out["post_score_avg"] / 30.0) * 100.0 if out["post_score_avg"] is not None else None

    if dists:
        acc_scores = [score_accuracy(d) for d in dists]
        out["acc_score_avg"] = mean(acc_scores)
        out["acc_percent_of_max"] = (out["acc_score_avg"] / 15.0) * 100.0 if out["acc_score_avg"] is not None else None

    return out


def recommend_settings(summary_left: Dict[str, Any], summary_right: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT: These are suggestions based on consistent bias.
    Convention used:
    - If your cuts land +X (to the right) on average, suggest moving saber Position X negative (left) to re-center.
    - If your cuts land +Y (above centre), suggest Position Y negative (down).
    Values are conservative: 50% of bias converted to cm.
    """
    def rec_for(hand_sum: Dict[str, Any]) -> Dict[str, Any]:
        x = hand_sum.get("center_bias_x_m")
        y = hand_sum.get("center_bias_y_m")
        rec = {"position_x_cm": 0.0, "position_y_cm": 0.0, "notes": []}

        if x is not None:
            # move opposite direction, convert m -> cm, apply 0.5 gain
            dx_cm = -x * 100.0 * 0.5
            rec["position_x_cm"] = round(dx_cm, 2)
            rec["notes"].append(f"Cut-centre X bias = {x*100:.2f}cm → suggest Position X {dx_cm:+.2f}cm")
        else:
            rec["notes"].append("No cut-centre X data in replay (can't suggest Position X).")

        if y is not None:
            dy_cm = -y * 100.0 * 0.5
            rec["position_y_cm"] = round(dy_cm, 2)
            rec["notes"].append(f"Cut-centre Y bias = {y*100:.2f}cm → suggest Position Y {dy_cm:+.2f}cm")
        else:
            rec["notes"].append("No cut-centre Y data in replay (can't suggest Position Y).")

        # rotation suggestions require direction deviation data
        dev = hand_sum.get("dir_dev_avg_deg")
        if dev is not None:
            rec["rotation_y_deg"] = round(-dev * 0.5, 2)
            rec["notes"].append(f"Avg cut direction deviation = {dev:.2f}° → suggest Rotation Y {(-dev*0.5):+.2f}° (conservative)")
        else:
            rec["rotation_y_deg"] = 0.0
            rec["notes"].append("No cut-direction deviation data (can't suggest Rotation Y).")

        return rec

    return {
        "left": rec_for(summary_left),
        "right": rec_for(summary_right),
        "disclaimer": "Suggestions are conservative. Apply half first, re-test, then iterate.",
        "axis_key": {
            "PositionX": "left/right (cm)",
            "PositionY": "up/down (cm)",
            "PositionZ": "forward/back (cm)",
            "RotationX": "tilt up/down (deg)",
            "RotationY": "turn left/right (deg)",
            "RotationZ": "roll/twist (deg)"
        }
    }


def build_human_report(payload: Dict[str, Any]) -> str:
    def fmt(v, unit=""):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.3f}{unit}"
        return f"{v}{unit}"

    L = payload["summary"]["left"]
    R = payload["summary"]["right"]

    rec = payload.get("recommendations", {})
    recL = rec.get("left", {})
    recR = rec.get("right", {})

    lines = []
    lines.append(f"{APP_NAME} {APP_VERSION}")
    lines.append("")
    lines.append("KEY TERMS (plain English)")
    lines.append("- PRE swing: your backswing before contact (max points at ~100°).")
    lines.append("- POST swing: your follow-through after contact (max points at ~60°).")
    lines.append("- Accuracy distance: how far from block centre you cut (closer = better).")
    lines.append("- Underswing: not reaching full swing thresholds (costs points).")
    lines.append("")

    lines.append("SUMMARY (ALL NOTES)")
    lines.append(f"Total cuts parsed: {payload['summary']['all'].get('n', 0)}")
    lines.append("")

    def hand_block(name, H):
        lines.append(f"{name.upper()} HAND")
        lines.append(f"PRE avg:  {fmt(H.get('pre_avg_deg'), '°')}   (score proxy: {fmt(H.get('pre_percent_of_max'), '%')} of max)")
        lines.append(f"POST avg: {fmt(H.get('post_avg_deg'), '°')}   (score proxy: {fmt(H.get('post_percent_of_max'), '%')} of max)")
        lines.append(f"ACC dist avg: {fmt(H.get('acc_dist_avg_m'), 'm')}  (~{fmt((H.get('acc_dist_avg_m') or 0)*100, 'cm')})")
        lines.append(f"Underswing PRE rate:  {fmt(H.get('underswing_rate_pre'))}")
        lines.append(f"Underswing POST rate: {fmt(H.get('underswing_rate_post'))}")
        lines.append(f"Centre bias X: {fmt(H.get('center_bias_x_m'), 'm')}  (~{fmt((H.get('center_bias_x_m') or 0)*100, 'cm')})")
        lines.append(f"Centre bias Y: {fmt(H.get('center_bias_y_m'), 'm')}  (~{fmt((H.get('center_bias_y_m') or 0)*100, 'cm')})")
        lines.append("")

    hand_block("Left", L)
    hand_block("Right", R)

    lines.append("RECOMMENDED IN-GAME CONTROLLER TWEAKS (conservative first pass)")
    lines.append("Apply HALF, re-test 1 song, then iterate.")
    lines.append("")
    lines.append("LEFT:")
    lines.append(f"- Position X: {recL.get('position_x_cm', 0.0):+.2f} cm")
    lines.append(f"- Position Y: {recL.get('position_y_cm', 0.0):+.2f} cm")
    lines.append(f"- Rotation Y: {recL.get('rotation_y_deg', 0.0):+.2f} °")
    for n in recL.get("notes", []):
        lines.append(f"  • {n}")
    lines.append("")
    lines.append("RIGHT:")
    lines.append(f"- Position X: {recR.get('position_x_cm', 0.0):+.2f} cm")
    lines.append(f"- Position Y: {recR.get('position_y_cm', 0.0):+.2f} cm")
    lines.append(f"- Rotation Y: {recR.get('rotation_y_deg', 0.0):+.2f} °")
    for n in recR.get("notes", []):
        lines.append(f"  • {n}")
    lines.append("")
    lines.append("TIME DEPENDENCY (10s windows): see report.json -> time_windows[]")
    lines.append("If your scores dip late, it's usually fatigue or grip instability rather than raw settings.")
    lines.append("")
    lines.append("Axis Key:")
    ax = payload.get("recommendations", {}).get("axis_key", {})
    for k, v in ax.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def summarize_bsor(bsor_obj: Any) -> Dict[str, Any]:
    cuts = extract_cuts(bsor_obj)

    summary_all = summarize_cuts(cuts)
    summary_left = summarize_cuts([c for c in cuts if c.hand == "left"])
    summary_right = summarize_cuts([c for c in cuts if c.hand == "right"])

    payload: Dict[str, Any] = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "meta": {
            "parsed_at": datetime.datetime.now().isoformat(),
        },
        "summary": {
            "all": summary_all,
            "left": summary_left,
            "right": summary_right,
        },
        "time_windows": bin_by_time(cuts, window_s=10.0),
    }

    payload["recommendations"] = recommend_settings(summary_left, summary_right)

    return payload


def save_outputs(payload: Dict[str, Any], label: str) -> Tuple[Path, Path]:
    out_dir = make_output_folder()
    base = re.sub(r"[^\w\-. ]+", "_", label).strip()
    json_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.json"
    txt_path = out_dir / f"{base} - {APP_NAME} {APP_VERSION}.txt"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(build_human_report(payload), encoding="utf-8")

    return json_path, txt_path


# -----------------------------
# Worker thread
# -----------------------------
class Worker(QThread):
    log = Signal(str)
    success = Signal(str)
    error = Signal(str)

    def __init__(self, bsor_path: Optional[str], score_input: Optional[str]):
        super().__init__()
        self.bsor_path = bsor_path
        self.score_input = score_input

    def run(self):
        try:
            if MAKE_BSOR is None:
                raise RuntimeError(f"BSOR parser not available. Import error: {_make_bsor_import_error}")

            bsor_obj = None
            label = None

            if self.bsor_path:
                self.log.emit(f"Loaded file: {self.bsor_path}")
                label = Path(self.bsor_path).stem
                self.log.emit("Parsing BSOR file...")
                with open(self.bsor_path, "rb") as f:
                    bsor_obj = MAKE_BSOR(f)

            else:
                sid = extract_score_id(self.score_input or "")
                if not sid:
                    raise RuntimeError("No .bsor loaded, and replay link / scoreId is missing or invalid.")
                self.log.emit(f"ScoreId detected: {sid}")
                self.log.emit("Downloading BSOR...")
                bsor_bytes = download_bsor_to_bytes_from_scoreid(sid)
                self.log.emit(f"Downloaded {len(bsor_bytes)} bytes")
                self.log.emit("Parsing BSOR bytes...")
                import io
                bsor_obj = MAKE_BSOR(io.BytesIO(bsor_bytes))
                label = f"scoreId_{sid}"

            self.log.emit("Building full report...")
            payload = summarize_bsor(bsor_obj)

            json_path, txt_path = save_outputs(payload, label)
            self.success.emit(f"Saved:\n{json_path}\n{txt_path}")

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")


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

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background: rgba(255,255,255,0.04); border-radius: 12px;")
        self.log.setMinimumHeight(200)
        left.addWidget(QLabel("Log"))
        left.addWidget(self.log)

        right.addWidget(QLabel("Replay link / scoreId"))
        self.replay_input = QLineEdit()
        self.replay_input.setPlaceholderText("Paste BeatLeader replay link (e.g. https://replay.beatleader.com/?scoreId=...) or just the scoreId")
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

        if MAKE_BSOR is None:
            self.append_log(f"WARNING: BSOR parser not imported. Import error: {_make_bsor_import_error}")

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

        score_input = self.replay_input.text().strip()
        self.append_log("---- Analyse clicked ----")

        self.worker = Worker(self.bsor_path, score_input)
        self.worker.log.connect(self.append_log)
        self.worker.success.connect(self.on_success)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_success(self, msg: str):
        self.analyse_btn.setEnabled(True)
        self.choose_btn.setEnabled(True)
        self.status.setText("Done.")
        QMessageBox.information(self, "Success", f"Report saved:\n\n{msg}")

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
