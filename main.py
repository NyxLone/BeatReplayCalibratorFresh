import os
import re
import json
import io
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import requests

from PySide6.QtCore import Qt, QMimeData, Signal, QObject
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QFrame,
    QTextEdit,
)


APP_NAME = "Beat Replay Calibrator"
APP_VERSION = "v1.3 (stream-fix)"
OUTPUT_ROOT = "BeatSaberReplayAnalysis"

# ---- BSOR parser import (py-bsor) ----
# IMPORTANT: py-bsor's make_bsor expects a file-like object (has .read()).
# Passing a string path causes: "'str' object has no attribute 'read'"
MAKE_BSOR = None
_bsor_import_error = None
try:
    from bsor.Bsor import make_bsor as MAKE_BSOR  # py-bsor
except Exception as e:
    _bsor_import_error = e
    MAKE_BSOR = None


def documents_dir() -> Path:
    # Windows "Documents" folder
    return Path.home() / "Documents"


def ensure_output_root() -> Path:
    base = documents_dir() / OUTPUT_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base


def safe_filename(name: str) -> str:
    # Keep it Windows-friendly
    name = re.sub(r"[^\w\-. ()]+", "_", name)
    name = name.strip().strip(".")
    return name[:180] if len(name) > 180 else name


def extract_score_id(text: str) -> Optional[int]:
    """
    Accepts:
      - https://replay.beatleader.com/?scoreId=30204803
      - https://replay.beatleader.xyz/?scoreId=30204803
      - scoreId=30204803
      - plain 30204803
    """
    if not text:
        return None

    text = text.strip()
    m = re.search(r"scoreId\s*=\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    if re.fullmatch(r"\d{5,}", text):
        return int(text)

    return None


def find_cdn_bsor_link_in_html(html: str) -> Optional[str]:
    """
    The BeatLeader web replay viewer supports passing a direct CDN link, e.g.:
      https://cdn.replays.beatleader.xyz/<scoreId>-<playerId>-... .bsor
    We'll scrape any such link from the page if present.
    """
    if not html:
        return None
    m = re.search(r"https://cdn\.replays\.beatleader\.[a-z]+/[^\s\"']+\.bsor", html)
    return m.group(0) if m else None


def try_get_replay_cdn_link(score_id: int) -> Tuple[Optional[str], str]:
    """
    Best effort strategy:
      1) Try BeatLeader API (may change; handle failures)
      2) Fallback: fetch replay viewer HTML and scrape CDN link
    Returns (link, debug_note)
    """
    # 1) Try API guess
    # If this endpoint changes, it will fail gracefully and we fall back.
    api_urls = [
        f"https://api.beatleader.xyz/score/{score_id}",
        f"https://api.beatleader.xyz/score/{score_id}/",
    ]
    for url in api_urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                # Try common fields
                # Some APIs provide replay as a nested object or direct URL
                for key_path in [
                    ("replay", "link"),
                    ("replay", "url"),
                    ("replay",),
                    ("replayLink",),
                    ("replayUrl",),
                    ("replayURL",),
                ]:
                    cur = data
                    ok = True
                    for k in key_path:
                        if isinstance(cur, dict) and k in cur:
                            cur = cur[k]
                        else:
                            ok = False
                            break
                    if ok and isinstance(cur, str) and ".bsor" in cur:
                        return cur, f"Found replay via API: {url} ({key_path})"

                # Sometimes API gives file name pieces, so we still fall back
        except Exception:
            pass

    # 2) Fallback: scrape replay page
    viewer_urls = [
        f"https://replay.beatleader.com/?scoreId={score_id}",
        f"https://replay.beatleader.xyz/?scoreId={score_id}",
    ]
    for vurl in viewer_urls:
        try:
            r = requests.get(vurl, timeout=20)
            if r.status_code == 200:
                cdn = find_cdn_bsor_link_in_html(r.text)
                if cdn:
                    return cdn, f"Scraped CDN replay link from: {vurl}"
        except Exception:
            pass

    return None, "Could not resolve CDN replay link (API + scrape failed)"


def download_bsor_to_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed ({r.status_code}).")
    return r.content


def parse_bsor_bytes(bsor_bytes: bytes):
    if MAKE_BSOR is None:
        raise RuntimeError(
            "BSOR parser not available. Install dependencies (py-bsor). "
            f"Import error: {_bsor_import_error}"
        )
    # make_bsor expects a file-like object (readable stream)
    bio = io.BytesIO(bsor_bytes)
    return MAKE_BSOR(bio)


def parse_bsor_file(path: Path):
    if MAKE_BSOR is None:
        raise RuntimeError(
            "BSOR parser not available. Install dependencies (py-bsor). "
            f"Import error: {_bsor_import_error}"
        )
    # IMPORTANT: pass a file object, not the path string
    with path.open("rb") as f:
        return MAKE_BSOR(f)


def summarize_bsor(bsor_obj) -> dict:
    """
    Keep this deliberately SIMPLE + readable.
    We’ll dump:
      - basic metadata (if present)
      - counts (frames, notes, walls)
      - any cut/cutDistance-like stats we can find
    The structure of bsor objects can vary by library/version, so we do safe probing.
    """
    def get_attr(o, name, default=None):
        return getattr(o, name, default)

    summary = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "bsor": {"parser": "py-bsor"},
        "meta": {},
        "counts": {},
        "notes": {},
        "hands": {},
        "debug": {},
    }

    # Try metadata-ish
    for k in ["info", "Info", "metadata", "MetaData", "metaData", "meta"]:
        obj = get_attr(bsor_obj, k, None)
        if obj is not None:
            summary["meta"]["raw_meta_field"] = k
            summary["meta"]["raw_meta_preview"] = str(obj)[:300]
            break

    # Try common fields
    # Different bsor libs expose different names; we probe gently.
    frames = None
    for k in ["frames", "Frames", "replayFrames", "ReplayFrames"]:
        frames = get_attr(bsor_obj, k, None)
        if frames is not None:
            break

    notes = None
    for k in ["notes", "Notes", "noteEvents", "NoteEvents"]:
        notes = get_attr(bsor_obj, k, None)
        if notes is not None:
            break

    walls = None
    for k in ["walls", "Walls", "wallEvents", "WallEvents"]:
        walls = get_attr(bsor_obj, k, None)
        if walls is not None:
            break

    summary["counts"]["frames"] = len(frames) if hasattr(frames, "__len__") else None
    summary["counts"]["notes"] = len(notes) if hasattr(notes, "__len__") else None
    summary["counts"]["walls"] = len(walls) if hasattr(walls, "__len__") else None

    # Now: pull some useful “calibration-ish” numbers if possible.
    # We look for per-note cut distance / deviation / cutPoint type fields.
    # This won’t be perfect yet, but it gives us something reliable to iterate.
    cut_distances = []
    time_deviations = []

    if notes and hasattr(notes, "__iter__"):
        for n in notes:
            # probe common cut-related attrs
            for cand in ["cutDistanceToCenter", "cutDistance", "distanceToCenter", "cutPoint", "cutPointDeviation"]:
                v = get_attr(n, cand, None)
                if isinstance(v, (int, float)):
                    cut_distances.append(float(v))
                    break

            for cand in ["timeDeviation", "td", "TimeDeviation"]:
                v = get_attr(n, cand, None)
                if isinstance(v, (int, float)):
                    time_deviations.append(float(v))
                    break

    def stats(vals):
        if not vals:
            return None
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        med = vals_sorted[n // 2] if n % 2 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2
        return {
            "n": n,
            "mean": mean,
            "median": med,
            "p90": vals_sorted[int(0.9 * (n - 1))],
            "p99": vals_sorted[int(0.99 * (n - 1))],
        }

    summary["notes"]["cut_distance"] = stats(cut_distances)
    summary["notes"]["time_deviation"] = stats(time_deviations)

    return summary


def save_report(report: dict, label: str) -> Path:
    out_root = ensure_output_root()
    safe = safe_filename(label)
    out_path = out_root / f"{safe} - {APP_NAME} {APP_VERSION}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out_path


# ---------------- UI ----------------

class WorkerSignals(QObject):
    success = Signal(str)
    error = Signal(str)
    log = Signal(str)


class DropZone(QFrame):
    dropped_file = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")
        self.setStyleSheet("""
            QFrame#DropZone {
                border: 2px dashed #666;
                border-radius: 12px;
                padding: 18px;
                background: rgba(255,255,255,0.02);
            }
        """)
        layout = QVBoxLayout(self)
        self.label = QLabel("Drop a .bsor replay here\n(or click 'Choose File')")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #ddd; font-size: 14px;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, event):
        md: QMimeData = event.mimeData()
        if md.hasUrls():
            # allow .bsor
            for url in md.urls():
                if url.toLocalFile().lower().endswith(".bsor"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        md: QMimeData = event.mimeData()
        if md.hasUrls():
            for url in md.urls():
                p = url.toLocalFile()
                if p.lower().endswith(".bsor"):
                    self.dropped_file.emit(p)
                    break
        event.acceptProposedAction()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.setMinimumWidth(860)
        self.setStyleSheet("background: #111; color: #eee;")

        title = QLabel(APP_NAME)
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))

        subtitle = QLabel(
            "Drag & drop a .bsor, OR paste a BeatLeader replay link / scoreId.\n"
            "Outputs a simple JSON report in Documents\\BeatSaberReplayAnalysis."
        )
        subtitle.setStyleSheet("color: #bbb;")

        self.drop = DropZone()
        self.drop.dropped_file.connect(self.on_file_dropped)

        self.link = QLineEdit()
        self.link.setPlaceholderText("Paste BeatLeader replay link or scoreId (e.g. https://replay.beatleader.com/?scoreId=30204803)")
        self.link.setStyleSheet("padding: 10px; border-radius: 10px; background: #1b1b1b; color: #eee;")

        self.choose = QPushButton("Choose File…")
        self.choose.clicked.connect(self.choose_file)

        self.analyse = QPushButton("Analyse")
        self.analyse.setStyleSheet("padding: 10px; font-weight: bold;")
        self.analyse.clicked.connect(self.run_analysis)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #9fd;")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#0c0c0c; border:1px solid #222; border-radius:10px; padding:10px; color:#ccc;")
        self.log.setMinimumHeight(140)

        left = QVBoxLayout()
        left.addWidget(self.drop)

        right = QVBoxLayout()
        right.addWidget(QLabel("Replay link / scoreId"))
        right.addWidget(self.link)
        right.addSpacing(8)
        right.addWidget(self.choose)
        right.addWidget(self.analyse)
        right.addStretch()
        right.addWidget(self.status)

        top = QHBoxLayout()
        top.addLayout(left, 2)
        top.addLayout(right, 1)

        root = QVBoxLayout(self)
        root.addWidget(title)
        root.addWidget(subtitle)
        root.addSpacing(8)
        root.addLayout(top)
        root.addSpacing(8)
        root.addWidget(QLabel("Log"))
        root.addWidget(self.log)

        self.current_file: Optional[Path] = None
        self.current_label: str = "Replay"

        self.signals = WorkerSignals()
        self.signals.success.connect(self.on_success)
        self.signals.error.connect(self.on_error)
        self.signals.log.connect(self.append_log)

        if MAKE_BSOR is None:
            self.append_log("WARNING: py-bsor not imported. The build may be missing dependencies.")


    def append_log(self, msg: str):
        self.log.append(msg)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select BSOR replay", "", "Beat Saber Replay (*.bsor)")
        if path:
            self.set_file(Path(path))

    def on_file_dropped(self, path: str):
        self.set_file(Path(path))

    def set_file(self, p: Path):
        self.current_file = p
        self.current_label = p.stem
        self.drop.label.setText(f"Loaded:\n{p.name}")
        self.status.setText("Ready.")
        self.append_log(f"Loaded file: {p}")

    def run_analysis(self):
        # lock UI
        self.analyse.setEnabled(False)
        self.choose.setEnabled(False)
        self.status.setText("Analysing…")
        self.append_log("---- Analyse clicked ----")

        t = threading.Thread(target=self._do_analysis, daemon=True)
        t.start()

    def _do_analysis(self):
        try:
            bsor_obj = None
            label = "Replay"

            # Prefer file if loaded
            if self.current_file and self.current_file.exists():
                label = self.current_file.stem
                self.signals.log.emit("Parsing BSOR file…")
                bsor_obj = parse_bsor_file(self.current_file)
            else:
                # Otherwise try link/scoreId
                txt = self.link.text().strip()
                sid = extract_score_id(txt)
                if sid is None:
                    raise RuntimeError("No .bsor loaded, and replay link / scoreId is missing or invalid.")

                self.signals.log.emit(f"ScoreId detected: {sid}")
                cdn_link, debug = try_get_replay_cdn_link(sid)
                self.signals.log.emit(debug)

                if not cdn_link:
                    raise RuntimeError("Could not find a BSOR download link for that scoreId.")

                self.signals.log.emit(f"Downloading BSOR from CDN…")
                bsor_bytes = download_bsor_to_bytes(cdn_link)
                self.signals.log.emit(f"Downloaded {len(bsor_bytes):,} bytes")

                self.signals.log.emit("Parsing BSOR bytes…")
                bsor_obj = parse_bsor_bytes(bsor_bytes)
                label = f"scoreId_{sid}"

            self.signals.log.emit("Building summary report…")
            report = summarize_bsor(bsor_obj)

            out_path = save_report(report, label)
            self.signals.success.emit(str(out_path))

        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n\n{tb}")

    def on_success(self, out_path: str):
        self.analyse.setEnabled(True)
        self.choose.setEnabled(True)
        self.status.setText(f"Done. Saved: {out_path}")
        QMessageBox.information(self, "Success", f"Report saved:\n{out_path}")

    def on_error(self, msg: str):
        self.analyse.setEnabled(True)
        self.choose.setEnabled(True)
        self.status.setText("Failed.")
        QMessageBox.critical(self, "Error", msg)
        self.append_log(msg)


def main():
    app = QApplication([])
    w = App()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
