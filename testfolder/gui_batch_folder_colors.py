"""
testfolder/gui_batch_folder_colors.py

GUI batch personal-color checker.

Select a root folder. The app scans images inside that folder and all child
folders, runs the same inference pipeline as the kiosk, and shows per-file
results in a table.

Usage:
  python testfolder/gui_batch_folder_colors.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ai" / "models" / "final_hierarchical.pt"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.infer import run_inference
from ai.model_loader import ModelLoader

SEASON_ALIASES = {
    "spring": "Spring",
    "summer": "Summer",
    "fall": "Autumn",
    "autumn": "Autumn",
    "winter": "Winter",
}

SEASON_KO = {
    "Spring": "봄",
    "Summer": "여름",
    "Autumn": "가을",
    "Winter": "겨울",
}


def load_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def top3_text(result: dict[str, Any]) -> str:
    items = []
    for item in result.get("top3", [])[:3]:
        label = item.get("label_ko") or item.get("color", "-")
        confidence = float(item.get("confidence", 0.0)) * 100
        items.append(f"{label} {confidence:.1f}%")
    return " / ".join(items)


def season_totals(result: dict[str, Any]) -> str:
    try:
        raw = json.loads(result.get("raw_scores") or "{}")
    except Exception:
        return "-"
    totals = {"Spring": 0.0, "Summer": 0.0, "Autumn": 0.0, "Winter": 0.0}
    for key, value in raw.items():
        season = str(key).split("_", 1)[0]
        if season in totals:
            totals[season] += float(value)
    return " / ".join(f"{season[:2]} {score * 100:.1f}%" for season, score in totals.items())


def expected_season_from_folder(folder: str) -> str:
    parts = [part.lower() for part in Path(folder).parts if part and part != "."]
    for part in parts:
        if part in SEASON_ALIASES:
            return SEASON_ALIASES[part]
    return ""


def ko_season(season: str) -> str:
    return SEASON_KO.get(season, season or "-")


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)
    row_ready = pyqtSignal(dict)
    summary_ready = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished_ok = pyqtSignal()

    def __init__(self, root_dir: Path) -> None:
        super().__init__()
        self._root_dir = root_dir
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            images = collect_images(self._root_dir)
            if not images:
                self.failed.emit("선택한 폴더와 하위 폴더에 이미지가 없습니다.")
                return

            ModelLoader().load(str(MODEL_PATH))
            folder_counts: dict[str, Counter[str]] = defaultdict(Counter)
            folder_eval: dict[str, Counter[str]] = defaultdict(Counter)
            total = len(images)

            for index, path in enumerate(images, 1):
                if self._stop_requested:
                    break

                rel_folder = str(path.parent.relative_to(self._root_dir)) if path.parent != self._root_dir else "."
                row = {
                    "folder": rel_folder,
                    "file": path.name,
                    "expected": expected_season_from_folder(rel_folder),
                    "predicted": "",
                    "match": "-",
                    "label": "실패",
                    "confidence": "-",
                    "top3": "-",
                    "season_totals": "-",
                    "message": "",
                    "ok": False,
                }

                bgr = load_bgr(path)
                if bgr is None:
                    row["message"] = "이미지 로드 실패"
                    self.row_ready.emit(row)
                    self.progress.emit(index, total)
                    continue

                result = run_inference(bgr)
                if result.get("success"):
                    label = result.get("label_ko", "-")
                    predicted = str(result.get("season") or "")
                    expected = row["expected"]
                    is_match = bool(expected and predicted == expected)
                    row.update({
                        "predicted": predicted,
                        "match": "일치" if is_match else ("불일치" if expected else "-"),
                        "label": label,
                        "confidence": f"{float(result.get('confidence', 0.0)) * 100:.1f}%",
                        "top3": top3_text(result),
                        "season_totals": season_totals(result),
                        "message": result.get("quality_warning") or "OK",
                        "ok": True,
                    })
                    folder_counts[rel_folder][label] += 1
                    folder_eval[rel_folder]["total"] += 1
                    if is_match:
                        folder_eval[rel_folder]["correct"] += 1
                else:
                    row["message"] = result.get("message", "분석 실패")

                self.row_ready.emit(row)
                self.progress.emit(index, total)

            self.summary_ready.emit(make_summary(folder_counts, folder_eval))
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


def make_summary(folder_counts: dict[str, Counter[str]], folder_eval: dict[str, Counter[str]]) -> str:
    if not folder_counts:
        return "성공한 분석 결과가 없습니다."
    lines = []
    for folder in sorted(folder_counts):
        counter = folder_counts[folder]
        total = sum(counter.values())
        top = counter.most_common(3)
        top_text = " / ".join(f"{label} {count}개" for label, count in top)
        expected = expected_season_from_folder(folder)
        eval_counter = folder_eval.get(folder, Counter())
        correct = eval_counter.get("correct", 0)
        evaluated = eval_counter.get("total", 0)
        acc = (correct / evaluated * 100.0) if evaluated else 0.0
        expected_text = f"정답={ko_season(expected)}  " if expected else ""
        lines.append(f"{folder}  {expected_text}정확도 {correct}/{evaluated} ({acc:.1f}%)  결과분포: {top_text}")
    return "\n".join(lines)


class BatchColorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Folder Personal Color Analyzer")
        self.resize(1280, 820)
        self._worker: InferenceWorker | None = None
        self._root_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        page = QWidget()
        self.setCentralWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(12)

        title = QLabel("폴더별 퍼스널 컬러 판별")
        title.setStyleSheet("font-size: 28px; font-weight: 900;")
        layout.addWidget(title)

        controls = QHBoxLayout()
        self._folder_label = QLabel("폴더를 선택하세요")
        self._folder_label.setStyleSheet("font-size: 15px; color: #5f5750;")
        choose_btn = QPushButton("폴더 선택")
        choose_btn.clicked.connect(self._choose_folder)
        self._run_btn = QPushButton("분석 시작")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._start_analysis)
        self._stop_btn = QPushButton("중지")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_analysis)
        controls.addWidget(self._folder_label, stretch=1)
        controls.addWidget(choose_btn)
        controls.addWidget(self._run_btn)
        controls.addWidget(self._stop_btn)
        layout.addLayout(controls)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._table = QTableWidget(0, 10)
        self._table.setHorizontalHeaderLabels([
            "폴더",
            "파일명",
            "정답계절",
            "예측계절",
            "일치",
            "결과",
            "신뢰도",
            "TOP3",
            "계절합계",
            "메시지",
        ])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        layout.addWidget(self._table, stretch=1)

        summary_title = QLabel("폴더별 요약")
        summary_title.setStyleSheet("font-size: 18px; font-weight: 900;")
        layout.addWidget(summary_title)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setFixedHeight(120)
        layout.addWidget(self._summary)

        page.setStyleSheet("""
            QWidget { background: #F8F4EF; color: #251D19; font-family: Pretendard, Arial; }
            QPushButton {
                background: #251D19;
                color: #FFFDF9;
                border: 1px solid #D6B979;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 15px;
                font-weight: 800;
            }
            QPushButton:disabled { background: #C8C0B8; color: #766E66; }
            QTableWidget { background: #FFFDF9; alternate-background-color: #F1EAE3; gridline-color: #D6B979; }
            QHeaderView::section { background: #EADBC6; padding: 8px; font-weight: 900; border: none; }
            QTextEdit { background: #FFFDF9; border: 1px solid #D6B979; border-radius: 6px; padding: 8px; }
            QProgressBar { border: 1px solid #D6B979; border-radius: 6px; height: 18px; background: #FFFDF9; }
            QProgressBar::chunk { background: #FF6B78; border-radius: 6px; }
        """)

    def _choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "분석할 최상위 폴더 선택")
        if not selected:
            return
        self._root_dir = Path(selected)
        self._folder_label.setText(str(self._root_dir))
        self._run_btn.setEnabled(True)

    def _start_analysis(self) -> None:
        if self._root_dir is None:
            return
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._summary.clear()
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = InferenceWorker(self._root_dir)
        self._worker.row_ready.connect(self._add_row)
        self._worker.progress.connect(self._set_progress)
        self._worker.summary_ready.connect(self._summary.setPlainText)
        self._worker.failed.connect(self._show_error)
        self._worker.finished_ok.connect(self._finish_analysis)
        self._worker.start()

    def _stop_analysis(self) -> None:
        if self._worker:
            self._worker.stop()
        self._stop_btn.setEnabled(False)

    def _set_progress(self, current: int, total: int) -> None:
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _add_row(self, data: dict) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        values = [
            data["folder"],
            data["file"],
            ko_season(data["expected"]),
            ko_season(data["predicted"]),
            data["match"],
            data["label"],
            data["confidence"],
            data["top3"],
            data["season_totals"],
            data["message"],
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            if not data.get("ok"):
                item.setBackground(QColor("#FFE3E3"))
            elif data.get("match") == "불일치":
                item.setBackground(QColor("#FFF0B8"))
            elif data.get("match") == "일치":
                item.setBackground(QColor("#E2F6E8"))
            self._table.setItem(row, col, item)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "분석 오류", message)
        self._finish_analysis()

    def _finish_analysis(self) -> None:
        self._table.setSortingEnabled(True)
        self._run_btn.setEnabled(self._root_dir is not None)
        self._stop_btn.setEnabled(False)
        self._worker = None

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = BatchColorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
