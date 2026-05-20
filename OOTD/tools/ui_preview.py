from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kiosk.ui import KioskWindow


def main() -> None:
    app = QApplication(sys.argv)
    w = KioskWindow(hide_cursor=False, preview_mode=True)
    w.show()
    w._goto_idle()
    QTimer.singleShot(1200, w._goto_guide)
    QTimer.singleShot(2400, w._goto_analysis)
    QTimer.singleShot(3600, lambda: w._on_analyze_finished(w._result_data))
    QTimer.singleShot(4800, w._goto_reco)
    QTimer.singleShot(6000, w._goto_qr)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
