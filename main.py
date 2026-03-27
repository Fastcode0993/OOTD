import sys, logging
from PyQt6.QtWidgets import QApplication
from main_window import MainWindow

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%H:%M:%S")
    app=QApplication(sys.argv); app.setApplicationName("AI Personal Color Platform"); app.setApplicationVersion("2.0.0")
    win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__=="__main__": main()
