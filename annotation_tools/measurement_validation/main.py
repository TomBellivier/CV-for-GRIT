"""Entry point of the measurement classification application.

    python annotation_tools/measurement_validation/main.py --annotations_csv annotations.csv

The input CSV is the one produced by annotation_tools/labelstudio_to_csv.py. The
output CSV (status of each measurement for each image) is written to
annotation_data/meas_classifier/, under the name "<input csv>_measurements.csv".
"""
import argparse
import sys
from pathlib import Path

from app_logging import LOG_DIR, install_qt_message_handler, logger, setup_logging

setup_logging()
install_qt_message_handler()

from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

import csv_io
from config import REPO_ROOT, load_config
from main_window import MainWindow

DEFAULT_EXPORT_DIR = REPO_ROOT / "annotation_data" / "meas_classifier"


def parse_args():
    parser = argparse.ArgumentParser(description="Measurable / non measurable classification of the measurements.")
    parser.add_argument("--annotations_csv", default=None,
                        help="CSV produced by labelstudio_to_csv.py (asked at launch if missing).")
    parser.add_argument("--output_dir", default=str(DEFAULT_EXPORT_DIR),
                        help=f"Folder of the status CSV (default: {DEFAULT_EXPORT_DIR}).")
    parser.add_argument("--kp_infos", default=None,
                        help="kp_infos.yaml the measurements are read from (default: kp_infos.yaml at the repository root).")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    logger.info("QApplication created")

    config = load_config(args.kp_infos)
    logger.info("%d measurement(s) loaded", len(config.measurements))

    csv_path = args.annotations_csv
    if not csv_path:
        csv_path, _ = QFileDialog.getOpenFileName(
            None, "Select the annotation CSV", str(REPO_ROOT), "CSV (*.csv)")
    if not csv_path:
        logger.info("No CSV selected, exiting")
        return

    annotations = csv_io.load_annotations(csv_path)
    if not annotations:
        QMessageBox.warning(None, "No annotation", f"No usable annotation in {csv_path}.")
        return

    project_name = Path(csv_path).stem
    win = MainWindow(config, annotations, project_name, args.output_dir)
    # Resume: status per segment and images already validated in a previous session.
    csv_io.load_state(win.state_path, annotations)
    win.show()
    # loaded after showing the window: the viewport then has its real size, which avoids
    # a broken zoom computation (window not laid out yet).
    win.load_image(win.first_unvalidated_index())
    logger.info("Project '%s': %d annotation(s), export to %s",
                project_name, len(annotations), win.csv_path)

    exit_code = app.exec()
    logger.info("End of the event loop (code %d)", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical("Fatal exception in main()", exc_info=True)
        raise
    finally:
        logger.info("Logs available in %s", LOG_DIR)
