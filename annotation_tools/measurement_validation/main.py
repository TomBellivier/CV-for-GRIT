"""Point d'entrée de l'application de classement des mesures.

    python annotation_tools/measurement_validation/main.py --annotations_csv annotations.csv

Le CSV d'entrée est celui produit par annotation_tools/labelstudio_to_csv.py. Le
CSV de sortie (statut de chaque mesure pour chaque image) est écrit dans
annotation_data/meas_classifier/, sous le nom "<csv d'entrée>_measurements.csv".
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
    parser = argparse.ArgumentParser(description="Classement mesurable / non mesurable des mesures.")
    parser.add_argument("--annotations_csv", default=None,
                        help="CSV produit par labelstudio_to_csv.py (demandé au lancement si absent).")
    parser.add_argument("--output_dir", default=str(DEFAULT_EXPORT_DIR),
                        help=f"Dossier du CSV de statuts (défaut : {DEFAULT_EXPORT_DIR}).")
    parser.add_argument("--kp_infos", default=None,
                        help="kp_infos.yaml d'où sont lues les mesures (défaut : kp_infos.yaml à la racine du dépôt).")
    return parser.parse_args()


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    logger.info("QApplication créée")

    config = load_config(args.kp_infos)
    logger.info("%d mesure(s) chargée(s)", len(config.measurements))

    csv_path = args.annotations_csv
    if not csv_path:
        csv_path, _ = QFileDialog.getOpenFileName(
            None, "Sélectionner le CSV d'annotations", str(REPO_ROOT), "CSV (*.csv)")
    if not csv_path:
        logger.info("Aucun CSV sélectionné, sortie")
        return

    annotations = csv_io.load_annotations(csv_path)
    if not annotations:
        QMessageBox.warning(None, "Aucune annotation", f"Aucune annotation exploitable dans {csv_path}.")
        return

    project_name = Path(csv_path).stem
    win = MainWindow(config, annotations, project_name, args.output_dir)
    # Reprise : statut par segment et images déjà validées lors d'une session précédente.
    csv_io.load_state(win.state_path, annotations)
    win.show()
    # chargée après l'affichage : le viewport a alors sa taille réelle, ce qui évite
    # un calcul de zoom foireux (fenêtre pas encore mise en page).
    win.load_image(win.first_unvalidated_index())
    logger.info("Projet '%s' : %d annotation(s), export vers %s",
                project_name, len(annotations), win.csv_path)

    exit_code = app.exec()
    logger.info("Fin de la boucle d'événements (code %d)", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical("Exception fatale dans main()", exc_info=True)
        raise
    finally:
        logger.info("Logs disponibles dans %s", LOG_DIR)
