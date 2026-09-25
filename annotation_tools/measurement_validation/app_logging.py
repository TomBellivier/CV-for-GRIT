"""Configuration centralisée des logs, pour diagnostiquer les plantages.

Écrit dans measurement_validation/logs/ :
- app.log          : logs applicatifs (DEBUG+, avec rotation), y compris les
                     tracebacks des exceptions Python non interceptées
- faulthandler.log : trace de bas niveau en cas de crash natif (segfault, etc.),
                     utile quand le plantage ne lève même pas d'exception Python
                     (fréquent avec Qt/PySide si un objet C++ est utilisé après
                     suppression).

N'écrit jamais dans le dossier des images ni dans celui du CSV exporté.
"""
import faulthandler
import logging
import logging.handlers
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"

logger = logging.getLogger("measurement_validation")

_faulthandler_file = None  # référence gardée en vie tout le long de l'exécution


def _apply_qt_windows_workarounds():
    """Contournements pour un plantage natif récurrent sur Windows :
    "QThreadStorage: entry N destroyed before end of thread" suivi d'un
    "Fatal Python error: Aborted", observé pendant app.exec() sans qu'aucune
    exception Python ne soit levée (donc impossible à intercepter côté Python).

    Deux pistes indépendantes, appliquées toutes les deux :
    1. Les wheels PySide6 (pip) ne contiennent pas de dossier fonts/ : Qt affiche
       "Cannot find font directory ... Qt no longer ships fonts" et se rabat sur
       une énumération des polices système. QT_QPA_FONTDIR supprime ce repli.
    2. Le moteur de polices par défaut de Qt6 sur Windows (DirectWrite) énumère les
       polices dans un thread d'arrière-plan dont la synchronisation/le teardown a
       des bugs connus sur certaines versions de Qt6 ; forcer le moteur FreeType
       (mono-thread, plus ancien et plus simple) évite ce chemin de code.

    Doit être appelé avant tout import de PySide6."""
    if sys.platform != "win32":
        return
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = Path(windir) / "Fonts"
    if fonts_dir.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fonts_dir))
    # setdefault ne doit pas écraser une valeur déjà fixée (ex. QT_QPA_PLATFORM=offscreen
    # positionné par les tests) : n'ajoute le paramètre fontengine que si la plateforme
    # windows par défaut n'a pas déjà été explicitement choisie autrement.
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"


def setup_logging():
    """À appeler une seule fois, au tout début de main(), avant tout import Qt."""
    global _faulthandler_file
    _apply_qt_windows_workarounds()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s:%(filename)s:%(lineno)d - %(message)s"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Exceptions Python non interceptées (dont celles levées dans les callbacks Qt,
    # qui autrement ne font que s'afficher dans la console puis disparaissent).
    def _log_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Exception non interceptée", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught

    # Plantages natifs (segfault, etc.) : Python ne lève alors aucune exception,
    # faulthandler est le seul moyen d'en garder une trace exploitable.
    _faulthandler_file = open(LOG_DIR / "faulthandler.log", "a", encoding="utf-8")
    faulthandler.enable(file=_faulthandler_file, all_threads=True)

    logger.info("=" * 70)
    logger.info("Démarrage de l'application (logs dans %s)", LOG_DIR)
    return logger


def install_qt_message_handler():
    """Redirige les messages Qt (warnings/erreurs internes de PySide6/Qt) vers le
    logger : ces messages précèdent souvent un plantage natif (ex. accès à un
    QGraphicsItem déjà supprimé de la scène) et sont sinon perdus."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(msg_type, context, message):
        level = level_map.get(msg_type, logging.WARNING)
        location = ""
        if context.file:
            location = f" ({context.file}:{context.line})"
        logger.log(level, "[Qt] %s%s", message, location)

    qInstallMessageHandler(handler)
