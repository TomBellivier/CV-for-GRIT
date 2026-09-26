"""Centralised logging configuration, to diagnose crashes.

Writes to measurement_validation/logs/:
- app.log          : application logs (DEBUG+, with rotation), including the
                     tracebacks of uncaught Python exceptions
- faulthandler.log : low-level trace on a native crash (segfault, etc.), useful
                     when the crash does not even raise a Python exception
                     (frequent with Qt/PySide when a C++ object is used after
                     deletion).

Never writes into the image folder nor into the folder of the exported CSV.
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

_faulthandler_file = None  # reference kept alive for the whole execution


def _apply_qt_windows_workarounds():
    """Workarounds for a recurring native crash on Windows:
    "QThreadStorage: entry N destroyed before end of thread" followed by a
    "Fatal Python error: Aborted", observed during app.exec() without any Python
    exception being raised (hence impossible to catch on the Python side).

    Two independent leads, both applied:
    1. The PySide6 wheels (pip) contain no fonts/ folder: Qt prints
       "Cannot find font directory ... Qt no longer ships fonts" and falls back to
       an enumeration of the system fonts. QT_QPA_FONTDIR removes this fallback.
    2. The default Qt6 font engine on Windows (DirectWrite) enumerates the fonts in
       a background thread whose synchronisation/teardown has known bugs in some
       Qt6 versions; forcing the FreeType engine (single-threaded, older and
       simpler) avoids this code path.

    Must be called before any PySide6 import."""
    if sys.platform != "win32":
        return
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = Path(windir) / "Fonts"
    if fonts_dir.is_dir():
        os.environ.setdefault("QT_QPA_FONTDIR", str(fonts_dir))
    # setdefault must not overwrite a value already set (e.g. QT_QPA_PLATFORM=offscreen
    # set by the tests): only add the fontengine parameter if the default windows
    # platform has not already been explicitly chosen otherwise.
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"


def setup_logging():
    """To call only once, at the very start of main(), before any Qt import."""
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

    # Uncaught Python exceptions (including those raised in the Qt callbacks, which
    # otherwise only show up in the console and then vanish).
    def _log_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_uncaught

    # Native crashes (segfault, etc.): Python then raises no exception, faulthandler
    # is the only way to keep a usable trace of them.
    _faulthandler_file = open(LOG_DIR / "faulthandler.log", "a", encoding="utf-8")
    faulthandler.enable(file=_faulthandler_file, all_threads=True)

    logger.info("=" * 70)
    logger.info("Application start (logs in %s)", LOG_DIR)
    return logger


def install_qt_message_handler():
    """Redirect the Qt messages (internal warnings/errors of PySide6/Qt) to the
    logger: these messages often precede a native crash (e.g. access to a
    QGraphicsItem already removed from the scene) and are otherwise lost."""
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
