"""Canvas de classement : QGraphicsView avec zoom/pan/rotation.

L'affichage est en lecture seule (les keypoints viennent des annotations et ne
sont ni déplaçables ni sélectionnables) : la seule interaction est la
classification des segments, clic/glissé gauche = mesurable, droit = non
mesurable.
"""
import functools

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QPixmap, QPainter, QPainterPath, QPainterPathStroker
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsLineItem
)

from app_logging import logger
from models import MEASURABLE, NON_MEASURABLE

KP_BASE_RADIUS = 4.0  # px écran, indépendant du zoom de l'image (ajusté par le zoom courant)
LINK_HIT_SCREEN_PX = 10.0  # largeur de la zone cliquable d'un segment de mesure, en px écran


def _safe_event(fn):
    """PySide6 termine le processus (pas juste une trace) quand une exception Python
    non gérée traverse un handler d'événement Qt appelé depuis le C++. On l'attrape
    ici pour la journaliser et continuer, plutôt que de faire planter le logiciel."""

    @functools.wraps(fn)
    def wrapper(self, event, *args, **kwargs):
        try:
            return fn(self, event, *args, **kwargs)
        except Exception:
            logger.exception("Exception dans %s.%s, ignorée pour éviter un plantage",
                             type(self).__name__, fn.__name__)
            try:
                event.accept()
            except Exception:
                pass
            return None

    return wrapper


class KeypointItem(QGraphicsEllipseItem):
    """Point annoté, purement indicatif : aucune interaction."""

    def __init__(self, name):
        super().__init__(-KP_BASE_RADIUS, -KP_BASE_RADIUS, KP_BASE_RADIUS * 2, KP_BASE_RADIUS * 2)
        self.name = name
        self.setBrush(QBrush(QColor(255, 220, 0)))
        pen = QPen(Qt.black, 1)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.setZValue(10)

    def apply_zoom_scale(self, zoom):
        self.setScale(1.0 / zoom if zoom else 1.0)


class LinkLine(QGraphicsLineItem):
    """Segment entre deux kp. Purement visuel : l'interaction (gomme) est gérée au
    niveau du canvas pour permettre un survol continu multi-segments."""

    def __init__(self, edge_key, canvas):
        super().__init__()
        self.edge_key = edge_key
        self.canvas = canvas
        self.setZValue(2)
        self.setCursor(Qt.CrossCursor)
        self.setAcceptedMouseButtons(Qt.NoButton)  # géré par le canvas (itemAt), pas par l'item

    def shape(self):
        # zone cliquable élargie et constante à l'écran (le trait lui-même est fin) :
        # le shape() par défaut d'une QGraphicsLineItem est quasi infinitésimal et
        # rendrait la détection au survol quasi impossible.
        path = QPainterPath()
        line = self.line()
        path.moveTo(line.p1())
        path.lineTo(line.p2())
        stroker = QPainterPathStroker()
        stroker.setWidth(LINK_HIT_SCREEN_PX / (self.canvas.zoom or 1.0))
        return stroker.createStroke(path)

    def set_status(self, status):
        if status == MEASURABLE:
            pen = QPen(QColor("lime"), 2)
        else:
            pen = QPen(QColor(150, 150, 150), 2, Qt.DashLine)
        pen.setCosmetic(True)  # épaisseur constante à l'écran, suit le zoom
        self.setPen(pen)


class MeasurementCanvas(QGraphicsView):
    edgeChanged = Signal(object, str)  # edge_key (tuple), statut
    strokeStarted = Signal()  # début d'un clic/glissé de gomme : point de sauvegarde undo

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        self.image_item = None
        self.kp_items = {}  # nom -> KeypointItem
        self.link_items = {}  # edge_key -> LinkLine
        self.zoom = 1.0
        self.rotation = 0  # degrés (0/90/180/270) : rotation d'affichage uniquement
        self._panning = False
        self._pan_start = None
        self._eraser_button = None
        self._stroke_notified = False

    # ---------- affichage d'une annotation ----------
    def show_annotation(self, ann):
        """(Re)construit la scène : image, keypoints annotés, segments des mesures."""
        self.scene_.clear()
        self.kp_items.clear()
        self.link_items.clear()

        pix = QPixmap(ann.image_path) if ann.image_path else QPixmap()
        if pix.isNull():
            logger.warning("Image introuvable ou illisible : %s", ann.image_path)
            pix = QPixmap(max(ann.width, 1), max(ann.height, 1))
            pix.fill(QColor(40, 40, 40))  # fond neutre : le squelette reste classable
        self.image_item = QGraphicsPixmapItem(pix)
        self.image_item.setZValue(0)
        self.scene_.addItem(self.image_item)
        self.scene_.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))

        for name, (x, y) in ann.keypoints.items():
            item = KeypointItem(name)
            item.setPos(x, y)
            self.scene_.addItem(item)
            self.kp_items[name] = item

        for key, (a, b) in self.config.edges.items():
            if not (ann.has_kp(a) and ann.has_kp(b)):
                continue
            line = LinkLine(key, self)
            ax, ay = ann.keypoints[a]
            bx, by = ann.keypoints[b]
            line.setLine(ax, ay, bx, by)
            line.set_status(ann.edge_status(key))
            self.scene_.addItem(line)
            self.link_items[key] = line

        self.rotation = ann.image_rotation
        self.fit_to_view()

    def refresh_statuses(self, ann):
        """Réapplique les couleurs des segments depuis le modèle (undo, preset, liste)."""
        for key, line in self.link_items.items():
            line.set_status(ann.edge_status(key))

    def set_link_status(self, edge_key, status):
        line = self.link_items.get(edge_key)
        if line is not None:
            line.set_status(status)

    # ---------- zoom / rotation / pan ----------
    def fit_to_view(self):
        """Réinitialise le transform (zoom + rotation) pour cadrer l'image dans la vue.

        La rotation est appliquée uniquement à la vue (QGraphicsView), jamais aux
        coordonnées scène : un kp annoté à (x, y) reste au même (x, y) quelle que
        soit la rotation d'affichage courante."""
        if self.image_item is None:
            return
        self.resetTransform()
        rect = self.image_item.boundingRect()
        viewport_rect = self.viewport().rect()
        # Si la vue n'est pas encore affichée/mise en page, le viewport peut être
        # dégénéré (0 ou quelques px) : un scale nul rendrait le transform singulier
        # et casserait toute la géométrie de la scène ensuite.
        if viewport_rect.width() < 10 or viewport_rect.height() < 10:
            logger.warning(
                "fit_to_view : viewport dégénéré (%dx%d), la fenêtre n'est peut-être pas encore affichée",
                viewport_rect.width(), viewport_rect.height(),
            )
        if rect.width() and rect.height() and viewport_rect.width() and viewport_rect.height():
            if self.rotation % 180 == 90:
                avail_w, avail_h = viewport_rect.height(), viewport_rect.width()
            else:
                avail_w, avail_h = viewport_rect.width(), viewport_rect.height()
            scale = min(avail_w / rect.width(), avail_h / rect.height())
        else:
            scale = 1.0
        scale = max(scale, 1e-3)  # jamais 0 : un transform singulier casserait mapToScene/itemAt
        self.scale(scale, scale)
        if self.rotation:
            self.rotate(self.rotation)
        self.zoom = scale
        self.centerOn(self.image_item)
        self._rescale_fixed_items()

    def rotate_view(self, delta):
        self.rotation = (self.rotation + delta) % 360
        self.fit_to_view()

    def _rescale_fixed_items(self):
        for it in self.kp_items.values():
            it.apply_zoom_scale(self.zoom)

    @_safe_event
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.zoom *= factor
        self.scale(factor, factor)
        self._rescale_fixed_items()
        event.accept()

    # ---------- classement des segments ----------
    def _apply_eraser(self, view_pos, button):
        item = self.itemAt(view_pos)
        if not isinstance(item, LinkLine):
            return
        # un seul point de sauvegarde par clic/glissé, et seulement s'il change
        # vraiment quelque chose (sinon un clic dans le vide remplirait la pile undo)
        if not self._stroke_notified:
            self._stroke_notified = True
            self.strokeStarted.emit()
        # clic/glissé gauche = mesurable, clic/glissé droit = non mesurable (gomme)
        status = MEASURABLE if button == Qt.LeftButton else NON_MEASURABLE
        self.edgeChanged.emit(item.edge_key, status)

    @_safe_event
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._eraser_button = event.button()
            self._stroke_notified = False
            self._apply_eraser(event.pos(), event.button())
            event.accept()
            return
        super().mousePressEvent(event)

    @_safe_event
    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._eraser_button is not None and (event.buttons() & self._eraser_button):
            self._apply_eraser(event.pos(), self._eraser_button)
            event.accept()
            return
        super().mouseMoveEvent(event)

    @_safe_event
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if self._eraser_button is not None and event.button() == self._eraser_button:
            self._eraser_button = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
