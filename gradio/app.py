"""
Insect Keypoint Detection – Gradio Interface
Loads a YOLO (.pt) pose/keypoint model, accepts images + YOLO-format labels,
and displays ground-truth vs. model predictions side by side.
"""

import os
import glob
import tempfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────
# Colour palette (keypoint index → BGR for cv2)
# ──────────────────────────────────────────────
KP_COLORS = [
    (255,   0,   0), (  0, 255,   0), (  0,   0, 255),
    (255, 255,   0), (  0, 255, 255), (255,   0, 255),
    (128, 255,   0), (  0, 128, 255), (255, 128,   0),
    (128,   0, 255), (  0, 255, 128), (255,   0, 128),
    (64,  255,  64), (255,  64,  64), ( 64,  64, 255),
    (200, 200,   0), (  0, 200, 200), (200,   0, 200),
]

SKELETON = []           # fill with [(i,j), …] if connectivity is known
BBOX_COLOR_GT   = (0, 200, 0)
BBOX_COLOR_PRED = (0, 0, 255)
KP_RADIUS = 5
KP_THICKNESS = -1
BBOX_THICKNESS = 2


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def color_for(idx: int):
    return KP_COLORS[idx % len(KP_COLORS)]


def parse_yolo_label(label_path: str, img_w: int, img_h: int):
    """
    Returns list of dicts:
      { 'cls': int, 'bbox': (x1,y1,x2,y2), 'keypoints': [(x,y,v), …] }
    YOLO keypoint row: cls cx cy bw bh  kx1 ky1 kv1  kx2 ky2 kv2 …
    All coords are normalised [0,1].
    """
    objects = []
    if not label_path or not os.path.isfile(label_path):
        return objects

    with open(label_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 5:
                continue

            cls = int(vals[0])
            cx, cy, bw, bh = vals[1], vals[2], vals[3], vals[4]
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)

            kps = []
            for i in range(5, len(vals) - 1, 3):
                kx = vals[i]   * img_w
                ky = vals[i+1] * img_h
                kv = vals[i+2] if i + 2 < len(vals) else 2.0
                kps.append((kx, ky, kv))

            # 2-value keypoint format: kx ky (no visibility)
            if not kps and len(vals) > 5:
                extra = vals[5:]
                for i in range(0, len(extra) - 1, 2):
                    kps.append((extra[i] * img_w, extra[i+1] * img_h, 2.0))

            objects.append({"cls": cls, "bbox": (x1, y1, x2, y2), "keypoints": kps})

    return objects


def draw_annotations(img_bgr, objects, bbox_color, label_prefix=""):
    """Draw bboxes + keypoints on a copy of img_bgr."""
    out = img_bgr.copy()
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, BBOX_THICKNESS)

        # label
        tag = f"{label_prefix}cls{obj['cls']}"
        cv2.putText(out, tag, (x1, max(y1 - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    bbox_color, 1, cv2.LINE_AA)

        kps = obj["keypoints"]
        for idx, (kx, ky, kv) in enumerate(kps):
            if kv == 0:           # not labelled / invisible
                continue
            c = color_for(idx)
            cv2.circle(out, (int(kx), int(ky)), KP_RADIUS, c, KP_THICKNESS)
            cv2.circle(out, (int(kx), int(ky)), KP_RADIUS + 1, (0, 0, 0), 1)

        for (a, b) in SKELETON:
            if a < len(kps) and b < len(kps):
                (ax, ay, av), (bx, by, bv) = kps[a], kps[b]
                if av > 0 and bv > 0:
                    cv2.line(out, (int(ax), int(ay)), (int(bx), int(by)),
                             (200, 200, 200), 1, cv2.LINE_AA)
    return out


def result_to_objects(result, conf_thresh: float):
    """Convert a single ultralytics Result to our internal format."""
    objects = []
    if result.boxes is None:
        return objects

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    kps_all = None
    if result.keypoints is not None:
        kps_all = result.keypoints.data.cpu().numpy()  # (N, K, 3)

    for i, (box, conf, cls) in enumerate(zip(boxes, confs, clses)):
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        kps = []
        if kps_all is not None and i < len(kps_all):
            for kp in kps_all[i]:
                kx, ky = float(kp[0]), float(kp[1])
                kv = float(kp[2]) if kp.shape[0] > 2 else 2.0
                kps.append((kx, ky, kv))
        objects.append({"cls": cls, "bbox": (x1, y1, x2, y2), "keypoints": kps, "conf": float(conf)})
    return objects


def build_legend(n_kps: int):
    """Return a small legend figure as numpy array."""
    if n_kps == 0:
        return None
    fig, ax = plt.subplots(figsize=(3, max(1.5, n_kps * 0.35)))
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=np.array(color_for(i)[::-1]) / 255,
                      markersize=9, label=f"KP {i}")
               for i in range(n_kps)]
    handles += [
        patches.Patch(facecolor=np.array(BBOX_COLOR_GT[::-1]) / 255, label="GT bbox"),
        patches.Patch(facecolor=np.array(BBOX_COLOR_PRED[::-1]) / 255, label="Pred bbox"),
    ]
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=8)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


# ──────────────────────────────────────────────
# State holder
# ──────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.model = None
        self.model_path = ""

state = AppState()


# ──────────────────────────────────────────────
# Gradio callbacks
# ──────────────────────────────────────────────

def load_model(model_file):
    if model_file is None:
        return "No file selected."
    try:
        path = model_file.name if hasattr(model_file, "name") else model_file
        state.model = YOLO(path)
        state.model_path = path
        # Quick introspection
        task = getattr(state.model, "task", "unknown")
        names = getattr(state.model.model, "names", {})
        n_kps = 0
        try:
            n_kps = state.model.model.kpt_shape[0]
        except Exception:
            pass
        cls_str = ", ".join(f"{k}:{v}" for k, v in names.items()) if names else "n/a"
        return (
            f"Model loaded: `{Path(path).name}`\n"
            f"• Task : {task}\n"
            f"• Classes : {cls_str}\n"
            f"• Keypoints per instance : {n_kps}"
        )
    except Exception as e:
        state.model = None
        return f"Failed to load model:\n{e}"


def run_inference(image_files, label_files, conf_thresh, img_size):
    if state.model is None:
        return [], "Please load a model first."
    if not image_files:
        return [], "No images uploaded."

    # Build a quick label lookup: stem → path
    label_map = {}
    if label_files:
        for lf in label_files:
            p = lf.name if hasattr(lf, "name") else lf
            label_map[Path(p).stem] = p

    gallery_imgs = []
    log_lines = []
    img_size = int(img_size)

    for img_file in image_files:
        img_path = img_file.name if hasattr(img_file, "name") else img_file
        img_stem  = Path(img_path).stem

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            log_lines.append(f"Could not read: {img_path}")
            continue

        h, w = img_bgr.shape[:2]

        # ── Ground truth ──
        gt_objects = []
        if img_stem in label_map:
            gt_objects = parse_yolo_label(label_map[img_stem], w, h)
            log_lines.append(f"{img_stem}: {len(gt_objects)} GT instance(s)")
        else:
            log_lines.append(f"{img_stem}: no label found")

        # ── Prediction ──
        results = state.model.predict(
            source=img_path,
            imgsz=img_size,
            conf=conf_thresh,
            verbose=False,
        )
        pred_objects = result_to_objects(results[0], conf_thresh)
        log_lines.append(f"{img_stem}: {len(pred_objects)} prediction(s)")

        # ── Draw GT panel ──
        gt_panel = draw_annotations(img_bgr, gt_objects, BBOX_COLOR_GT, "gt-")

        # ── Draw Pred panel ──
        pred_panel = draw_annotations(img_bgr, pred_objects, BBOX_COLOR_PRED, "pred-")

        # ── Side-by-side composite ──
        divider = np.full((h, 4, 3), 40, dtype=np.uint8)
        composite = np.concatenate([gt_panel, divider, pred_panel], axis=1)

        # Add header banners
        banner_h = 30
        banner_gt   = np.full((banner_h, w, 3),   (30, 120, 30), dtype=np.uint8)
        banner_pred = np.full((banner_h, w, 3),   (30,  30, 120), dtype=np.uint8)
        banner_div  = np.full((banner_h, 4, 3),    40,             dtype=np.uint8)

        cv2.putText(banner_gt,   "Ground Truth",  (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 1)
        cv2.putText(banner_pred, "Prediction",    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 1)

        banner = np.concatenate([banner_gt, banner_div, banner_pred], axis=1)
        composite = np.concatenate([banner, composite], axis=0)

        # Per-image keypoint stats
        if gt_objects and pred_objects:
            gt_kps_flat  = [kp for o in gt_objects  for kp in o["keypoints"] if kp[2] > 0]
            pr_kps_flat  = [kp for o in pred_objects for kp in o["keypoints"] if kp[2] > 0]
            log_lines.append(
                f"   GT kps: {len(gt_kps_flat)}  |  Pred kps: {len(pr_kps_flat)}"
            )

        # Convert BGR → RGB for Gradio
        gallery_imgs.append(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))

    return gallery_imgs, "\n".join(log_lines)


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────

CSS = """
#title { text-align: center; }
.panel-header { font-weight: bold; color: #555; }
"""

with gr.Blocks(css=CSS, title="Insect Keypoint Detector") as demo:

    gr.Markdown("# Insect Keypoint Detection", elem_id="title")
    gr.Markdown(
        "Load a YOLO pose model (`.pt`), upload insect images with their YOLO-format "
        "label files, then run inference to compare **ground truth** and **predictions**."
    )

    # ── Row 1 : Model loading ──
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 1 · Load Model")
            model_file   = gr.File(label="YOLO model (.pt)", file_types=[".pt"], height=80)
            load_btn     = gr.Button("Load Model", variant="primary")
        with gr.Column(scale=3):
            model_status = gr.Markdown("*No model loaded yet.*")

    # gr.Divider()

    # ── Row 2 : Data upload ──
    gr.Markdown("### 2 · Upload Images and Labels")
    with gr.Row():
        image_files = gr.File(
            label="Images (.jpg / .png / …)",
            file_count="multiple",
            file_types=["image"],
            height=120,
        )
        label_files = gr.File(
            label="YOLO labels (.txt) — optional",
            file_count="multiple",
            file_types=[".txt"],
            height=120,
        )

    # gr.Divider()

    # ── Row 3 : Inference settings ──
    gr.Markdown("### 3 · Inference Settings")
    with gr.Row():
        conf_slider = gr.Slider(0.01, 1.0, value=0.25, step=0.01,
                                label="Confidence threshold")
        imgsz_dd    = gr.Dropdown(
            choices=["320", "416", "512", "640", "768", "1024", "1280"],
            value="640",
            label="Inference image size",
        )
        run_btn     = gr.Button("Run Inference", variant="primary", scale=1)

    # gr.Divider()

    # ── Row 4 : Results ──
    gr.Markdown("### 4 · Results")
    with gr.Row():
        gallery = gr.Gallery(
            label="GT  |  Prediction",
            columns=2,
            object_fit="contain",
            height=520,
        )
    log_box = gr.Textbox(label="Log", lines=8, interactive=False)

    # ── Wiring ──
    load_btn.click(fn=load_model, inputs=[model_file], outputs=[model_status])

    run_btn.click(
        fn=run_inference,
        inputs=[image_files, label_files, conf_slider, imgsz_dd],
        outputs=[gallery, log_box],
    )

    gr.Markdown(
        "---\n"
        "**Label format:** YOLO pose — `class cx cy bw bh  kx1 ky1 kv1  kx2 ky2 kv2 …` (normalised)\n\n"
        "Visibility values: `0` = not labelled · `1` = occluded · `2` = visible"
    )


if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)