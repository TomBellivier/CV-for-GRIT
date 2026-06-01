from lora_config import *
from lora_classes import *

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class InsectPosePredictor:
    """
    pipeline d'inférence complet.

    utilisation avec classifieur futur :
        predictor = InsectPosePredictor(model, lora_manager)
        group = classifier.predict(image)          # à implémenter
        results = predictor.predict(image, group=group)
    """

    def __init__(
        self,
        model: YOLOPoseLoRA,
        manager: LoRAWeightManager,
        img_size: int = 640,
    ):
        self.model    = model
        self.manager  = manager
        self.img_size = img_size
        self._transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        image: np.ndarray,
        group: str,
        conf_threshold: float = 0.25,
    ) -> Dict:
        """
        args:
            image : np.ndarray rgb (h, w, 3)
            group : nom du groupe fourni par le classifieur
        returns:
            dict avec 'boxes', 'keypoints', 'scores'
        """
        if self.model.active_group != group:
            self.model.switch_group(group, self.manager)

        h_orig, w_orig = image.shape[:2]
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        tensor = self._transform(img_resized).unsqueeze(0).to(DEVICE)

        self.model.backbone.eval()
        with torch.no_grad():
            raw_preds = self.model(tensor)

        return self._decode_predictions(raw_preds, (h_orig, w_orig), conf_threshold)

    def _decode_predictions(self, raw, orig_shape, conf_thr):
        """placeholder — remplacer par le décodeur ultralytics."""
        boxes, keypoints, scores = [], [], []

        if raw.boxes is not None:
            for box, score in zip(raw.boxes.xyxy.cpu(), raw.boxes.conf.cpu()):
                boxes.append(box.tolist())
                scores.append(score.item())

        if raw.keypoints is not None:
            for kps in raw.keypoints.data.cpu():
                keypoints.append(kps.tolist())

        return {
            "boxes":     boxes,
            "keypoints": keypoints,
            "scores":    scores,
            "group":     self.model.active_group,
        }

    @staticmethod
    def visualize(
        image: np.ndarray,
        results: Dict,
        skeleton: Optional[List[Tuple[int, int]]] = None,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
        colors_kp = plt.cm.plasma(np.linspace(0, 1, 17))

        for box, kps, score in zip(results["boxes"], results["keypoints"], results["scores"]):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      lw=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f"{score:.2f}", color='lime', fontsize=9)

            for j, (kx, ky, kv) in enumerate(kps):
                if kv > 0:
                    ax.scatter(kx, ky, c=[colors_kp[j % len(colors_kp)]], s=30, zorder=5)

            if skeleton:
                for i, j in skeleton:
                    if i < len(kps) and j < len(kps) and kps[i][2] > 0 and kps[j][2] > 0:
                        ax.plot([kps[i][0], kps[j][0]], [kps[i][1], kps[j][1]],
                                color='cyan', lw=1.5, alpha=0.8)

        ax.set_title(f"groupe : {results.get('group', 'N/A')} — {len(results['boxes'])} insecte(s)",
                     fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.show()