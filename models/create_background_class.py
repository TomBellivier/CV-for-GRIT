"""Genere la classe `background` : memes images, insecte efface.

Ces images servent DEUX fois :
  1. comme 5e classe du dataset de classification -> elle donne a `cls` la
     capacite de S'ABSTENIR, que `detect` et `pose` ont nativement (aucune
     boite au-dessus du seuil). Sans elle, la comparaison des trois modeles
     est biaisee : `cls` est force de choisir un groupe meme sur une image
     vide.
  2. comme jeu CONTRE-FACTUEL au moment de l'evaluation.

CORRECTIONS PAR RAPPORT A LA VERSION D'ORIGINE
----------------------------------------------
1. SEULE LA PREMIERE INSTANCE ETAIT EFFACEE (`lines[0]`). Sur une image a
   plusieurs insectes, les autres restaient visibles : les images "background"
   contenaient encore des insectes, ce qui pollue a la fois l'entrainement et
   le test contre-factuel. Toutes les lignes sont maintenant traitees.

2. DEBORDEMENT uint8 sur le bruit :
   `(rand * mean*0.1 + mean + 0.9).astype(np.uint8)` repasse a 0 quand la somme
   depasse 255 -> pixels noirs parasites sur fond clair. Ajout d'un `np.clip`.

3. BOITES DEGENEREES : si `x1 == x2` ou `y1 == y2`, les tranches de bordure
   sont vides, `np.mean` renvoie NaN, et `astype(np.uint8)` produit une couleur
   arbitraire. Ces boites sont maintenant ignorees proprement.

4bis. (voir CORRECTIONS 10 et 11 dans le code : contamination entre insectes
   voisins, revelee par une image a deux instances.)

4. IMAGES RGBA / niveaux de gris : `np.asarray` renvoyait 4 ou 1 canal alors
   que le bruit etait construit en 3 canaux -> exception ou couleurs fausses.
   Conversion en RGB systematique.

5. `tuple(mean_color)` passait des `np.uint8` a PIL (rejete par certaines
   versions) -> conversion en `int` natif.

6. CHEMIN ABSOLU EN DUR (`C:/Users/tombe/...`) supprime : la ligne etait de
   toute facon redondante avec le `mkdir` de la boucle.

7. DESEQUILIBRE DE CLASSES : le script produisait une image `background` par
   image du dataset, donc autant de background que TOUS les groupes reunis
   (~50 % du dataset pour une seule classe). Le modele apprend alors a repondre
   `background` par defaut. Parametre `background_ratio` pour sous-echantillonner.

8. NOUVEAU — plusieurs methodes d'effacement (voir ci-dessous).

9. NOUVEAU — sortie letterboxee, coherente avec `create_cls_dataset`.

POURQUOI PLUSIEURS METHODES
---------------------------
Ta methode d'origine (couleur moyenne + bruit + flou) ne reconstruit pas le
fond : elle le remplace par un aplat bruite. C'est tres proche d'un CONTROLE
« trou ». Consequence : si un modele echoue sur ces images, on ne sait pas s'il
avait besoin de l'insecte ou s'il reagit simplement a l'artefact.

Pire, pour `cls` : ces images sont dans son ENTRAINEMENT. Il peut apprendre la
signature de l'artefact au lieu d'apprendre l'absence d'insecte.

Le diagnostic tient en une comparaison. Genere deux variantes :
  - `mean_noise` (ta methode) -> entre dans le dataset cls
  - `telea` (inpainting OpenCV) -> N'ENTRE PAS dans l'entrainement, sert
    uniquement au test contre-factuel
Si `cls` s'abstient sur `mean_noise` mais pas sur `telea`, il a appris
l'artefact, pas l'absence d'insecte. Le notebook fait ce test automatiquement.
"""

from pathlib import Path

import numpy as np
import tqdm
from PIL import Image, ImageFilter

# ======================================================================
DATASET_DIR = Path("./models/datasets/")
POSE_DATASET = DATASET_DIR / "AllSpecies-pose"
CLS_DATASET = DATASET_DIR / "AllSpecies-cls"
COUNTERFACTUAL_DIR = DATASET_DIR / "counterfactual"

IMGSZ = 640                  # identique a fuze_datasets.py et au notebook
LETTERBOX = True
LETTERBOX_COLOR = (114, 114, 114)
SPLITS = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
BACKGROUND_RATIO = 0.35      # cf. correction 7
SEED = 0


# ======================================================================
def letterbox_pil(img: Image.Image, size: int = IMGSZ) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    canvas = Image.new("RGB", (size, size), LETTERBOX_COLOR)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    return canvas


def read_boxes(label_path: Path, width: int, height: int):
    """Toutes les instances -> boites pixel. CORRECTION 1 : plus seulement la premiere."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        x, y, w, h = map(float, parts[1:5])
        x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
        x2, y2 = int((x + w / 2) * width), int((y + h / 2) * height)
        x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
        x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
        if x2 - x1 >= 2 and y2 - y1 >= 2:     # CORRECTION 3
            boxes.append((x1, y1, x2, y2))
    return boxes


def boxes_mask(shape, boxes) -> np.ndarray:
    m = np.zeros(shape[:2], bool)
    for (x1, y1, x2, y2) in boxes:
        m[y1:y2, x1:x2] = True
    return m


def border_mean_color(arr: np.ndarray, box, exclude: np.ndarray | None = None,
                      margin: int = 1):
    """Couleur moyenne des 4 bords de la boite.

    CORRECTION 10 (revelee par une image a deux insectes) : quand deux insectes
    sont proches, la bande de bord de la premiere boite traverse la seconde. La
    couleur de remplissage devenait alors celle de l'insecte voisin, et la zone
    "effacee" restait coloree comme un insecte. Les pixels appartenant a une
    AUTRE boite sont maintenant exclus du calcul.
    """
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    top, left = max(y1 - margin, 0), max(x1 - margin, 0)
    bottom, right = min(y2 + margin, h - 1), min(x2 + margin, w - 1)
    if exclude is None:
        exclude = np.zeros((h, w), bool)
    strips = [(arr[top, left:right], exclude[top, left:right]),
              (arr[bottom, left:right], exclude[bottom, left:right]),
              (arr[top:bottom, left], exclude[top:bottom, left]),
              (arr[top:bottom, right], exclude[top:bottom, right])]
    means = []
    for pix, bad in strips:
        if pix.size == 0:
            continue                                    # CORRECTION 3
        keep = pix.reshape(-1, 3)[~bad.reshape(-1)]
        if keep.size:
            means.append(keep.mean(0))
    if not means:
        return np.array(LETTERBOX_COLOR, float)
    return np.mean(means, axis=0)


# ======================================================================
# Methodes d'effacement
# ======================================================================
def erase_mean_noise(img: Image.Image, boxes, rng, blur_radius: int = 20,
                     blur_margin: int = 20) -> Image.Image:
    """Methode d'origine, corrigee : aplat de couleur de bord + bruit + flou.

    CORRECTION 11 : les remplissages sont TOUS faits avant les flous. Dans la
    version d'origine, flouter le voisinage de la premiere boite pendant que la
    seconde etait encore visible ramenait la couleur de l'insecte voisin dans
    la zone deja effacee.
    """
    img = img.convert("RGB").copy()          # CORRECTION 4
    w, h = img.size
    arr = np.asarray(img)
    other = boxes_mask(arr.shape, boxes)

    for (x1, y1, x2, y2) in boxes:           # passe 1 : remplissage
        mean = border_mean_color(arr, (x1, y1, x2, y2), exclude=other)
        img.paste(tuple(int(c) for c in mean), (x1, y1, x2, y2))   # CORRECTION 5
        noise = np.clip(                                            # CORRECTION 2
            rng.random((y2 - y1, x2 - x1, 3)) * (mean * 0.1) + mean,
            0, 255).astype(np.uint8)
        img.paste(Image.fromarray(noise), (x1, y1, x2, y2))

    for (x1, y1, x2, y2) in boxes:           # passe 2 : raccord des bords
        bx = (max(x1 - blur_margin, 0), max(y1 - blur_margin, 0),
              min(x2 + blur_margin, w), min(y2 + blur_margin, h))
        img.paste(img.crop(bx).filter(ImageFilter.GaussianBlur(radius=blur_radius)), bx)
    return img


def erase_telea(img: Image.Image, boxes, rng) -> Image.Image:
    """Inpainting OpenCV : reconstruit le fond au lieu de le remplacer.

    Variante de CONTRASTE : a garder hors de l'entrainement de `cls` pour
    pouvoir detecter l'apprentissage de l'artefact (voir en-tete).
    """
    import cv2

    bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)
    for (x1, y1, x2, y2) in boxes:
        mask[y1:y2, x1:x2] = 255
    out = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))


def erase_gray(img: Image.Image, boxes, rng) -> Image.Image:
    """Controle pur : aplat gris, aucune reconstruction, aucun raccord."""
    img = img.convert("RGB").copy()
    for box in boxes:
        img.paste(LETTERBOX_COLOR, box)
    return img


METHODS = {"mean_noise": erase_mean_noise, "telea": erase_telea, "gray": erase_gray}


# ======================================================================
def generate(method: str, out_root: Path, subdir: str = "", splits=SPLITS,
             ratio: float = 1.0, letterbox: bool = LETTERBOX, seed: int = SEED,
             skip_existing: bool = True):
    """Ecrit `out_root/<split>/<subdir>/*.png`.

    - classe background du dataset cls : out_root=AllSpecies-cls, subdir="background"
    - jeu contre-factuel                : out_root=counterfactual/<methode>, subdir=""
    """
    fn = METHODS[method]
    rng = np.random.default_rng(seed)
    total = kept = 0

    for split in splits:
        imdir = POSE_DATASET / "images" / split
        if not imdir.exists():
            print(f"  {split} : absent, ignore")
            continue
        images = sorted(p for p in imdir.iterdir() if p.suffix.lower() in IMG_EXT)

        if ratio < 1.0:                                  # CORRECTION 7
            n = max(1, int(len(images) * ratio))
            idx = np.random.default_rng(seed).permutation(len(images))[:n]
            images = [images[i] for i in sorted(idx)]

        out_dir = Path(out_root) / split / subdir if subdir else Path(out_root) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm.tqdm(images, desc=f"{method}/{split}"):
            total += 1
            dst = out_dir / f"{img_path.stem}.png"
            if skip_existing and dst.exists():
                kept += 1
                continue
            img = Image.open(img_path).convert("RGB")    # CORRECTION 4
            boxes = read_boxes(POSE_DATASET / "labels" / split / f"{img_path.stem}.txt",
                               *img.size)
            out = fn(img, boxes, rng) if boxes else img  # sans boite : image telle quelle
            if letterbox:
                out = letterbox_pil(out, IMGSZ)
            out.save(dst)
            kept += 1
    print(f"{method} : {kept}/{total} images -> {out_root}")


def report_balance():
    """Verifie l'equilibre des classes du dataset cls (CORRECTION 7)."""
    print("\nequilibre du dataset de classification :")
    for split in SPLITS:
        d = CLS_DATASET / split
        if not d.exists():
            continue
        counts = {p.name: sum(1 for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
                  for p in sorted(d.iterdir()) if p.is_dir()}
        tot = sum(counts.values()) or 1
        print(f"  {split:<6} " + "  ".join(f"{k}={v} ({100*v/tot:.0f}%)"
                                           for k, v in counts.items()))
        if counts.get("background", 0) > 0.4 * tot:
            print("    ATTENTION : background > 40 % du split. Baisser "
                  "BACKGROUND_RATIO, sinon le modele repondra background par defaut.")


if __name__ == "__main__":
    # 1. Classe `background` du dataset de classification.
    #    Sous-echantillonnee (BACKGROUND_RATIO) pour ne pas ecraser les 3 groupes.
    #    Methode `mean_noise` : c'est celle qui entre dans l'ENTRAINEMENT de cls.
    generate("mean_noise", CLS_DATASET, subdir="background", ratio=BACKGROUND_RATIO)

    # 2. Jeux contre-factuels, split test uniquement, sans sous-echantillonnage.
    #    `mean_noise` = meme methode que l'entrainement (cls l'a deja vue)
    #    `telea`      = reconstruction, JAMAIS vue a l'entrainement
    #    `gray`       = controle pur, aucun raccord
    #    L'ecart entre les trois est ce qui rend le test interpretable.
    for method in ("mean_noise", "telea", "gray"):
        generate(method, COUNTERFACTUAL_DIR / method, splits=("test",), ratio=1.0)

    report_balance()