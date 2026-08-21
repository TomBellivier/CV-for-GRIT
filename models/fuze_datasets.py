"""Fusion de datasets par groupe -> datasets pose / detect / cls unifies.

CORRECTIONS PAR RAPPORT A LA VERSION D'ORIGINE
----------------------------------------------
1. COLLISIONS DE NOMS (bug silencieux, probablement la source de tes ecarts) :
   la deduplication testait `if label_file in all_label_files`, c'est-a-dire des
   chemins ABSOLUS, toujours differents d'un dataset source a l'autre. Deux
   fichiers `0001.txt` venant de Coleoptera et de Lepidoptera passaient donc le
   test, puis `shutil.copy` ecrasait le premier par le second. Resultat : moins
   d'images que prevu et des labels attribues au mauvais groupe. La dedup se
   fait maintenant sur le NOM, et les collisions sont resolues par un prefixe
   `<dataset>__` plutot que par un ecrasement.

2. IMAGES ET LABELS DEDUPLIQUES SEPAREMENT : une image pouvait etre copiee sans
   son label (ou l'inverse). Les deux sont maintenant traites par paire, et un
   controle de coherence est affiche a la fin.

3. `flip_idx` : si `filter_keywords` supprimait un cote sans son symetrique
   (ex. `["left-antenna"]`), `base_idx[x]` levait un KeyError. Les keypoints
   dont le miroir a disparu pointent maintenant vers eux-memes.

4. `filter` masquait le builtin Python du meme nom -> renomme `filter_names`.

5. `config["path"]` etait un chemin relatif reconstruit a la main
   (`"models/datasets/" + name`), resolu par Ultralytics par rapport a son
   propre repertoire de datasets et non au CWD. Il est maintenant absolu.

6. `increment_path` etait applique inconditionnellement : impossible de
   regenerer un dataset en place, on accumulait `-detect2`, `-detect3`...
   Controle par `increment=`.

7. NOUVEAU — `create_cls_dataset` ecrit les images LETTERBOXEES en carre.
   Ultralytics applique `Resize + CenterCrop` en classification : sur une image
   deja carree, cette transformation devient l'identite. Sans cela, le modele
   `cls` voit un recadrage central que `detect`/`pose` ne voient pas, et
   comparer leurs cartes de saillance n'a plus de sens.

8. NOUVEAU — `write_cls_names()` ecrit `classes.txt`. Ultralytics ordonne les
   classes de classification par ORDRE ALPHABETIQUE des dossiers, pas par
   l'ordre de `dataset_list`. Avec des groupes en minuscules, `background`
   passerait en premier et decalerait tous les indices. Le notebook lit
   `model.names` et remappe, mais ce fichier documente l'ordre attendu.
"""

from pathlib import Path
import shutil

import tqdm
import yaml
from PIL import Image
from ultralytics.utils.files import increment_path

IMGSZ = 640              # doit valoir la meme chose que dans le notebook
LETTERBOX_COLOR = (114, 114, 114)

TOTAL = [
    "head-top", "head-left", "head-right", "left-eye", "right-eye", "neck",
    "thorax-left", "thorax-right", "thorax-bottom",
    "body-left", "body-right", "body-tip",
    "left-antenna-0", "left-antenna-1", "left-antenna-2",
    "right-antenna-0", "right-antenna-1", "right-antenna-2",
    "left-forewing-base", "left-forewing-tip", "left-forewing-front", "left-forewing-rear",
    "right-forewing-base", "right-forewing-tip", "right-forewing-front", "right-forewing-rear",
    "left-hindwing-base", "left-hindwing-tip", "left-hindwing-front", "left-hindwing-rear",
    "right-hindwing-base", "right-hindwing-tip", "right-hindwing-front", "right-hindwing-rear",
    "left-leg-0", "left-leg-1", "left-leg-2", "left-leg-3",
    "right-leg-0", "right-leg-1", "right-leg-2", "right-leg-3",
]

SKELETON_NAMES = [
    ["head-top", "neck"], ["head-top", "left-antenna-0"], ["head-top", "right-antenna-0"],
    ["left-antenna-0", "left-antenna-1"], ["left-antenna-1", "left-antenna-2"],
    ["right-antenna-0", "right-antenna-1"], ["right-antenna-1", "right-antenna-2"],
    ["neck", "head-left"], ["neck", "head-right"], ["neck", "left-eye"], ["neck", "right-eye"],
    ["neck", "thorax-left"], ["neck", "thorax-right"], ["neck", "thorax-bottom"],
    ["thorax-left", "thorax-bottom"], ["thorax-right", "thorax-bottom"],
    ["thorax-bottom", "body-left"], ["thorax-bottom", "body-right"], ["thorax-bottom", "body-tip"],
    ["body-left", "body-tip"], ["body-right", "body-tip"],
    ["thorax-bottom", "left-forewing-base"],
    ["left-forewing-base", "left-forewing-front"], ["left-forewing-base", "left-forewing-rear"],
    ["left-forewing-tip", "left-forewing-front"], ["left-forewing-tip", "left-forewing-rear"],
    ["thorax-bottom", "right-forewing-base"],
    ["right-forewing-base", "right-forewing-front"], ["right-forewing-base", "right-forewing-rear"],
    ["right-forewing-tip", "right-forewing-front"], ["right-forewing-tip", "right-forewing-rear"],
    ["thorax-bottom", "left-hindwing-base"],
    ["left-hindwing-base", "left-hindwing-front"], ["left-hindwing-base", "left-hindwing-rear"],
    ["left-hindwing-tip", "left-hindwing-front"], ["left-hindwing-tip", "left-hindwing-rear"],
    ["thorax-bottom", "right-hindwing-base"],
    ["right-hindwing-base", "right-hindwing-front"], ["right-hindwing-base", "right-hindwing-rear"],
    ["right-hindwing-tip", "right-hindwing-front"], ["right-hindwing-tip", "right-hindwing-rear"],
    ["thorax-bottom", "left-leg-0"], ["body-left", "left-leg-1"],
    ["left-leg-0", "left-leg-1"], ["left-leg-1", "left-leg-2"], ["left-leg-2", "left-leg-3"],
    ["thorax-bottom", "right-leg-0"], ["body-right", "right-leg-1"],
    ["right-leg-0", "right-leg-1"], ["right-leg-1", "right-leg-2"], ["right-leg-2", "right-leg-3"],
]

SPLITS = ("train", "val", "test")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# ======================================================================
# Letterbox (identique a celui du notebook)
# ======================================================================
def letterbox_pil(img: Image.Image, size: int = IMGSZ) -> Image.Image:
    """Redimensionne en conservant le ratio, complete en carre avec du gris."""
    img = img.convert("RGB")
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    canvas = Image.new("RGB", (size, size), LETTERBOX_COLOR)
    canvas.paste(img.resize((nw, nh), Image.BILINEAR),
                 ((size - nw) // 2, (size - nh) // 2))
    return canvas


# ======================================================================
# Configuration pose / detect
# ======================================================================
def check_filter(kp_name, keywords):
    return not any(kw in kp_name for kw in keywords)


def filter_names(all_kps, keywords):
    """Filtre recursif. Une paire de squelette n'est gardee que si ses deux
    extremites survivent. (Renomme : masquait le builtin `filter`.)"""
    out = []
    for element in all_kps:
        if isinstance(element, list):
            kept = filter_names(element, keywords)
            if len(kept) == len(element):
                out.append(kept)
        elif isinstance(element, str) and check_filter(element, keywords):
            out.append(element)
    return out


def build_flip_idx(filtered_total, base_idx):
    """Indices de symetrie gauche/droite.

    CORRECTION : si le miroir d'un keypoint a ete filtre, il pointe vers
    lui-meme au lieu de lever un KeyError.
    """
    flip = []
    for name in filtered_total:
        if "left" in name:
            mirror = name.replace("left", "right")
        elif "right" in name:
            mirror = name.replace("right", "left")
        else:
            mirror = name
        flip.append(base_idx.get(mirror, base_idx[name]))
    return flip


def make_pose_config_file(dataset_dir, filter_keywords=(), cls_groups=(), printing=False):
    dataset_dir = Path(dataset_dir)
    filtered_total = filter_names(TOTAL, filter_keywords)
    filtered_skeleton = filter_names(SKELETON_NAMES, filter_keywords)
    base_idx = {name: i for i, name in enumerate(filtered_total)}

    config = {
        "path": str(dataset_dir.resolve()),          # CORRECTION : absolu
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [len(filtered_total), 3],
        "skeleton": [[base_idx[a], base_idx[b]] for a, b in filtered_skeleton],
        "flip_idx": build_flip_idx(filtered_total, base_idx),
        "names": ({i: g for i, g in enumerate(cls_groups)} if cls_groups
                  else {0: "insects"}),
        "kpt_names": ({i: filtered_total for i in range(len(cls_groups))} if cls_groups
                      else {0: filtered_total}),
    }
    if printing:
        print("kpt_shape:", config["kpt_shape"], "| flip_idx:", config["flip_idx"])

    path = dataset_dir / "yolo-config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)
    print(f"config -> {path}")
    return path


def make_detect_config_file(dataset_dir, groups):
    dataset_dir = Path(dataset_dir)
    config = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: g for i, g in enumerate(groups)},
    }
    path = dataset_dir / "yolo-config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=None)
    print(f"config -> {path}")
    return path


# ======================================================================
# Copie avec gestion des collisions
# ======================================================================
def _unique_name(name: str, dataset: str, taken: set) -> str:
    """CORRECTION du bug d'ecrasement silencieux.

    Deux datasets sources peuvent contenir `0001.png`. Avant, le second
    ecrasait le premier. On prefixe desormais par le nom du dataset, et on
    numerote en dernier recours.
    """
    if name not in taken:
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate = f"{dataset}__{stem}{suffix}"
    k = 1
    while candidate in taken:
        candidate = f"{dataset}__{stem}_{k}{suffix}"
        k += 1
    print(f"  collision : {name} -> {candidate}")
    return candidate


def _rewrite_label(src, dst, new_class_id, keep_kpts):
    lines = []
    for line in Path(src).read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        parts = parts if keep_kpts else parts[:5]
        parts[0] = str(new_class_id)
        lines.append(" ".join(parts))
    Path(dst).write_text("\n".join(lines) + ("\n" if lines else ""))


def _copy_geometric(dataset_list, final_folder, dataset_folder, keep_kpts, remap_class):
    """Copie images + labels APPARIES (correction : plus de dedup separee)."""
    final_folder = Path(final_folder)
    taken = {s: set() for s in SPLITS}
    stats = {}

    for cid, dataset in enumerate(tqdm.tqdm(dataset_list, colour="red", desc="datasets")):
        dpath = Path(dataset_folder) / dataset
        n_ok = n_skip = 0
        for split in SPLITS:
            imdir, lbdir = dpath / "images" / split, dpath / "labels" / split
            if not imdir.exists():
                continue
            for img in tqdm.tqdm(sorted(imdir.iterdir()), colour="blue",
                                 desc=f"{dataset}/{split}", leave=False):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = lbdir / f"{img.stem}.txt"
                if not lbl.exists():        # CORRECTION : plus d'image orpheline
                    n_skip += 1
                    continue
                new_name = _unique_name(img.name, dataset, taken[split])
                taken[split].add(new_name)
                shutil.copy(img, final_folder / "images" / split / new_name)
                _rewrite_label(lbl,
                               final_folder / "labels" / split / f"{Path(new_name).stem}.txt",
                               cid if remap_class else 0, keep_kpts)
                n_ok += 1
        stats[dataset] = (n_ok, n_skip)

    print("\nrecapitulatif :")
    for d, (ok, skip) in stats.items():
        print(f"  {d:<22} {ok:5d} paires copiees"
              + (f"  ({skip} images sans label, ignorees)" if skip else ""))
    return stats


def create_pose_dataset(dataset_list, final_folder, dataset_folder="./models/datasets/",
                        cls=False, filter_keywords=()):
    _copy_geometric(dataset_list, final_folder, dataset_folder,
                    keep_kpts=True, remap_class=cls)
    make_pose_config_file(final_folder, filter_keywords,
                          cls_groups=dataset_list if cls else ())


def create_detect_dataset(dataset_list, final_folder, dataset_folder="./models/datasets/"):
    _copy_geometric(dataset_list, final_folder, dataset_folder,
                    keep_kpts=False, remap_class=True)
    make_detect_config_file(final_folder, dataset_list)


def create_cls_dataset(dataset_list, final_folder, dataset_folder="./models/datasets/",
                       letterbox=True, imgsz=IMGSZ):
    """Arborescence `split/groupe/*.png`.

    NOUVEAU : `letterbox=True` ecrit des images carrees. Ultralytics applique
    `Resize + CenterCrop` en classification ; sur une image deja carree, cette
    transformation devient l'identite, et le modele `cls` voit exactement le
    meme cadrage que `detect`/`pose`. Sans cela, comparer leurs cartes de
    saillance revient a comparer deux cadrages differents.
    """
    final_folder = Path(final_folder)
    taken = {s: set() for s in SPLITS}
    total = 0
    for dataset in tqdm.tqdm(dataset_list, colour="red", desc="datasets"):
        dpath = Path(dataset_folder) / dataset
        for split in SPLITS:
            imdir = dpath / "images" / split
            if not imdir.exists():
                continue
            dst_dir = final_folder / split / dataset
            dst_dir.mkdir(parents=True, exist_ok=True)
            for img in tqdm.tqdm(sorted(imdir.iterdir()), colour="blue",
                                 desc=f"{dataset}/{split}", leave=False):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                new_name = _unique_name(img.name, dataset, taken[split])
                taken[split].add(new_name)
                if letterbox:
                    letterbox_pil(Image.open(img), imgsz).save(dst_dir / new_name)
                else:
                    shutil.copy(img, dst_dir / new_name)
                total += 1
    write_cls_names(final_folder, dataset_list)
    print(f"\n{total} images copiees" + (" (letterboxees)" if letterbox else ""))


def write_cls_names(final_folder, dataset_list):
    """Documente l'ordre des classes.

    PIEGE : Ultralytics ordonne les classes de classification par ordre
    ALPHABETIQUE des dossiers, pas par l'ordre de `dataset_list`. Si tes
    groupes sont en minuscules, `background` passe en premier et decale tous
    les indices. Le notebook lit `model.names` et remappe, donc il est
    immunise, mais ce fichier permet de verifier a l'oeil.
    """
    final_folder = Path(final_folder)
    observed = sorted({p.name for split in SPLITS
                       for p in (final_folder / split).glob("*") if p.is_dir()})
    (final_folder / "classes.txt").write_text("\n".join(observed) + "\n")
    print(f"ordre alphabetique effectif des classes : {observed}")
    if [g for g in observed if g in dataset_list] != list(dataset_list):
        print("  ATTENTION : cet ordre differe de dataset_list "
              f"{list(dataset_list)}. Le notebook remappe via model.names.")


# ======================================================================
def fuze(dataset_name, dataset_list, dataset_folder="./models/datasets/",
         erase=False, task="pose", increment=True, letterbox_cls=True):
    f = Path(dataset_folder) / dataset_name
    if increment:
        f = Path(increment_path(f))       # CORRECTION : desormais optionnel
    print(f"destination : {f.resolve()}")

    if task in ("pose", "pose+cls", "detect"):
        for sub in ("labels", "images"):
            for split in SPLITS:
                (f / sub / split).mkdir(parents=True, exist_ok=True)
    elif task == "cls":
        for split in SPLITS:
            for dataset in dataset_list:
                (f / split / dataset).mkdir(parents=True, exist_ok=True)

    if task == "pose":
        create_pose_dataset(dataset_list, f, dataset_folder)
    elif task == "pose+cls":
        create_pose_dataset(dataset_list, f, dataset_folder, cls=True)
    elif task == "detect":
        create_detect_dataset(dataset_list, f, dataset_folder)
    elif task == "cls":
        create_cls_dataset(dataset_list, f, dataset_folder, letterbox=letterbox_cls)
    else:
        raise ValueError(f"task '{task}' non supportee")

    if erase and input("Supprimer les datasets sources ? (y/n) ") == "y":
        for dataset in dataset_list:
            shutil.rmtree(Path(dataset_folder) / dataset)
    return f


if __name__ == "__main__":
    GROUPS = ["coleoptera", "diptera", "hymenoptera", "lepidoptera"]
    ROOT = "./models/datasets/"

    # `pose+cls` et non `pose` : le modele pose doit avoir 4 classes (une par
    # groupe) pour produire une prediction image-level comparable a `cls`.
    fuze("AllSpecies-pose",   GROUPS, ROOT, task="pose+cls", increment=False)
    fuze("AllSpecies-detect", GROUPS, ROOT, task="detect",   increment=False)
    fuze("AllSpecies-cls",    GROUPS, ROOT, task="cls",      increment=False)
    # puis : python create_background_class.py