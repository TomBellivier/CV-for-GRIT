# Trois approches de spécialisation par groupe — YOLO26-pose, 42 keypoints

Trois projets indépendants, un socle commun, **un format de sortie strictement
identique** à celui de votre `train_eval_pose.py`.

```
common/              fichiers partagés — à copier dans chacun des 3 dossiers
lora_adapters/       approche 1 : adaptateurs bas-rang par groupe
group_batchnorm/     approche 2 : BatchNorm spécifiques par groupe
two_stage/           approche 3 : détection puis pose sur crop recadré
```

---

## 1. Installation

Les fichiers de `common/` doivent être visibles depuis chaque projet. Deux
options, au choix :

```bash
# option A — copie (la plus simple)
for d in lora_adapters group_batchnorm two_stage; do cp common/*.py "$d/"; done

# option B — PYTHONPATH
export PYTHONPATH="$PWD/common:$PYTHONPATH"
```

Puis :

```bash
cp common/groups.example.yaml groups.yaml   # et adapter les chemins
pip install ultralytics pandas openpyxl pyyaml opencv-python
```

---

## 2. Format de sortie — ce qui est garanti

Chaque script `eval_*.py` écrit un classeur `results_<run_tag>.xlsx` avec les
quatre mêmes feuilles et **exactement les mêmes colonnes, dans le même ordre** :

| Feuille | Colonnes |
|---|---|
| `metadata` | `field`, `value` |
| `summary` | `group`, `num_val_images`, `pose_map`, `pose_map50`, `pose_map75`, `box_map`, `box_map50`, `num_matched`, `mean_kpt_conf`, `mpjpe_px`, `nmpjpe`, `mean_oks`, `pck_0.05`, `pck_0.1`, `training_time_sec` |
| `per_keypoint` | `group`, `kpt_index`, `kpt_name`, `n_obs`, `kpt_conf`, `mpjpe_px`, `nmpjpe`, `pck_0.05`, `pck_0.1` |
| `learning_curves` | `group` + les colonnes de `results.csv` |

`common/pose_metrics.py` est un portage **verbatim** de vos fonctions : mêmes
seuils PCK, même `OKS_SIGMA = 0.05`, même appariement glouton par IoU, mêmes
formules. Les colonnes `mpjpe_px`, `nmpjpe`, `mean_oks`, `pck_*` sont donc
directement comparables à vos classeurs existants.

Les informations propres à chaque approche (rang LoRA, marge de crop, poids de
base…) sont ajoutées en **lignes supplémentaires de la feuille `metadata`**, ce
qui ne perturbe aucune comparaison de colonnes.

### Le seul point de vigilance : `pose_map` / `box_map`

Vos runs précédents utilisaient le validateur natif d'Ultralytics. Les
approches 1 et 2 l'utilisent aussi (le modèle reste un `YOLO` complet), donc ces
colonnes restent comparables telles quelles.

**Le pipeline two-stage ne peut pas** : aucun modèle Ultralytics unique ne
représente la chaîne détecteur + pose. `common/map_eval.py` réimplémente donc un
mAP COCO-style aligné sur les conventions d'Ultralytics :

- seuils 0.50:0.05:0.95, AP interpolé en 101 points ;
- OKS avec `e = d² / (8σ²·aire)`, aire = aire de boîte × 0.53 ;
- σ par défaut = `1/n_kpts`, ce qui est **précisément** ce qu'Ultralytics
  applique dès que `kpt_shape != [17, 3]` — donc ce avec quoi vos 42 keypoints
  ont déjà été notés.

La ligne `map_source` de la feuille `metadata` indique `native` ou `custom`.
La réimplémentation suit de près le validateur natif mais n'est pas garantie
numériquement identique. **Pour une comparaison à armes égales avec le
two-stage**, relancez les autres approches avec `--map-source custom` :

```bash
python eval_single.py    --weights base_model/weights/best.pt --map-source custom ...
python eval_lora.py      --manifest ... --map-source custom
python eval_group_bn.py  --manifest ... --map-source custom
```

Vous obtenez alors deux familles de classeurs : `native` (comparable à
l'historique) et `custom` (comparable entre les quatre approches).

> À noter : avec σ = 1/42, le `pose_map` est très sévère. Un écart moyen de 5 px
> sur une boîte de 120 px suffit à faire tomber l'AP à zéro. Sur votre problème,
> `nmpjpe` et `pck_0.05` seront des indicateurs bien plus lisibles que le mAP.

---

## 3. Ordre d'exécution

### Étape 0 — la baseline (indispensable)

Les approches 1 et 2 partent toutes deux d'un **modèle généraliste unique
entraîné sur les quatre groupes**. Ce modèle est aussi votre point de référence.

```bash
python common/train_base.py \
    --model yolo26n-pose.pt --data-config groups.yaml \
    --epochs 150 --batch 16 --imgsz 640 \
    --degrees 180 --mosaic 0.5 \
    --out-dir base_model --runs-dir runs_base

python common/eval_single.py \
    --weights runs_base/base/weights/best.pt \
    --data-config groups.yaml \
    --out-dir pose_results --run-tag baseline_shared
```

`train_base.py` fabrique automatiquement un `combined_pose.yaml` qui pointe vers
les quatre dossiers à la fois (YOLO accepte une liste de chemins) — aucune image
n'est dupliquée sur le disque. Il refuse de fusionner des groupes dont le
`kpt_shape` ou le `flip_idx` diffèrent.

### Étapes 1 à 3

Voir le README de chaque dossier. En résumé :

```bash
# approche 1
python lora_adapters/train_lora.py --base-weights runs_base/base/weights/best.pt ...
python lora_adapters/eval_lora.py  --manifest lora_weights/lora_manifest.json ...

# approche 2
python group_batchnorm/train_group_bn.py --base-weights runs_base/base/weights/best.pt ...
python group_batchnorm/eval_group_bn.py  --manifest gbn_weights/gbn_manifest.json ...

# approche 3
python two_stage/prepare_two_stage_dataset.py --data-config groups.yaml --out-dir two_stage_data
python two_stage/train_two_stage.py --det-data ... --pose-data ...
python two_stage/eval_two_stage.py  --manifest two_stage_weights/two_stage_manifest.json ...
```

---

## 4. Augmentation — le réglage le plus rentable

`--degrees 180` est activé par défaut partout. Vos vues dorsales n'ont pas
d'orientation canonique : la rotation dans le plan est une augmentation
gratuite et massive, et Ultralytics la laisse à 0 par défaut.

`--fliplr` est en revanche à **0 par défaut**, volontairement. Vos insectes ont
une symétrie bilatérale : sans `flip_idx` correct dans le `data.yaml` (la
permutation des 42 indices), un miroir horizontal apprend au modèle que la patte
antérieure gauche est parfois à droite. La dégradation est silencieuse. Ne
montez `--fliplr` qu'une fois le `flip_idx` vérifié.

`--mosaic` est abaissé à 0.5 : sur des macros, coller quatre images
redimensionnées détruit les statistiques d'échelle.

---

## 5. Vérifications utiles avant de conclure

**Le partage a-t-il tenu ?** L'intérêt des approches 1 et 2 tient entièrement au
fait que les poids de base sont réellement partagés. Deux garde-fous :

```bash
python group_batchnorm/train_group_bn.py ... --verify-shared
```

compare bit à bit les poids non-BN des quatre checkpoints. Et côté LoRA,
`eval_lora.py` lève une erreur explicite si les tenseurs d'adaptateurs ne
correspondent pas au modèle injecté.

**Validation croisée.** Avec ~40 images d'hyménoptères en validation, un split
unique donne des métriques très bruitées. Faites tourner chaque approche sur 5
plis stratifiés par groupe (5 fichiers `groups.yaml` pointant vers 5 splits) et
comparez les moyennes ± écarts-types, pas les valeurs brutes.

**Regardez par groupe, jamais la moyenne.** Les hyménoptères (200 images) sont
le point de rupture de toutes ces méthodes. Une moyenne globale les noiera.

---

## 6. Ce que ces scripts ne font pas

- Ils n'implémentent pas de recherche d'hyperparamètres. Les valeurs par défaut
  sont des points de départ raisonnables, pas des optima.
- `train_base.py`, `train_lora.py` et `train_group_bn.py` supposent que chaque
  dataset de groupe est **mono-classe avec l'indice 0**. Si vos labels portent
  des indices de classe différents par groupe, la fusion produira des labels
  incohérents.
- L'approche two-stage n'implémente pas de pondération d'échantillonnage pour
  compenser le déséquilibre 800/500/500/200. Si le détecteur sous-performe sur
  les hyménoptères, c'est le premier levier à ajouter.
