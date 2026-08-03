
# insectpose

Socle experimental pour comparer plusieurs approches d'estimation de pose sur 4 datasets
d'insectes (Coleoptera, Diptera, Hymenoptera, Lepidoptera).

**Lire `CONVENTIONS.md` avant toute contribution.** Ce fichier-ci ne contient que le demarrage
rapide ; toute la doctrine (contrats, regles, protocole) est dans `CONVENTIONS.md`, qui fait foi.

## Installation

```bash
pip install -e ".[dev]"
```

torch et ultralytics sont des dependances de premier rang (ADR-0019). `train.device: auto`
utilise le GPU 0 s'il est disponible, la precision mixte est active par defaut et desactivee
en `mode: debug`. Le materiel resolu est enregistre dans chaque manifeste, et l'agregation
alerte si des runs compares proviennent de materiels differents.

## Chaine complete

```bash
# 1. raw -> format canonique (contrat 1)
python -m insectpose.cli prepare data=coleoptera

# 2. folds partages par TOUTES les approches (contrat 2)
python -m insectpose.cli split cv=kfold5_grouped

# 3. entrainement + prediction + evaluation d'un fold
#    experiences disponibles : exp_a_yolo_pooled, exp_b_yolo_per_dataset,
#    exp_c_detect_then_pose, exp_ref_mean_pose
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0

# 4. optimisation Optuna (metrique primaire de configs/eval/default.yaml)
python -m insectpose.cli tune experiment=exp_a_yolo_pooled

# 5. agregation de tous les runs + tableaux
python -m insectpose.cli report
```

## Ajouter une approche

Copier `src/insectpose/approaches/TEMPLATE.py.txt` et suivre `CONVENTIONS.md` §11
(6 artefacts). Le smoke test est parametre sur le registre : une approche enregistree
y entre automatiquement, sans modifier `tests/`.

## Etat

Socle generique complet (contrats, registre, splits, evaluation, tuning niche, reporting, CLI).

| Approche                | Etat                                                                          |
| ----------------------- | ----------------------------------------------------------------------------- |
| `mean_pose`           | implementee - reference et baseline plancher (bbox GT, diagnostic)            |
| `yolo_pooled`         | **implementee** - GPU CUDA, AMP, FP16 a l'inference                     |
| `yolo_per_dataset`    | **implementee** - N modeles routes par dataset (ADR-0023)               |
| `detect_then_pose`    | **implementee** - detecteur poule + pose sur crop (ADR-0024)            |
| `lora`                | **implementee** - adaptateurs sur le cou, tetes entrainables (ADR-0025) |
| `group_bn`            | **implementee** - BatchNorm par dataset, lots mixtes (ADR-0026)         |
| `yolo_pooled_reduced` | **implementee** - A sans pattes ni ailes posterieures (ADR-0027)        |

Une approche dont la dependance lourde est absente est **ignoree** par le smoke test
(mecanisme `availability()`), jamais en echec.

Chaque run produit : `manifest.json`, `config.yaml` resolu, `predictions/`,
`metrics.parquet`, `logs/` et `figures/` (12 exemples pred vs GT dont les 6 pires cas).

### A quoi sert `mean_pose`

Ce n'est pas un modele candidat : elle predit, pour chaque instance, la **pose moyenne du
train** replacee dans la bbox de verite terrain. Elle remplit trois roles :

1. **baseline plancher** - un modele entraine qui ne la depasse pas nettement a un probleme
   (convergence, labels mal formes, ordre de keypoints errone). C'est un test de sanite, pas
   un concurrent ;
2. **gabarit** - c'est l'implementation de reference du protocole `Approach`, a copier pour
   ecrire une nouvelle approche ;
3. **smoke test** - elle s'execute en quelques secondes, sans GPU ni dependance lourde, ce qui
   valide toute la chaine `train -> predict -> evaluate -> figures` a chaque `pytest -m smoke`.

Elle utilise les bboxes GT (`bbox_source: gt`) : ses chiffres ne sont donc **pas comparables**
a ceux des approches bout-en-bout et ne doivent jamais figurer dans le meme tableau (§9.3).

L'approche `mean_pose` est une **implementation de reference** : elle sert de gabarit, de test
de bout en bout et de baseline plancher. Elle utilise les bboxes GT (`bbox_source: gt`) et n'est
donc **pas comparable** aux approches bout-en-bout (cf. §9.3).

## Protocole fige

Toutes les decisions de protocole sont tranchees (ADR-0006 a 0015, cf. `DECISIONS.md`) :

| Point             | Valeur                                                               |
| ----------------- | -------------------------------------------------------------------- |
| Keypoints         | `insect42_v1` : 42 points, commun aux 4 datasets, union = identite |
| Sigmas OKS        | `difficulty x 0.0025` (10 -> 0.025 ... 40 -> 0.100)                |
| PCK               | `alpha x largeur du thorax`, reference alpha = 0.25                |
| Metrique primaire | `oks_ap` (surchargeable sans reevaluer)                            |
| Mesures           | 27 mesures morphometriques + 9 paires de symetrie                    |
| Folds             | 5 externes, group_id = image_id (une image = un specimen)            |
| HPO               | nichee : recherche sur folds internes, test externe jamais vu        |
| Resolution        | 640x640 pour toutes les approches (garde-fou strict)                 |

**Cout de l'HPO nichee** : `n_folds x n_trials x inner_folds` entrainements par approche,
soit 600 aux valeurs par defaut. A calibrer avant de lancer une approche lourde, puis a figer
a l'identique pour toutes (equite du budget

# insectpose

Socle experimental pour comparer plusieurs approches d'estimation de pose sur 4 datasets
d'insectes (Coleoptera, Diptera, Hymenoptera, Lepidoptera).

**Lire `CONVENTIONS.md` avant toute contribution.** Ce fichier-ci ne contient que le demarrage
rapide ; toute la doctrine (contrats, regles, protocole) est dans `CONVENTIONS.md`, qui fait foi.

## Installation

```bash
pip install -e ".[dev]"
```

torch et ultralytics sont des dependances de premier rang (ADR-0019). `train.device: auto`
utilise le GPU 0 s'il est disponible, la precision mixte est active par defaut et desactivee
en `mode: debug`. Le materiel resolu est enregistre dans chaque manifeste, et l'agregation
alerte si des runs compares proviennent de materiels differents.

## Chaine complete

```bash
# 1. raw -> format canonique (contrat 1)
python -m insectpose.cli prepare data=coleoptera

# 2. folds partages par TOUTES les approches (contrat 2)
python -m insectpose.cli split cv=kfold5_grouped

# 3. entrainement + prediction + evaluation d'un fold
#    experiences disponibles : exp_a_yolo_pooled, exp_b_yolo_per_dataset,
#    exp_c_detect_then_pose, exp_ref_mean_pose
python -m insectpose.cli train experiment=exp_a_yolo_pooled cv.fold=0

# 4. optimisation Optuna (metrique primaire de configs/eval/default.yaml)
python -m insectpose.cli tune experiment=exp_a_yolo_pooled

# 5. agregation de tous les runs + tableaux
python -m insectpose.cli report
```

## Ajouter une approche

Copier `src/insectpose/approaches/TEMPLATE.py.txt` et suivre `CONVENTIONS.md` §11
(6 artefacts). Le smoke test est parametre sur le registre : une approche enregistree
y entre automatiquement, sans modifier `tests/`.

## Etat

Socle generique complet (contrats, registre, splits, evaluation, tuning niche, reporting, CLI).

| Approche             | Etat                                                               |
| -------------------- | ------------------------------------------------------------------ |
| `mean_pose`        | implementee - reference et baseline plancher (bbox GT, diagnostic) |
| `yolo_pooled`      | **implementee** - GPU CUDA, AMP, FP16 a l'inference          |
| `yolo_per_dataset` | **implementee** - N modeles routes par dataset (ADR-0023)    |
| `detect_then_pose` | **implementee** - detecteur poule + pose sur crop (ADR-0024) |
| `lora`             | a faire                                                            |
| `group_bn`         | a faire                                                            |

Une approche dont la dependance lourde est absente est **ignoree** par le smoke test
(mecanisme `availability()`), jamais en echec.

Chaque run produit : `manifest.json`, `config.yaml` resolu, `predictions/`,
`metrics.parquet`, `logs/` et `figures/` (12 exemples pred vs GT dont les 6 pires cas).

### A quoi sert `mean_pose`

Ce n'est pas un modele candidat : elle predit, pour chaque instance, la **pose moyenne du
train** replacee dans la bbox de verite terrain. Elle remplit trois roles :

1. **baseline plancher** - un modele entraine qui ne la depasse pas nettement a un probleme
   (convergence, labels mal formes, ordre de keypoints errone). C'est un test de sanite, pas
   un concurrent ;
2. **gabarit** - c'est l'implementation de reference du protocole `Approach`, a copier pour
   ecrire une nouvelle approche ;
3. **smoke test** - elle s'execute en quelques secondes, sans GPU ni dependance lourde, ce qui
   valide toute la chaine `train -> predict -> evaluate -> figures` a chaque `pytest -m smoke`.

Elle utilise les bboxes GT (`bbox_source: gt`) : ses chiffres ne sont donc **pas comparables**
a ceux des approches bout-en-bout et ne doivent jamais figurer dans le meme tableau (§9.3).

L'approche `mean_pose` est une **implementation de reference** : elle sert de gabarit, de test
de bout en bout et de baseline plancher. Elle utilise les bboxes GT (`bbox_source: gt`) et n'est
donc **pas comparable** aux approches bout-en-bout (cf. §9.3).

## Protocole fige

Toutes les decisions de protocole sont tranchees (ADR-0006 a 0015, cf. `DECISIONS.md`) :

| Point             | Valeur                                                               |
| ----------------- | -------------------------------------------------------------------- |
| Keypoints         | `insect42_v1` : 42 points, commun aux 4 datasets, union = identite |
| Sigmas OKS        | `difficulty x 0.0025` (10 -> 0.025 ... 40 -> 0.100)                |
| PCK               | `alpha x largeur du thorax`, reference alpha = 0.25                |
| Metrique primaire | `oks_ap` (surchargeable sans reevaluer)                            |
| Mesures           | 27 mesures morphometriques + 9 paires de symetrie                    |
| Folds             | 5 externes, group_id = image_id (une image = un specimen)            |
| HPO               | nichee : recherche sur folds internes, test externe jamais vu        |
| Resolution        | 640x640 pour toutes les approches (garde-fou strict)                 |

**Cout de l'HPO nichee** : `n_folds x n_trials x inner_folds` entrainements par approche,
soit 600 aux valeurs par defaut. A calibrer avant de lancer une approche lourde, puis a figer
a l'identique pour toutes (equite du budget).
