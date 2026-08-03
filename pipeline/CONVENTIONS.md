# CONVENTIONS.md — Règles d'architecture et de génération de code

**Projet :** estimation de pose sur 4 datasets d'insectes (Coleoptera, Diptera, Hymenoptera, Lepidoptera)
**Statut :** contrat normatif. Ce fichier fait autorité sur tout autre document du dépôt.
**Version du contrat :** 2.0 — décisions de protocole tranchées (ADR-0006 à 0015)

---

## 0. Comment utiliser ce fichier

Ce document est destiné à être fourni **en entier et en contexte** à toute IA générative (ou tout contributeur humain) chargée d'écrire du code dans ce dépôt.

Règle zéro : **toute génération de code doit citer, en commentaire d'en-tête du fichier produit, les sections de ce document qu'elle applique.** Si une instruction utilisateur contredit ce fichier, l'IA doit s'arrêter et signaler le conflit au lieu de trancher seule.

Vocabulaire normatif : **DOIT** / **NE DOIT PAS** = contrainte dure, non négociable. **DEVRAIT** = recommandation forte, dérogation possible si documentée dans `DECISIONS.md`. **PEUT** = libre.

---

## 1. Principes directeurs

1. **L'approche est un plugin, pas une branche de `if`.** Ajouter une 6ᵉ approche NE DOIT PAS modifier le code d'entraînement générique, d'évaluation, d'optimisation ou de reporting. Si vous devez modifier `evaluation/` pour ajouter une approche, l'abstraction est mauvaise : signalez-le.
2. **Les contrats de données sont l'API du projet.** Les approches ne communiquent jamais entre elles ni avec l'évaluateur par des objets Python : elles communiquent par des **fichiers au format figé** (§3). Cela permet d'entraîner avec Ultralytics, PyTorch pur, HuggingFace/PEFT, ou un modèle externe, sans que l'évaluateur ne le sache jamais.
3. **Une seule implémentation des métriques.** Aucune métrique NE DOIT être lue depuis les logs d'un framework tiers. Les métriques internes d'Ultralytics, de PyTorch Lightning ou d'un autre entraîneur servent **uniquement au monitoring**, jamais à la comparaison entre approches (§7.1).
4. **Séparation stricte : `fit` ≠ `predict` ≠ `evaluate` ≠ `aggregate`.** Quatre étapes, quatre artefacts, quatre points de reprise. On DOIT pouvoir ré-évaluer une expérience vieille de trois mois sans réentraîner.
5. **Tout est configuration ; rien n'est en dur.** Aucun chemin, hyperparamètre, seuil, taille d'image, nom de classe ou de keypoint NE DOIT apparaître littéralement dans un `.py`. Tout vient d'un YAML ou du `RunContext`.
6. **Reproductibilité par construction.** `run_id` déterministe, seeds explicites, config résolue sérialisée dans le dossier de run, versions de dépendances figées (§6.4).
7. **Coût de l'ignorance.** Toute approche DOIT être exécutable en mode `smoke` (2 epochs, 8 images, 1 fold) pour valider le branchement de bout en bout en < 2 minutes, avant tout entraînement réel.

---

## 2. Arborescence du dépôt

```
insectpose/
├── CONVENTIONS.md              # ce fichier — fait autorité
├── DECISIONS.md                # journal des choix méthodologiques (ADR, append-only)
├── README.md                   # démarrage rapide uniquement, pas de doctrine
├── pyproject.toml              # dépendances figées, config ruff/mypy/pytest
├── Makefile                    # raccourcis: make smoke / make tune / make eval / make report
│
├── configs/                    # composition Hydra — SEULE source de paramètres
│   ├── config.yaml             # config racine + defaults list
│   ├── paths.yaml              # racines de chemins (surchargées par machine)
│   ├── data/                   # coleoptera.yaml diptera.yaml ... pooled.yaml
│   ├── keypoints/              # schémas de keypoints par dataset + union (§3.1)
│   ├── approach/               # yolo_pooled.yaml yolo_per_dataset.yaml
│   │                           # detect_then_pose.yaml lora.yaml group_bn.yaml
│   ├── cv/                     # kfold5.yaml kfold5_grouped.yaml holdout.yaml
│   ├── eval/                   # default.yaml (métriques, seuils, sigmas OKS)
│   ├── tuning/                 # optuna_default.yaml + budgets par approche
│   └── experiment/             # compositions nommées et figées (§5.3)
│
├── data/
│   ├── raw/                    # IMMUABLE, jamais écrit par le code, jamais commité
│   ├── interim/                # sorties d'adaptateurs, régénérable
│   ├── processed/              # format canonique (§3.2), régénérable
│   └── splits/                 # assignations de folds versionnées et hashées (§3.3)
│
├── src/insectpose/
│   ├── contracts.py            # dataclasses/TypedDict des 5 contrats — INTOUCHABLE sans bump
│   ├── registry.py             # registre par nom (approches, métriques, adaptateurs)
│   ├── paths.py                # unique endroit qui construit des chemins
│   ├── context.py              # RunContext (run_id, seed, fold, dossiers, logger)
│   │
│   ├── data/
│   │   ├── schema.py           # validation du format canonique
│   │   ├── adapters/           # raw -> canonique, un module par source
│   │   ├── keypoints.py        # mapping par-dataset <-> espace union
│   │   ├── datamodule.py       # canonique -> batches (superset de champs, §4.3)
│   │   └── splits.py           # génération et lecture des folds
│   │
│   ├── approaches/
│   │   ├── base.py             # Protocol Approach + BaseApproach
│   │   ├── yolo_pooled.py
│   │   ├── yolo_per_dataset.py
│   │   ├── detect_then_pose.py
│   │   ├── lora.py
│   │   └── group_bn.py
│   │
│   ├── models/                 # briques réutilisables (backbones, têtes, adaptateurs LoRA, GroupBN)
│   ├── training/               # boucles génériques, callbacks, early stopping
│   ├── evaluation/
│   │   ├── metrics/            # une métrique = un module enregistré
│   │   ├── matching.py         # appariement pred<->gt (OKS/IoU), partagé
│   │   ├── evaluator.py        # predictions.parquet -> metrics.parquet
│   │   └── aggregate.py        # tous les runs -> results/master.parquet
│   ├── tuning/
│   │   ├── search_spaces.py    # espaces Optuna, un par approche
│   │   └── objective.py        # objectif générique (§6.3)
│   ├── reporting/              # tableaux, figures, tests statistiques
│   ├── cli.py                  # points d'entrée (§5.4)
│   └── utils/                  # seed, io, hashing, geometry, logging
│
├── runs/                       # artefacts d'exécution, non commités (§8)
├── results/                    # agrégats consolidés, parquet + figures
├── reports/                    # livrables (notebooks exportés, PDF, slides)
└── tests/                      # unitaires + contrat + smoke (§10)
```

**Règle d'or de l'arborescence :** un fichier `.py` NE DOIT PAS écrire hors de `runs/<run_id>/`, `data/interim/`, `data/processed/`, `data/splits/` et `results/`. Toute autre écriture est un bug.

---

## 3. Les cinq contrats

Ce sont les cinq formats figés qui rendent le projet modulaire. Chacun porte un champ `schema_version`. **Modifier un contrat DOIT se faire par incrément de version + lecteur rétrocompatible**, jamais par modification en place.

### 3.1 Contrat 0 — Schéma de keypoints (`configs/keypoints/insect42_v1.yaml`)

Les quatre datasets partagent **un seul schéma de 42 points** (ADR-0006). L'espace union est ce schéma lui-même : le mapping est l'identité, et le mécanisme d'union reste en place pour absorber une divergence future sans refonte.

```yaml
schema_version: 1
name: insect42_v1
status: VALIDATED
union_space: insect42_v1
sigma_from_difficulty: {scale: 0.0025}   # sigma = difficulty * scale (ADR-0007)
keypoints:
  - {name: thorax-left,  union: thorax-left,  difficulty: 30, flip: thorax-right}
  - {name: thorax-right, union: thorax-right, difficulty: 30, flip: thorax-left}
skeleton: [[0, 5], [0, 12], ...]         # 51 arêtes anatomiques
```

Règles :

- **L'ordre des 42 points est figé à vie** : il est encodé dans tous les artefacts produits. Ajouter un point = l'ajouter *en fin de liste* et bumper `schema_version`.
- Les tolérances OKS ne sont **pas** écrites en dur : `sigma = difficulty × scale`, où `difficulty` (10 à 40) est la difficulté de positionnement précis fournie par l'expert. Un point difficile à annoter est jugé avec plus d'indulgence, ce qui évite que la métrique soit dominée par le bruit d'annotation. Modifier `scale` change la définition de l'OKS : bumper `eval.version` et rejouer les runs.
- `flip` définit les paires de symétrie ; toute augmentation par miroir sans cette table est interdite. Les points de l'axe médian sont leur propre miroir.
- Un schéma marqué `status: PLACEHOLDER` est refusé quand `strict.require_validated_keypoints` est vrai (valeur par défaut).
- **Mesures morphométriques** (`configs/measurements/insect42_v1.yaml`, ADR-0008) : 27 mesures définies comme des polylignes de keypoints, plus 9 paires gauche/droite. C'est la grandeur réellement consommée en aval, donc une métrique de premier plan — pas une annexe.

### 3.2 Contrat 1 — Annotations canoniques (`data/processed/<dataset>/annotations.parquet`)

Une ligne = une **instance annotée**. Format unique quelle que soit la source d'origine (COCO, CVAT, CSV…).

| colonne                           | type             | description                                                                    |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `schema_version`                | int              | 1                                                                              |
| `dataset`                       | str              | `coleoptera` \| `diptera` \| `hymenoptera` \| `lepidoptera`            |
| `image_id`                      | str              | identifiant**globalement unique** : `<dataset>/<nom_fichier_sans_ext>` |
| `image_path`                    | str              | chemin**relatif à `paths.data_root`**, jamais absolu                  |
| `image_width`, `image_height` | int              | pixels, image d'origine                                                        |
| `instance_id`                   | str              | `<image_id>#<n>`                                                             |
| `group_id`                      | str              | clé anti-fuite : spécimen, planche, session de capture (§6.1)               |
| `bbox_xywh`                     | list[float] (4)  | coordonnées**image d'origine**, pixels absolus                          |
| `kpts_xy`                       | list[float] (2K) | ordre du schéma local, pixels absolus, image d'origine                        |
| `kpts_vis`                      | list[int] (K)    | 0 absent / 1 occulté / 2 visible                                              |
| `area`                          | float            | aire du segment ou de la bbox                                                  |
| `keypoint_schema`               | str              | nom du schéma de §3.1                                                        |
| `split_source`                  | str              | `train` \| `test_officiel` \| `unknown` si un découpage amont existe    |

Règles :

- **Toutes les coordonnées, partout, dans tous les fichiers, sont exprimées dans le repère de l'image d'origine, en pixels absolus.** Aucun format normalisé, aucun `xyxy` relatif, aucune coordonnée dans un repère de crop ne doit jamais quitter un module.
- Les adaptateurs (`data/adapters/`) sont les **seuls** modules autorisés à connaître les formats sources. Un adaptateur ne fait que : lire → convertir → valider (`schema.py`) → écrire. Aucun filtrage, aucune augmentation, aucune décision méthodologique.
- Les instances invalides (keypoints hors image, bbox nulle) sont **conservées** avec un flag `qc_flags`, pas supprimées ; le filtrage est une décision de config, pas d'adaptateur.

### 3.3 Contrat 2 — Splits (`data/splits/<split_id>.parquet` + `.json`)

| colonne      | type                              |
| ------------ | --------------------------------- |
| `split_id` | str, ex.`kfold5_grouped_seed42` |
| `image_id` | str                               |
| `fold`     | int                               |
| `role`     | `train` \| `val` \| `test`  |

Règles :

- Les folds sont **générés une seule fois** et **partagés par toutes les approches**. Une approche NE DOIT JAMAIS créer ses propres splits.
- L'unité de découpage est `group_id`, pas `image_id` (§6.1).
- Le `.json` compagnon contient : seed, stratégie, stratification, comptages par dataset/fold, et un `content_hash` des annotations utilisées. **Si le hash des annotations change, les splits sont invalidés** et le pipeline DOIT refuser de tourner.

### 3.4 Contrat 3 — Prédictions (`runs/<run_id>/predictions/<split>_fold<k>.parquet`)

C'est **le** contrat qui rend les approches interchangeables. Une ligne = une instance prédite.

| colonne                                      | type             | description                                                                                                                                          |
| -------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                           | int              | 1                                                                                                                                                    |
| `run_id`, `fold`, `split`, `dataset` | str/int          |                                                                                                                                                      |
| `image_id`                                 | str              |                                                                                                                                                      |
| `pred_id`                                  | str              | unique                                                                                                                                               |
| `bbox_xywh`                                | list[float] (4)  | repère image d'origine ; obligatoire même pour une approche pose-only (alors = bbox GT ou bbox englobante des kpts, et`bbox_source` le précise) |
| `bbox_score`                               | float            | 1.0 si non applicable                                                                                                                                |
| `kpts_xy`                                  | list[float] (2K) | **repère image d'origine**, schéma local du dataset                                                                                          |
| `kpts_score`                               | list[float] (K)  |                                                                                                                                                      |
| `keypoint_schema`                          | str              | doit correspondre au schéma du dataset de l'image                                                                                                   |
| `bbox_source`                              | str              | `predicted` \| `gt` \| `derived`                                                                                                               |
| `inference_ms`                             | float            | temps par instance, pour la comparaison coût/perf                                                                                                   |

Règles :

- **Aucun seuil de score n'est appliqué à l'écriture.** On écrit toutes les prédictions au-dessus d'un seuil très bas (ex. 0.001) ; le seuillage est une opération d'évaluation, paramétrée en config. Sinon les courbes P/R sont tronquées et les approches deviennent incomparables.
- Toute approche opérant sur **crop** (pipeline détection→pose, §9.3) DOIT conserver la transformation affine crop→image et **rétro-projeter** avant écriture. Écrire des coordonnées dans le repère du crop est une erreur bloquante.
- Un modèle entraîné dans l'espace union DOIT projeter vers le schéma local avant écriture (§3.1).
- Les prédictions sur `test` d'un fold ne DOIVENT contenir que les images de ce fold.

### 3.5 Contrat 4 — Métriques (`runs/<run_id>/metrics.parquet`) et 5 — Manifeste (`runs/<run_id>/manifest.json`)

`metrics.parquet` — format long, jamais large :

| colonne                                       | description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `run_id`, `approach`, `fold`, `split` |                                                        |
| `scope`                                     | `overall` \| `dataset:<nom>` \| `keypoint:<nom>` |
| `metric`                                    | nom canonique, ex.`pck@0.05_bboxdiag`                |
| `value`                                     | float                                                  |
| `n`                                         | taille de l'échantillon sous-jacent                   |

`manifest.json` : `run_id`, timestamp, `approach`, `split_id`, config Hydra **résolue** (pas les surcharges CLI), `content_hash` des données, commit git + état propre/sale du dépôt, versions des dépendances clés, seeds, chemins des artefacts produits, durées, ressources GPU, et `optuna_study`/`trial_number` si applicable. **Un run sans manifeste complet est exclu de l'agrégation.**

---

## 4. Interfaces et registre

### 4.1 Registre

Un décorateur unique, un espace de noms par famille :

```python
@register_approach("lora")            # approches
@register_metric("pck")               # métriques
@register_adapter("coleoptera_cvat")  # adaptateurs de données
```

Le nom enregistré DOIT être identique au nom du fichier YAML de config correspondant. Aucun `import` conditionnel, aucun `if approach == ...` ailleurs que dans le registre.

### 4.2 Protocole `Approach`

Toute approche DOIT implémenter exactement cette interface, ni plus ni moins côté pipeline :

```python
class Approach(Protocol):
    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...
        # entraîne sur data.train, valide sur data.val ; écrit ses poids dans ctx.run_dir/weights/
        # NE DOIT PAS toucher à data.test

    def predict(self, images: ImageSet, ctx: RunContext) -> Path: ...
        # retourne le chemin d'un predictions parquet conforme au Contrat 3

    @classmethod
    def load(cls, run_dir: Path, cfg: DictConfig) -> "Approach": ...
        # reconstruit un modèle prédictif depuis les artefacts, sans réentraînement

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict: ...
        # surcharges de config proposées à Optuna ; aucune logique d'entraînement ici
```

Règles :

- `fit` NE DOIT JAMAIS accéder à `data.test`. Un test unitaire vérifie cette propriété (§10).
- `predict` NE DOIT JAMAIS calculer de métrique.
- Une approche PEUT s'appuyer sur plusieurs sous-modèles (cf. détection+pose) : c'est son affaire interne, invisible du pipeline.
- Une approche « par dataset » (§9.2) reste **une seule** approche : elle encapsule N modèles et route selon `dataset`. Le pipeline ne doit pas voir la différence.

### 4.3 DataModule : superset de champs

Le batch produit par le datamodule DOIT toujours contenir le **superset** des champs utiles à toutes les approches, même si une approche donnée les ignore :

```
images, bboxes, keypoints, visibility, meta{image_id, instance_id, dataset,
dataset_index, group_id, orig_size, transform_matrix}
```

`dataset_index` est indispensable à l'approche BatchNorm par groupe (§9.5) ; `transform_matrix` à la rétro-projection. Les ajouter au coup par coup casse la modularité : ils sont là dès le départ.

---

## 5. Configuration

### 5.1 Outil

Hydra + OmegaConf. Composition par `defaults`, surcharge CLI par `clé=valeur`. Pas de `argparse` manuel, pas de dictionnaires de config codés en Python.

### 5.2 Règles

- Un fichier YAML par entité nommée ; le nom du fichier est l'identifiant.
- Toute clé DOIT avoir une valeur par défaut explicite ; interdiction du `cfg.get("x", 3)` disséminé dans le code.
- Les configs d'approche contiennent **uniquement** ce qui est spécifique à l'approche. Les paramètres communs (taille d'image, batch, epochs, seuils d'éval) vivent dans `config.yaml` et sont surchargeables.
- Interdiction d'interpolations Hydra qui traversent plus d'un niveau (`${a.b.c.d}` illisible) : préférer un champ explicite.
- La config **résolue** est écrite dans `runs/<run_id>/config.yaml` **avant** tout entraînement.

### 5.3 Expériences nommées

Toute exécution destinée au rapport final DOIT passer par un fichier `configs/experiment/*.yaml` figé et commité (ex. `exp_A_yolo_pooled_kfold5.yaml`). Les surcharges CLI ad hoc sont réservées à l'exploration et NE DOIVENT PAS produire de résultats cités dans le rapport.

### 5.4 CLI

Cinq verbes, pas plus :

```
python -m insectpose.cli prepare   data=coleoptera
python -m insectpose.cli split     cv=kfold5_grouped
python -m insectpose.cli train     experiment=exp_A cv.fold=0
python -m insectpose.cli predict   run_id=<...> split=test
python -m insectpose.cli evaluate  run_id=<...>
python -m insectpose.cli tune      experiment=exp_A tuning=optuna_default
python -m insectpose.cli report
```

`train` PEUT enchaîner `predict` + `evaluate` par commodité, mais chacun DOIT rester appelable indépendamment.

---

## 6. Protocole expérimental

### 6.1 Anti-fuite

- Le découpage se fait par `group_id`. Si un spécimen apparaît sur plusieurs images, toutes ses images sont dans le même fold. **Si le `group_id` n'est pas connu pour un dataset, la valeur par défaut est `image_id` et cette limitation DOIT être écrite dans `DECISIONS.md`.**
- Stratification par `dataset` (et par nombre d'instances par image si déséquilibré) obligatoire pour les folds poolés.
- Aucune statistique (moyenne/écart-type de normalisation, taille d'ancres, clustering de keypoints) NE DOIT être calculée sur autre chose que le `train` du fold courant.

### 6.2 Cross-validation

- Schéma par défaut : **K=5 folds groupés stratifiés**, seed fixe, `split_id` unique partagé par toutes les approches. Les mêmes folds pour tout le monde, sinon aucune comparaison n'est valide.
- Les approches « par dataset » (§9.2) utilisent **les mêmes folds**, simplement restreints à leur dataset. Ne jamais régénérer un découpage local.
- Une approche est comparée sur la **moyenne ± écart-type inter-folds**, et les résultats par fold sont conservés pour les tests appariés (§8.3).

### 6.3 Optimisation Optuna

- **Nichée par défaut** (ADR-0012). Pour chaque fold externe, la recherche tourne sur des folds **internes** construits à partir du seul train externe. Ces découpages internes (`<split_id>__outer<k>`) sont générés par `cli split` et versionnés exactement comme les folds externes. Les meilleurs hyperparamètres sont ensuite appliqués au fold externe entier. **Le test externe n'a jamais servi à choisir un hyperparamètre.** Un test automatique vérifie cette propriété.
- Mode dégradé `tune_once` : la recherche n'a lieu que sur les folds internes d'un seul fold externe, et le résultat est réutilisé pour tous les autres. Acceptable si documenté ; le budget de trials doit alors être identique entre approches.
- **Coût** : `n_folds × n_trials × inner_folds` entraînements par approche. À calibrer avant de lancer une approche lourde ; le budget effectif est enregistré dans chaque manifeste.
- L'objectif est **toujours la métrique primaire calculée par l'évaluateur partagé**, lue depuis `metrics.parquet` — jamais une loss de validation ni une métrique interne de framework.
- Un trial = un run complet avec son propre `run_id` et son manifeste ; les trials sont donc évaluables et auditables comme n'importe quel run. Ils n'exportent pas de figures qualitatives (bruit inutile).
- Stockage SQLite sous `runs/optuna/`, une étude par (approche, découpage, objectif, fold externe), reprise activée.
- Pruning `MedianPruner` par défaut ; une approche qui ne peut pas rapporter d'intermédiaire déclare `prunable: false`.
- **Budget équitable** : comparer 100 trials contre 10 invalide la conclusion.

### 6.4 Déterminisme

- Seed unique dans la config, dérivée par `seed_for(run_id, fold, purpose)` pour numpy / torch / python / dataloader workers.
- `torch.use_deterministic_algorithms(True)` en mode `debug` ; en mode `full` on autorise cudnn benchmark mais on l'enregistre dans le manifeste.
- Le non-déterminisme résiduel est absorbé par la répétition : toute conclusion finale DEVRAIT reposer sur ≥ 2 seeds pour l'approche gagnante.

---

## 7. Évaluation

### 7.1 Règle absolue

L'évaluateur prend **uniquement** : un `predictions.parquet` (Contrat 3), les annotations canoniques (Contrat 1), et `configs/eval/*.yaml`. Il ne charge aucun modèle, n'importe aucun module d'approche, et ignore totalement comment les prédictions ont été produites. **Si l'évaluateur doit savoir quelle approche l'a alimenté, le design est cassé.**

### 7.2 Jeu de métriques figé

Identique pour toutes les approches, calculé en `overall`, par `dataset:*`, par `keypoint:*` et par `measurement:*` :

- **Détection** (si `bbox_source == predicted`) : `det_ap@0.5`, `det_ap@[.5:.95]`.
- **Pose** : `oks_ap`, `oks_ap@0.5`, `oks_ar` (sigmas dérivés de la difficulté, ADR-0007) ; `pck@{0.125, 0.25, 0.5}_thorax_width` — un point est correct si son erreur est inférieure à `alpha × largeur du thorax` (ADR-0009), `alpha = 0.25` étant la référence du projet ; `nme_matched_only`, `kpt_coverage`, PCK par keypoint.
- **Échelle de référence** : `pck_normalizer_fallback_rate`. Quand les points de thorax ne sont pas annotés, la normalisation retombe sur la diagonale de bbox — et ce taux de repli est **publié**, jamais silencieux.
- **Mesures morphométriques** (ADR-0008) : `measurement_mape_median`, `measurement_mape_worst`, détail par mesure, et `symmetry_gap_median` / `symmetry_gap_p90` — l'écart gauche/droite des mesures prédites, calculable **sans vérité terrain**, donc utilisable comme contrôle qualité en production.
- **Bout-en-bout** : la métrique primaire pénalise les échecs de détection. Une pipeline qui ne détecte pas l'insecte n'a pas « 0 keypoint évalué », elle a un échec compté.
- **Coût** : latence par instance et p95, nombre de paramètres, VRAM, temps d'entraînement — métriques de premier ordre, pas des annexes.

**Métrique primaire du projet** : `oks_ap` (ADR-0010). C'est la **seule clé d'évaluation librement surchargeable** — elle ne modifie aucun calcul, seulement l'objectif d'Optuna et le classement des approches. Toutes les métriques étant calculées à chaque run, changer d'objectif n'oblige jamais à réévaluer :

```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```

### 7.3 Appariement

L'appariement prédiction↔GT (par OKS ou IoU, greedy par score décroissant) est implémenté **une seule fois** dans `evaluation/matching.py`. Aucune métrique ne réimplémente son propre appariement.

### 7.4 Comparaison des approches sur périmètre commun

Les approches n'ont pas le même périmètre naturel (une approche par dataset ne prédit rien hors de son dataset). Règle : **toute comparaison se fait sur l'union des images de test de tous les folds**, une approche restreinte étant évaluée comme la concaténation de ses N modèles. Un tableau de résultats DOIT indiquer le `n` sous-jacent de chaque cellule (§3.5) ; deux valeurs avec des `n` différents ne sont pas comparables et le rapport DOIT le signaler.

---

## 8. Runs, artefacts et résultats

### 8.1 `run_id`

```
<approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
ex. lora__pooled__kfold5grouped_seed42__fold2__baseline__a3f91c07
```

`hash8` = 8 premiers caractères du hash de la config résolue + du `content_hash` des données. Deux runs identiques ont le même `run_id` : le pipeline DOIT alors sauter le run (idempotence) sauf `force=true`.

### 8.2 Contenu d'un dossier de run

```
runs/<run_id>/
├── manifest.json          # Contrat 5, écrit en dernier -> sa présence signale un run complet
├── config.yaml            # config résolue
├── weights/               # poids, checkpoints, adaptateurs LoRA
├── predictions/           # Contrat 3
├── metrics.parquet        # Contrat 4
├── logs/                  # stdout, courbes, tensorboard/mlflow
└── figures/               # visualisations qualitatives (§8.4)
```

`manifest.json` est écrit **en dernier**. Un dossier sans manifeste = run interrompu, ignoré par l'agrégation, supprimable sans discussion.

### 8.3 Langue des livrables

**Tout ce qui est écrit dans un fichier produit est en anglais** : titres, axes, légendes et annotations de figures, en-têtes et valeurs textuelles de tableaux, champs de manifestes et de rapports JSON, noms de fichiers. Les livrables circulent hors de l'équipe et finissent dans des publications ; une figure en français y est inutilisable.

Le code, les commentaires, les docstrings, les messages de log et la documentation interne (`CONVENTIONS.md`, `DECISIONS.md`, `README.md`) restent en français. La frontière est nette : ce qui sort dans `results/`, `runs/` ou `reports/` est en anglais, le reste non.

Corollaire : un nom de métrique, de scope ou de colonne est un identifiant, jamais une phrase à traduire. `oks_ap`, `dataset:coleoptera`, `measurement_mape_median` sont figés (§3.5) et ne changent pas de langue.

### 8.4 Agrégation et reporting

- `aggregate.py` scanne `runs/*/metrics.parquet` + manifestes → `results/master.parquet`. **C'est le seul chemin vers un tableau de résultats.** Aucune figure, aucun tableau du rapport ne DOIT être produit à partir d'un copier-coller de console.
- Les comparaisons entre approches DEVRAIENT utiliser des tests appariés par fold (Wilcoxon signé ou t apparié) avec correction pour comparaisons multiples, et rapporter des intervalles de confiance plutôt que des rangs bruts.
- `reporting/` produit : tableau principal (approche × dataset × métrique), courbes PCK, scatter coût vs performance, matrice d'erreurs par keypoint, échecs qualitatifs.

### 8.5 Qualitatif obligatoire

Chaque run DOIT exporter au moins 12 images de test annotées pred vs GT, incluant les 6 pires cas selon la métrique primaire. Un modèle n'est jamais validé sur des chiffres seuls.

---

## 9. Contraintes spécifiques par approche

Ces notes fixent les pièges connus de chaque famille. Elles n'ajoutent aucune interface : tout passe par §4.2.

### 9.1 YOLO poolé (une classe « insecte », tous datasets) — IMPLÉMENTÉ

Le schéma de keypoints étant commun aux 4 ordres (ADR-0006), le modèle prédit directement dans le schéma attendu : aucune reprojection union → local n'est nécessaire. Les points absents d'un dataset (ADR-0016) sortent en `vis = 0` dans les labels et sont masqués dans la loss, jamais appris comme des zéros.

Toute la logique risquée est isolée dans `data/yolo_export.py`, testée par aller-retour sans GPU :

- la bbox YOLO est **centrée**, le contrat 1 est en coin haut-gauche ;
- `flip_idx` est obligatoire dans `data.yaml` dès que `fliplr > 0`, sinon le miroir échange gauche et droite sans permuter les labels ;
- les noms de fichiers sont aplatis (`coleoptera__img000`), sinon deux datasets ayant un `img000.png` se recouvrent silencieusement.

Les fichiers YOLO sont un artefact **dérivé**, régénéré par fold sous `runs/<run_id>/yolo_dataset/`, jamais écrit dans `data/processed/`. `conf = 0.001` à l'inférence : le seuillage est une opération d'évaluation.

Matériel (ADR-0019) : `train.device: auto` prend le GPU 0 si CUDA est disponible ; AMP activée par défaut mais désactivée en `mode: debug` ; FP16 à l'inférence sur GPU. VRAM maximale, temps d'entraînement et nombre de paramètres entrent dans le manifeste — ce sont des métriques de coût de premier ordre, et l'agrégation alerte si des runs comparés viennent de matériels différents.

### 9.2 YOLO par dataset — IMPLÉMENTÉ

Une **seule** classe `Approach` encapsulant N modèles, routés par `meta.dataset`. Le pipeline ne voit pas la différence : c'est ce qui garantit que A et B sont évaluées identiquement.

Trois choix de protocole (ADR-0023) :

- chaque modèle repart des **poids de base**, jamais du modèle poolé — A et B restent indépendantes, et la question posée est bien « un spécialiste vaut-il un généraliste ? » ;
- les hyperparamètres sont **partagés** par les N modèles, un trial d'Optuna les entraînant tous. Le budget d'HPO reste ainsi strictement égal à celui de A. Une recherche indépendante par dataset le quadruplerait, et B gagnerait par l'optimisation plutôt que par la méthode ;
- **même nombre d'époques** pour tous les datasets. Conséquence à garder en tête à la lecture : avec 192 images pour Hymenoptera contre 935 pour Coleoptera, le premier voit cinq fois moins de pas d'optimisation. Un écart de performance entre ordres n'est donc pas nécessairement une différence de difficulté.

Les folds sont ceux du découpage partagé, simplement restreints (§6.2). Chaque sous-modèle range ses artefacts sous `weights/<dataset>/`, `yolo_dataset/<dataset>/`, `logs/<dataset>/`, et ses coûts sont préfixés dans le manifeste.

### 9.3 Détection puis pose sur crop — IMPLÉMENTÉ

Deux modèles dans un même run : un détecteur **poolé** (une classe, images entières, labels sans keypoints) puis un modèle YOLO-pose entraîné sur des crops normalisés à la résolution du protocole (ADR-0024).

- Le modèle de pose est entraîné sur des crops issus de bboxes GT **bruitées** (`jitter_scale`, `jitter_shift`), jamais sur des cadrages parfaits : sinon décalage train/test garanti, puisqu'à l'inférence les cadrages viennent d'un détecteur. La validation, elle, utilise des cadrages nets — une métrique de validation bruitée ne servirait à rien.
- Une marge (`crop.padding`) entoure la bbox. Sans elle, tarses et antennes tombent hors du crop et deviennent irrécupérables quelle que soit la qualité du modèle. Les points hors cadre sont marqués `vis = 0` : ni appris comme des zéros, ni comptés comme des erreurs.
- La transformation crop → image est conservée et **toute prédiction est rétro-projetée** vers le repère de l'image d'origine avant écriture (contrat 3).
- L'évaluation bout-en-bout utilise les bboxes **prédites**. Le mode `pose_on_gt_boxes: true` écrit `bbox_source: gt` : diagnostic uniquement, jamais dans le même tableau que les approches bout-en-bout.

### 9.4 LoRA — IMPLÉMENTÉ

Adaptateurs injectés sur les convolutions du dernier segment du cou, backbone et cou gelés, têtes entraînables (ADR-0025). Les index de blocs sont **calculés depuis la structure du modèle**, jamais écrits en dur : changer de taille de réseau (n/s/m/l) décale tout.

Le manifeste enregistre le **nombre de paramètres entraînables** et la liste des modules adaptés. Sans cela, « LoRA rang 8 » ne désigne rien : la même étiquette recouvre des configurations très différentes selon ce qui reste dégelé à côté des adaptateurs.

### 9.5 BatchNorm par groupe — IMPLÉMENTÉ

Toutes les `BatchNorm2d` dupliquées en N copies, statistiques **et** paramètres affines par dataset (ADR-0026). Les poids convolutifs restent partagés : c'est l'hypothèse testée. Chaque branche est initialisée depuis les statistiques du modèle pré-entraîné, jamais aléatoirement.

Les lots sont **mixtes** ; le forward se scinde par groupe puis recompose dans l'ordre. Le groupe vient du nom de fichier exporté (`<dataset>__<stem>`) à l'entraînement, et d'une déclaration explicite à l'inférence. Un dataset inconnu est une **erreur explicite** (ADR-0014), jamais un repli deviné.

### 9.6 Variante à keypoints réduits — IMPLÉMENTÉ

Approche A privée de la supervision sur les pattes et les ailes postérieures (ADR-0027) : ces 16 points passent à `vis = 0` dans les labels d'entraînement et de validation, **jamais dans le test**.

Précaution de lecture obligatoire : la vérité terrain conserve ces points et l'évaluation les compte. Les métriques `overall` de cette variante sont donc **mécaniquement moins bonnes** et ne se comparent pas à celles de A. La comparaison valide porte sur les scopes `keypoint:*` des points conservés :

```
python scripts/compare_models.py --exclude-keypoints leg hindwing
```

qui ajoute une ligne `MEAN (retained)` — le chiffre à comparer.

### 9.7 Patch du modèle Ultralytics

Les approches 9.4 et 9.5 modifient le `nn.Module` construit par Ultralytics, qui ne prévoit ni l'un ni l'autre. Toute cette dépendance aux internes est isolée dans `training/patching.py`, avec deux contraintes vérifiées sur la version installée :

- **les callbacks ne conviennent pas** : `on_pretrain_routine_start` précède la construction du modèle, `on_pretrain_routine_end` suit l'optimiseur et l'EMA. On passe donc un trainer dérivé (`train(trainer=...)`) et le patch s'applique dans `get_model` ;
- **Ultralytics dégèle ce qu'on gèle** : sa boucle de `freeze` remet `requires_grad=True` sur tout paramètre gelé hors `args.freeze`. Le gel est donc réappliqué dans `_build_train_pipeline`, juste avant la construction de l'optimiseur.

Un compte de paramètres entraînables est journalisé et enregistré au manifeste à chaque run. Si une future version d'Ultralytics change cet ordre, ce chiffre le signale immédiatement au lieu de laisser passer un entraînement silencieusement faux.

### 9.6 Approches futures

Toute nouvelle approche (multi-tâches, distillation, pré-entraînement auto-supervisé, ensembles…) s'ajoute par §11 sans dérogation. Si une approche ne rentre pas dans le protocole `Approach`, **on modifie le protocole pour tout le monde, en bumpant sa version** — on ne crée pas de cas particulier.

---

## 10. Tests

Trois niveaux, tous obligatoires avant toute exécution longue :

1. **Tests de contrat** (`tests/contracts/`) : valident qu'un parquet produit respecte le schéma, les bornes de coordonnées, l'unicité des identifiants, la cohérence keypoint_schema ↔ dimension. Exécutés automatiquement à l'écriture de tout artefact en mode `debug`.
2. **Tests unitaires** : rétro-projection crop→image (aller-retour = identité à 1e-6 près), mapping local↔union, appariement, chaque métrique sur un cas calculé à la main, non-fuite (`fit` ne lit pas `test`), reproductibilité (deux runs même seed = mêmes prédictions).
3. **Smoke test** (`make smoke`) : chaque approche enregistrée est exécutée sur un fixture de 8 images et 1 fold, de `train` à `report`. **Une approche qui ne passe pas le smoke test n'est pas considérée comme implémentée.** Le fixture est commité dans `tests/fixtures/`.

CI : ruff + mypy (strict sur `contracts.py`, `registry.py`, `evaluation/`) + pytest + smoke.

---

## 11. Procédure : ajouter une approche

Exactement 6 artefacts, ni plus ni moins. Si vous devez toucher un 7ᵉ fichier existant, c'est un signal de conception à remonter.

1. `src/insectpose/approaches/<nom>.py` — classe décorée `@register_approach("<nom>")`, implémentant §4.2.
2. `configs/approach/<nom>.yaml` — hyperparamètres par défaut, `_target_` vers la classe.
3. `search_space` dans la classe (ou `tuning/search_spaces.py` si volumineux).
4. `tests/approaches/test_<nom>.py` — smoke + tests spécifiques (ex. §9.5).
5. `configs/experiment/exp_<lettre>_<nom>.yaml` — expérience figée pour le rapport.
6. Une entrée dans `DECISIONS.md` : ce que l'approche teste, ses hypothèses, ses limites connues.

---

## 12. Règles de génération pour les IA

À respecter par toute IA produisant du code sur ce dépôt.

**Obligations**

- Déclarer les dépendances lourdes via `availability()` : le smoke test ignore proprement une approche indisponible plutôt que d'échouer.
- Écrire en **anglais** tout texte destiné à un fichier produit (§8.3), en français le code et ses commentaires.
- Lire ce fichier et annoncer, avant d'écrire, quels contrats sont touchés.
- Écrire des signatures typées ; `contracts.py` fait foi pour les types de données.
- Toute fonction publique a une docstring indiquant : entrées, sorties, **effets de bord fichiers** (chemin exact écrit).
- Valider les entrées aux frontières de module (schéma parquet, présence de clés de config) et échouer tôt, bruyamment, avec un message actionnable.
- Produire, avec tout nouveau module, son test correspondant. Code sans test = non livré.
- Toute décision méthodologique non triviale prise en cours de route → ligne dans `DECISIONS.md`, pas un commentaire enterré dans le code.

**Interdictions**

- Pas de chemin en dur, pas de constante magique, pas de seuil littéral dans un `.py`.
- Pas de `try/except` silencieux, pas de `except Exception: pass`, pas de valeur de repli qui masque une donnée manquante.
- Pas de logique d'approche dans `training/`, `evaluation/`, `tuning/`, `reporting/` — aucun `if approach == ...` nulle part.
- Pas de calcul de métrique hors de `evaluation/metrics/`.
- Pas de mutation de `data/raw/`. Jamais.
- Pas de dépendance nouvelle sans justification et ajout à `pyproject.toml`.
- Pas de notebook comme source de vérité : un notebook appelle le package, il ne contient pas de logique.
- Pas de fichier « utils.py » fourre-tout : un module = une responsabilité nommable en une phrase.
- Pas de refactor opportuniste hors du périmètre demandé.

**Quand s'arrêter et demander**
Une IA DOIT interrompre la génération et poser la question si : un contrat devrait changer ; deux approches exigeraient un champ incompatible ; une métrique est ambiguë ; le schéma de keypoints d'un dataset est inconnu ; une décision affecterait la comparabilité entre approches. Inventer une convention pour continuer est la faute la plus coûteuse du projet.

**Gabarit de prompt de tâche recommandé**

```
Contexte : CONVENTIONS.md v1.0 (fourni intégralement).
Tâche : implémenter <X>.
Périmètre : fichiers autorisés à créer/modifier = [...]. Tout le reste est en lecture seule.
Contrats touchés : [aucun | n° ...].
Livrables : code + tests + entrée DECISIONS.md si décision prise.
Critère d'acceptation : `make smoke` passe pour l'approche <X>.
Si une règle de CONVENTIONS.md bloque : arrête-toi et explique.
```

---

## 13. Décisions de protocole (toutes tranchées)

Consignées dans `DECISIONS.md`. Elles sont **fermées** : les modifier invalide les résultats déjà produits.

| #        | Décision                | Valeur retenue                                                                      |
| -------- | ------------------------ | ----------------------------------------------------------------------------------- |
| ADR-0006 | Schéma de keypoints     | `insect42_v1`, 42 points, commun aux 4 datasets, union = identité                |
| ADR-0007 | Sigmas OKS               | `sigma = difficulty × 0.0025` (10→0.025 … 40→0.100)                           |
| ADR-0008 | Mesures morphométriques | 27 mesures + 9 paires symétriques, métriques de premier plan                      |
| ADR-0009 | Normalisation PCK        | `alpha × largeur du thorax`, référence `alpha = 0.25`, taux de repli publié |
| ADR-0010 | Métrique primaire       | `oks_ap`, seule clé d'évaluation librement surchargeable                        |
| ADR-0011 | Groupement anti-fuite    | une image = un specimen,`group_id = image_id`                                     |
| ADR-0012 | HPO                      | nichée : recherche sur folds internes, test externe jamais vu                      |
| ADR-0013 | Résolution d'entrée    | 640×640 pour toutes les approches, garde-fou strict                                |
| ADR-0014 | Dataset à l'inférence  | toujours déclaré ; un dataset inconnu est une erreur explicite                    |
| ADR-0015 | Suivi d'expériences     | manifestes +`master.parquet` uniquement                                           |

Restent ouverts sans bloquer : le budget d'HPO réellement soutenable (OPEN-09) et le traitement des keypoints systématiquement absents dans un dataset donné (OPEN-10).

---

*Fin du contrat. Toute évolution passe par un incrément de version de ce fichier et une entrée dans `DECISIONS.md`*

# CONVENTIONS.md — Règles d'architecture et de génération de code

**Projet :** estimation de pose sur 4 datasets d'insectes (Coleoptera, Diptera, Hymenoptera, Lepidoptera)
**Statut :** contrat normatif. Ce fichier fait autorité sur tout autre document du dépôt.
**Version du contrat :** 2.0 — décisions de protocole tranchées (ADR-0006 à 0015)

---

## 0. Comment utiliser ce fichier

Ce document est destiné à être fourni **en entier et en contexte** à toute IA générative (ou tout contributeur humain) chargée d'écrire du code dans ce dépôt.

Règle zéro : **toute génération de code doit citer, en commentaire d'en-tête du fichier produit, les sections de ce document qu'elle applique.** Si une instruction utilisateur contredit ce fichier, l'IA doit s'arrêter et signaler le conflit au lieu de trancher seule.

Vocabulaire normatif : **DOIT** / **NE DOIT PAS** = contrainte dure, non négociable. **DEVRAIT** = recommandation forte, dérogation possible si documentée dans `DECISIONS.md`. **PEUT** = libre.

---

## 1. Principes directeurs

1. **L'approche est un plugin, pas une branche de `if`.** Ajouter une 6ᵉ approche NE DOIT PAS modifier le code d'entraînement générique, d'évaluation, d'optimisation ou de reporting. Si vous devez modifier `evaluation/` pour ajouter une approche, l'abstraction est mauvaise : signalez-le.
2. **Les contrats de données sont l'API du projet.** Les approches ne communiquent jamais entre elles ni avec l'évaluateur par des objets Python : elles communiquent par des **fichiers au format figé** (§3). Cela permet d'entraîner avec Ultralytics, PyTorch pur, HuggingFace/PEFT, ou un modèle externe, sans que l'évaluateur ne le sache jamais.
3. **Une seule implémentation des métriques.** Aucune métrique NE DOIT être lue depuis les logs d'un framework tiers. Les métriques internes d'Ultralytics, de PyTorch Lightning ou d'un autre entraîneur servent **uniquement au monitoring**, jamais à la comparaison entre approches (§7.1).
4. **Séparation stricte : `fit` ≠ `predict` ≠ `evaluate` ≠ `aggregate`.** Quatre étapes, quatre artefacts, quatre points de reprise. On DOIT pouvoir ré-évaluer une expérience vieille de trois mois sans réentraîner.
5. **Tout est configuration ; rien n'est en dur.** Aucun chemin, hyperparamètre, seuil, taille d'image, nom de classe ou de keypoint NE DOIT apparaître littéralement dans un `.py`. Tout vient d'un YAML ou du `RunContext`.
6. **Reproductibilité par construction.** `run_id` déterministe, seeds explicites, config résolue sérialisée dans le dossier de run, versions de dépendances figées (§6.4).
7. **Coût de l'ignorance.** Toute approche DOIT être exécutable en mode `smoke` (2 epochs, 8 images, 1 fold) pour valider le branchement de bout en bout en < 2 minutes, avant tout entraînement réel.

---

## 2. Arborescence du dépôt

```
insectpose/
├── CONVENTIONS.md              # ce fichier — fait autorité
├── DECISIONS.md                # journal des choix méthodologiques (ADR, append-only)
├── README.md                   # démarrage rapide uniquement, pas de doctrine
├── pyproject.toml              # dépendances figées, config ruff/mypy/pytest
├── Makefile                    # raccourcis: make smoke / make tune / make eval / make report
│
├── configs/                    # composition Hydra — SEULE source de paramètres
│   ├── config.yaml             # config racine + defaults list
│   ├── paths.yaml              # racines de chemins (surchargées par machine)
│   ├── data/                   # coleoptera.yaml diptera.yaml ... pooled.yaml
│   ├── keypoints/              # schémas de keypoints par dataset + union (§3.1)
│   ├── approach/               # yolo_pooled.yaml yolo_per_dataset.yaml
│   │                           # detect_then_pose.yaml lora.yaml group_bn.yaml
│   ├── cv/                     # kfold5.yaml kfold5_grouped.yaml holdout.yaml
│   ├── eval/                   # default.yaml (métriques, seuils, sigmas OKS)
│   ├── tuning/                 # optuna_default.yaml + budgets par approche
│   └── experiment/             # compositions nommées et figées (§5.3)
│
├── data/
│   ├── raw/                    # IMMUABLE, jamais écrit par le code, jamais commité
│   ├── interim/                # sorties d'adaptateurs, régénérable
│   ├── processed/              # format canonique (§3.2), régénérable
│   └── splits/                 # assignations de folds versionnées et hashées (§3.3)
│
├── src/insectpose/
│   ├── contracts.py            # dataclasses/TypedDict des 5 contrats — INTOUCHABLE sans bump
│   ├── registry.py             # registre par nom (approches, métriques, adaptateurs)
│   ├── paths.py                # unique endroit qui construit des chemins
│   ├── context.py              # RunContext (run_id, seed, fold, dossiers, logger)
│   │
│   ├── data/
│   │   ├── schema.py           # validation du format canonique
│   │   ├── adapters/           # raw -> canonique, un module par source
│   │   ├── keypoints.py        # mapping par-dataset <-> espace union
│   │   ├── datamodule.py       # canonique -> batches (superset de champs, §4.3)
│   │   └── splits.py           # génération et lecture des folds
│   │
│   ├── approaches/
│   │   ├── base.py             # Protocol Approach + BaseApproach
│   │   ├── yolo_pooled.py
│   │   ├── yolo_per_dataset.py
│   │   ├── detect_then_pose.py
│   │   ├── lora.py
│   │   └── group_bn.py
│   │
│   ├── models/                 # briques réutilisables (backbones, têtes, adaptateurs LoRA, GroupBN)
│   ├── training/               # boucles génériques, callbacks, early stopping
│   ├── evaluation/
│   │   ├── metrics/            # une métrique = un module enregistré
│   │   ├── matching.py         # appariement pred<->gt (OKS/IoU), partagé
│   │   ├── evaluator.py        # predictions.parquet -> metrics.parquet
│   │   └── aggregate.py        # tous les runs -> results/master.parquet
│   ├── tuning/
│   │   ├── search_spaces.py    # espaces Optuna, un par approche
│   │   └── objective.py        # objectif générique (§6.3)
│   ├── reporting/              # tableaux, figures, tests statistiques
│   ├── cli.py                  # points d'entrée (§5.4)
│   └── utils/                  # seed, io, hashing, geometry, logging
│
├── runs/                       # artefacts d'exécution, non commités (§8)
├── results/                    # agrégats consolidés, parquet + figures
├── reports/                    # livrables (notebooks exportés, PDF, slides)
└── tests/                      # unitaires + contrat + smoke (§10)
```

**Règle d'or de l'arborescence :** un fichier `.py` NE DOIT PAS écrire hors de `runs/<run_id>/`, `data/interim/`, `data/processed/`, `data/splits/` et `results/`. Toute autre écriture est un bug.

---

## 3. Les cinq contrats

Ce sont les cinq formats figés qui rendent le projet modulaire. Chacun porte un champ `schema_version`. **Modifier un contrat DOIT se faire par incrément de version + lecteur rétrocompatible**, jamais par modification en place.

### 3.1 Contrat 0 — Schéma de keypoints (`configs/keypoints/insect42_v1.yaml`)

Les quatre datasets partagent **un seul schéma de 42 points** (ADR-0006). L'espace union est ce schéma lui-même : le mapping est l'identité, et le mécanisme d'union reste en place pour absorber une divergence future sans refonte.

```yaml
schema_version: 1
name: insect42_v1
status: VALIDATED
union_space: insect42_v1
sigma_from_difficulty: {scale: 0.0025}   # sigma = difficulty * scale (ADR-0007)
keypoints:
  - {name: thorax-left,  union: thorax-left,  difficulty: 30, flip: thorax-right}
  - {name: thorax-right, union: thorax-right, difficulty: 30, flip: thorax-left}
skeleton: [[0, 5], [0, 12], ...]         # 51 arêtes anatomiques
```

Règles :

- **L'ordre des 42 points est figé à vie** : il est encodé dans tous les artefacts produits. Ajouter un point = l'ajouter *en fin de liste* et bumper `schema_version`.
- Les tolérances OKS ne sont **pas** écrites en dur : `sigma = difficulty × scale`, où `difficulty` (10 à 40) est la difficulté de positionnement précis fournie par l'expert. Un point difficile à annoter est jugé avec plus d'indulgence, ce qui évite que la métrique soit dominée par le bruit d'annotation. Modifier `scale` change la définition de l'OKS : bumper `eval.version` et rejouer les runs.
- `flip` définit les paires de symétrie ; toute augmentation par miroir sans cette table est interdite. Les points de l'axe médian sont leur propre miroir.
- Un schéma marqué `status: PLACEHOLDER` est refusé quand `strict.require_validated_keypoints` est vrai (valeur par défaut).
- **Mesures morphométriques** (`configs/measurements/insect42_v1.yaml`, ADR-0008) : 27 mesures définies comme des polylignes de keypoints, plus 9 paires gauche/droite. C'est la grandeur réellement consommée en aval, donc une métrique de premier plan — pas une annexe.

### 3.2 Contrat 1 — Annotations canoniques (`data/processed/<dataset>/annotations.parquet`)

Une ligne = une **instance annotée**. Format unique quelle que soit la source d'origine (COCO, CVAT, CSV…).

| colonne                           | type             | description                                                                    |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `schema_version`                | int              | 1                                                                              |
| `dataset`                       | str              | `coleoptera` \| `diptera` \| `hymenoptera` \| `lepidoptera`            |
| `image_id`                      | str              | identifiant**globalement unique** : `<dataset>/<nom_fichier_sans_ext>` |
| `image_path`                    | str              | chemin**relatif à `paths.data_root`**, jamais absolu                  |
| `image_width`, `image_height` | int              | pixels, image d'origine                                                        |
| `instance_id`                   | str              | `<image_id>#<n>`                                                             |
| `group_id`                      | str              | clé anti-fuite : spécimen, planche, session de capture (§6.1)               |
| `bbox_xywh`                     | list[float] (4)  | coordonnées**image d'origine**, pixels absolus                          |
| `kpts_xy`                       | list[float] (2K) | ordre du schéma local, pixels absolus, image d'origine                        |
| `kpts_vis`                      | list[int] (K)    | 0 absent / 1 occulté / 2 visible                                              |
| `area`                          | float            | aire du segment ou de la bbox                                                  |
| `keypoint_schema`               | str              | nom du schéma de §3.1                                                        |
| `split_source`                  | str              | `train` \| `test_officiel` \| `unknown` si un découpage amont existe    |

Règles :

- **Toutes les coordonnées, partout, dans tous les fichiers, sont exprimées dans le repère de l'image d'origine, en pixels absolus.** Aucun format normalisé, aucun `xyxy` relatif, aucune coordonnée dans un repère de crop ne doit jamais quitter un module.
- Les adaptateurs (`data/adapters/`) sont les **seuls** modules autorisés à connaître les formats sources. Un adaptateur ne fait que : lire → convertir → valider (`schema.py`) → écrire. Aucun filtrage, aucune augmentation, aucune décision méthodologique.
- Les instances invalides (keypoints hors image, bbox nulle) sont **conservées** avec un flag `qc_flags`, pas supprimées ; le filtrage est une décision de config, pas d'adaptateur.

### 3.3 Contrat 2 — Splits (`data/splits/<split_id>.parquet` + `.json`)

| colonne      | type                              |
| ------------ | --------------------------------- |
| `split_id` | str, ex.`kfold5_grouped_seed42` |
| `image_id` | str                               |
| `fold`     | int                               |
| `role`     | `train` \| `val` \| `test`  |

Règles :

- Les folds sont **générés une seule fois** et **partagés par toutes les approches**. Une approche NE DOIT JAMAIS créer ses propres splits.
- L'unité de découpage est `group_id`, pas `image_id` (§6.1).
- Le `.json` compagnon contient : seed, stratégie, stratification, comptages par dataset/fold, et un `content_hash` des annotations utilisées. **Si le hash des annotations change, les splits sont invalidés** et le pipeline DOIT refuser de tourner.

### 3.4 Contrat 3 — Prédictions (`runs/<run_id>/predictions/<split>_fold<k>.parquet`)

C'est **le** contrat qui rend les approches interchangeables. Une ligne = une instance prédite.

| colonne                                      | type             | description                                                                                                                                          |
| -------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                           | int              | 1                                                                                                                                                    |
| `run_id`, `fold`, `split`, `dataset` | str/int          |                                                                                                                                                      |
| `image_id`                                 | str              |                                                                                                                                                      |
| `pred_id`                                  | str              | unique                                                                                                                                               |
| `bbox_xywh`                                | list[float] (4)  | repère image d'origine ; obligatoire même pour une approche pose-only (alors = bbox GT ou bbox englobante des kpts, et`bbox_source` le précise) |
| `bbox_score`                               | float            | 1.0 si non applicable                                                                                                                                |
| `kpts_xy`                                  | list[float] (2K) | **repère image d'origine**, schéma local du dataset                                                                                          |
| `kpts_score`                               | list[float] (K)  |                                                                                                                                                      |
| `keypoint_schema`                          | str              | doit correspondre au schéma du dataset de l'image                                                                                                   |
| `bbox_source`                              | str              | `predicted` \| `gt` \| `derived`                                                                                                               |
| `inference_ms`                             | float            | temps par instance, pour la comparaison coût/perf                                                                                                   |

Règles :

- **Aucun seuil de score n'est appliqué à l'écriture.** On écrit toutes les prédictions au-dessus d'un seuil très bas (ex. 0.001) ; le seuillage est une opération d'évaluation, paramétrée en config. Sinon les courbes P/R sont tronquées et les approches deviennent incomparables.
- Toute approche opérant sur **crop** (pipeline détection→pose, §9.3) DOIT conserver la transformation affine crop→image et **rétro-projeter** avant écriture. Écrire des coordonnées dans le repère du crop est une erreur bloquante.
- Un modèle entraîné dans l'espace union DOIT projeter vers le schéma local avant écriture (§3.1).
- Les prédictions sur `test` d'un fold ne DOIVENT contenir que les images de ce fold.

### 3.5 Contrat 4 — Métriques (`runs/<run_id>/metrics.parquet`) et 5 — Manifeste (`runs/<run_id>/manifest.json`)

`metrics.parquet` — format long, jamais large :

| colonne                                       | description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `run_id`, `approach`, `fold`, `split` |                                                        |
| `scope`                                     | `overall` \| `dataset:<nom>` \| `keypoint:<nom>` |
| `metric`                                    | nom canonique, ex.`pck@0.05_bboxdiag`                |
| `value`                                     | float                                                  |
| `n`                                         | taille de l'échantillon sous-jacent                   |

`manifest.json` : `run_id`, timestamp, `approach`, `split_id`, config Hydra **résolue** (pas les surcharges CLI), `content_hash` des données, commit git + état propre/sale du dépôt, versions des dépendances clés, seeds, chemins des artefacts produits, durées, ressources GPU, et `optuna_study`/`trial_number` si applicable. **Un run sans manifeste complet est exclu de l'agrégation.**

---

## 4. Interfaces et registre

### 4.1 Registre

Un décorateur unique, un espace de noms par famille :

```python
@register_approach("lora")            # approches
@register_metric("pck")               # métriques
@register_adapter("coleoptera_cvat")  # adaptateurs de données
```

Le nom enregistré DOIT être identique au nom du fichier YAML de config correspondant. Aucun `import` conditionnel, aucun `if approach == ...` ailleurs que dans le registre.

### 4.2 Protocole `Approach`

Toute approche DOIT implémenter exactement cette interface, ni plus ni moins côté pipeline :

```python
class Approach(Protocol):
    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...
        # entraîne sur data.train, valide sur data.val ; écrit ses poids dans ctx.run_dir/weights/
        # NE DOIT PAS toucher à data.test

    def predict(self, images: ImageSet, ctx: RunContext) -> Path: ...
        # retourne le chemin d'un predictions parquet conforme au Contrat 3

    @classmethod
    def load(cls, run_dir: Path, cfg: DictConfig) -> "Approach": ...
        # reconstruit un modèle prédictif depuis les artefacts, sans réentraînement

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict: ...
        # surcharges de config proposées à Optuna ; aucune logique d'entraînement ici
```

Règles :

- `fit` NE DOIT JAMAIS accéder à `data.test`. Un test unitaire vérifie cette propriété (§10).
- `predict` NE DOIT JAMAIS calculer de métrique.
- Une approche PEUT s'appuyer sur plusieurs sous-modèles (cf. détection+pose) : c'est son affaire interne, invisible du pipeline.
- Une approche « par dataset » (§9.2) reste **une seule** approche : elle encapsule N modèles et route selon `dataset`. Le pipeline ne doit pas voir la différence.

### 4.3 DataModule : superset de champs

Le batch produit par le datamodule DOIT toujours contenir le **superset** des champs utiles à toutes les approches, même si une approche donnée les ignore :

```
images, bboxes, keypoints, visibility, meta{image_id, instance_id, dataset,
dataset_index, group_id, orig_size, transform_matrix}
```

`dataset_index` est indispensable à l'approche BatchNorm par groupe (§9.5) ; `transform_matrix` à la rétro-projection. Les ajouter au coup par coup casse la modularité : ils sont là dès le départ.

---

## 5. Configuration

### 5.1 Outil

Hydra + OmegaConf. Composition par `defaults`, surcharge CLI par `clé=valeur`. Pas de `argparse` manuel, pas de dictionnaires de config codés en Python.

### 5.2 Règles

- Un fichier YAML par entité nommée ; le nom du fichier est l'identifiant.
- Toute clé DOIT avoir une valeur par défaut explicite ; interdiction du `cfg.get("x", 3)` disséminé dans le code.
- Les configs d'approche contiennent **uniquement** ce qui est spécifique à l'approche. Les paramètres communs (taille d'image, batch, epochs, seuils d'éval) vivent dans `config.yaml` et sont surchargeables.
- Interdiction d'interpolations Hydra qui traversent plus d'un niveau (`${a.b.c.d}` illisible) : préférer un champ explicite.
- La config **résolue** est écrite dans `runs/<run_id>/config.yaml` **avant** tout entraînement.

### 5.3 Expériences nommées

Toute exécution destinée au rapport final DOIT passer par un fichier `configs/experiment/*.yaml` figé et commité (ex. `exp_A_yolo_pooled_kfold5.yaml`). Les surcharges CLI ad hoc sont réservées à l'exploration et NE DOIVENT PAS produire de résultats cités dans le rapport.

### 5.4 CLI

Cinq verbes, pas plus :

```
python -m insectpose.cli prepare   data=coleoptera
python -m insectpose.cli split     cv=kfold5_grouped
python -m insectpose.cli train     experiment=exp_A cv.fold=0
python -m insectpose.cli predict   run_id=<...> split=test
python -m insectpose.cli evaluate  run_id=<...>
python -m insectpose.cli tune      experiment=exp_A tuning=optuna_default
python -m insectpose.cli report
```

`train` PEUT enchaîner `predict` + `evaluate` par commodité, mais chacun DOIT rester appelable indépendamment.

---

## 6. Protocole expérimental

### 6.1 Anti-fuite

- Le découpage se fait par `group_id`. Si un spécimen apparaît sur plusieurs images, toutes ses images sont dans le même fold. **Si le `group_id` n'est pas connu pour un dataset, la valeur par défaut est `image_id` et cette limitation DOIT être écrite dans `DECISIONS.md`.**
- Stratification par `dataset` (et par nombre d'instances par image si déséquilibré) obligatoire pour les folds poolés.
- Aucune statistique (moyenne/écart-type de normalisation, taille d'ancres, clustering de keypoints) NE DOIT être calculée sur autre chose que le `train` du fold courant.

### 6.2 Cross-validation

- Schéma par défaut : **K=5 folds groupés stratifiés**, seed fixe, `split_id` unique partagé par toutes les approches. Les mêmes folds pour tout le monde, sinon aucune comparaison n'est valide.
- Les approches « par dataset » (§9.2) utilisent **les mêmes folds**, simplement restreints à leur dataset. Ne jamais régénérer un découpage local.
- Une approche est comparée sur la **moyenne ± écart-type inter-folds**, et les résultats par fold sont conservés pour les tests appariés (§8.3).

### 6.3 Optimisation Optuna

- **Nichée par défaut** (ADR-0012). Pour chaque fold externe, la recherche tourne sur des folds **internes** construits à partir du seul train externe. Ces découpages internes (`<split_id>__outer<k>`) sont générés par `cli split` et versionnés exactement comme les folds externes. Les meilleurs hyperparamètres sont ensuite appliqués au fold externe entier. **Le test externe n'a jamais servi à choisir un hyperparamètre.** Un test automatique vérifie cette propriété.
- Mode dégradé `tune_once` : la recherche n'a lieu que sur les folds internes d'un seul fold externe, et le résultat est réutilisé pour tous les autres. Acceptable si documenté ; le budget de trials doit alors être identique entre approches.
- **Coût** : `n_folds × n_trials × inner_folds` entraînements par approche. À calibrer avant de lancer une approche lourde ; le budget effectif est enregistré dans chaque manifeste.
- L'objectif est **toujours la métrique primaire calculée par l'évaluateur partagé**, lue depuis `metrics.parquet` — jamais une loss de validation ni une métrique interne de framework.
- Un trial = un run complet avec son propre `run_id` et son manifeste ; les trials sont donc évaluables et auditables comme n'importe quel run. Ils n'exportent pas de figures qualitatives (bruit inutile).
- Stockage SQLite sous `runs/optuna/`, une étude par (approche, découpage, objectif, fold externe), reprise activée.
- Pruning `MedianPruner` par défaut ; une approche qui ne peut pas rapporter d'intermédiaire déclare `prunable: false`.
- **Budget équitable** : comparer 100 trials contre 10 invalide la conclusion.

### 6.4 Déterminisme

- Seed unique dans la config, dérivée par `seed_for(run_id, fold, purpose)` pour numpy / torch / python / dataloader workers.
- `torch.use_deterministic_algorithms(True)` en mode `debug` ; en mode `full` on autorise cudnn benchmark mais on l'enregistre dans le manifeste.
- Le non-déterminisme résiduel est absorbé par la répétition : toute conclusion finale DEVRAIT reposer sur ≥ 2 seeds pour l'approche gagnante.

---

## 7. Évaluation

### 7.1 Règle absolue

L'évaluateur prend **uniquement** : un `predictions.parquet` (Contrat 3), les annotations canoniques (Contrat 1), et `configs/eval/*.yaml`. Il ne charge aucun modèle, n'importe aucun module d'approche, et ignore totalement comment les prédictions ont été produites. **Si l'évaluateur doit savoir quelle approche l'a alimenté, le design est cassé.**

### 7.2 Jeu de métriques figé

Identique pour toutes les approches, calculé en `overall`, par `dataset:*`, par `keypoint:*` et par `measurement:*` :

- **Détection** (si `bbox_source == predicted`) : `det_ap@0.5`, `det_ap@[.5:.95]`.
- **Pose** : `oks_ap`, `oks_ap@0.5`, `oks_ar` (sigmas dérivés de la difficulté, ADR-0007) ; `pck@{0.125, 0.25, 0.5}_thorax_width` — un point est correct si son erreur est inférieure à `alpha × largeur du thorax` (ADR-0009), `alpha = 0.25` étant la référence du projet ; `nme_matched_only`, `kpt_coverage`, PCK par keypoint.
- **Échelle de référence** : `pck_normalizer_fallback_rate`. Quand les points de thorax ne sont pas annotés, la normalisation retombe sur la diagonale de bbox — et ce taux de repli est **publié**, jamais silencieux.
- **Mesures morphométriques** (ADR-0008) : `measurement_mape_median`, `measurement_mape_worst`, détail par mesure, et `symmetry_gap_median` / `symmetry_gap_p90` — l'écart gauche/droite des mesures prédites, calculable **sans vérité terrain**, donc utilisable comme contrôle qualité en production.
- **Bout-en-bout** : la métrique primaire pénalise les échecs de détection. Une pipeline qui ne détecte pas l'insecte n'a pas « 0 keypoint évalué », elle a un échec compté.
- **Coût** : latence par instance et p95, nombre de paramètres, VRAM, temps d'entraînement — métriques de premier ordre, pas des annexes.

**Métrique primaire du projet** : `oks_ap` (ADR-0010). C'est la **seule clé d'évaluation librement surchargeable** — elle ne modifie aucun calcul, seulement l'objectif d'Optuna et le classement des approches. Toutes les métriques étant calculées à chaque run, changer d'objectif n'oblige jamais à réévaluer :

```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```

### 7.3 Appariement

L'appariement prédiction↔GT (par OKS ou IoU, greedy par score décroissant) est implémenté **une seule fois** dans `evaluation/matching.py`. Aucune métrique ne réimplémente son propre appariement.

### 7.4 Comparaison des approches sur périmètre commun

Les approches n'ont pas le même périmètre naturel (une approche par dataset ne prédit rien hors de son dataset). Règle : **toute comparaison se fait sur l'union des images de test de tous les folds**, une approche restreinte étant évaluée comme la concaténation de ses N modèles. Un tableau de résultats DOIT indiquer le `n` sous-jacent de chaque cellule (§3.5) ; deux valeurs avec des `n` différents ne sont pas comparables et le rapport DOIT le signaler.

---

## 8. Runs, artefacts et résultats

### 8.1 `run_id`

```
<approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
ex. lora__pooled__kfold5grouped_seed42__fold2__baseline__a3f91c07
```

`hash8` = 8 premiers caractères du hash de la config résolue + du `content_hash` des données. Deux runs identiques ont le même `run_id` : le pipeline DOIT alors sauter le run (idempotence) sauf `force=true`.

### 8.2 Contenu d'un dossier de run

```
runs/<run_id>/
├── manifest.json          # Contrat 5, écrit en dernier -> sa présence signale un run complet
├── config.yaml            # config résolue
├── weights/               # poids, checkpoints, adaptateurs LoRA
├── predictions/           # Contrat 3
├── metrics.parquet        # Contrat 4
├── logs/                  # stdout, courbes, tensorboard/mlflow
└── figures/               # visualisations qualitatives (§8.4)
```

`manifest.json` est écrit **en dernier**. Un dossier sans manifeste = run interrompu, ignoré par l'agrégation, supprimable sans discussion.

### 8.3 Langue des livrables

**Tout ce qui est écrit dans un fichier produit est en anglais** : titres, axes, légendes et annotations de figures, en-têtes et valeurs textuelles de tableaux, champs de manifestes et de rapports JSON, noms de fichiers. Les livrables circulent hors de l'équipe et finissent dans des publications ; une figure en français y est inutilisable.

Le code, les commentaires, les docstrings, les messages de log et la documentation interne (`CONVENTIONS.md`, `DECISIONS.md`, `README.md`) restent en français. La frontière est nette : ce qui sort dans `results/`, `runs/` ou `reports/` est en anglais, le reste non.

Corollaire : un nom de métrique, de scope ou de colonne est un identifiant, jamais une phrase à traduire. `oks_ap`, `dataset:coleoptera`, `measurement_mape_median` sont figés (§3.5) et ne changent pas de langue.

### 8.4 Agrégation et reporting

- `aggregate.py` scanne `runs/*/metrics.parquet` + manifestes → `results/master.parquet`. **C'est le seul chemin vers un tableau de résultats.** Aucune figure, aucun tableau du rapport ne DOIT être produit à partir d'un copier-coller de console.
- Les comparaisons entre approches DEVRAIENT utiliser des tests appariés par fold (Wilcoxon signé ou t apparié) avec correction pour comparaisons multiples, et rapporter des intervalles de confiance plutôt que des rangs bruts.
- `reporting/` produit : tableau principal (approche × dataset × métrique), courbes PCK, scatter coût vs performance, matrice d'erreurs par keypoint, échecs qualitatifs.

### 8.5 Qualitatif obligatoire

Chaque run DOIT exporter au moins 12 images de test annotées pred vs GT, incluant les 6 pires cas selon la métrique primaire. Un modèle n'est jamais validé sur des chiffres seuls.

---

## 9. Contraintes spécifiques par approche

Ces notes fixent les pièges connus de chaque famille. Elles n'ajoutent aucune interface : tout passe par §4.2.

### 9.1 YOLO poolé (une classe « insecte », tous datasets) — IMPLÉMENTÉ

Le schéma de keypoints étant commun aux 4 ordres (ADR-0006), le modèle prédit directement dans le schéma attendu : aucune reprojection union → local n'est nécessaire. Les points absents d'un dataset (ADR-0016) sortent en `vis = 0` dans les labels et sont masqués dans la loss, jamais appris comme des zéros.

Toute la logique risquée est isolée dans `data/yolo_export.py`, testée par aller-retour sans GPU :

- la bbox YOLO est **centrée**, le contrat 1 est en coin haut-gauche ;
- `flip_idx` est obligatoire dans `data.yaml` dès que `fliplr > 0`, sinon le miroir échange gauche et droite sans permuter les labels ;
- les noms de fichiers sont aplatis (`coleoptera__img000`), sinon deux datasets ayant un `img000.png` se recouvrent silencieusement.

Les fichiers YOLO sont un artefact **dérivé**, régénéré par fold sous `runs/<run_id>/yolo_dataset/`, jamais écrit dans `data/processed/`. `conf = 0.001` à l'inférence : le seuillage est une opération d'évaluation.

Matériel (ADR-0019) : `train.device: auto` prend le GPU 0 si CUDA est disponible ; AMP activée par défaut mais désactivée en `mode: debug` ; FP16 à l'inférence sur GPU. VRAM maximale, temps d'entraînement et nombre de paramètres entrent dans le manifeste — ce sont des métriques de coût de premier ordre, et l'agrégation alerte si des runs comparés viennent de matériels différents.

### 9.2 YOLO par dataset — IMPLÉMENTÉ

Une **seule** classe `Approach` encapsulant N modèles, routés par `meta.dataset`. Le pipeline ne voit pas la différence : c'est ce qui garantit que A et B sont évaluées identiquement.

Trois choix de protocole (ADR-0023) :

- chaque modèle repart des **poids de base**, jamais du modèle poolé — A et B restent indépendantes, et la question posée est bien « un spécialiste vaut-il un généraliste ? » ;
- les hyperparamètres sont **partagés** par les N modèles, un trial d'Optuna les entraînant tous. Le budget d'HPO reste ainsi strictement égal à celui de A. Une recherche indépendante par dataset le quadruplerait, et B gagnerait par l'optimisation plutôt que par la méthode ;
- **même nombre d'époques** pour tous les datasets. Conséquence à garder en tête à la lecture : avec 192 images pour Hymenoptera contre 935 pour Coleoptera, le premier voit cinq fois moins de pas d'optimisation. Un écart de performance entre ordres n'est donc pas nécessairement une différence de difficulté.

Les folds sont ceux du découpage partagé, simplement restreints (§6.2). Chaque sous-modèle range ses artefacts sous `weights/<dataset>/`, `yolo_dataset/<dataset>/`, `logs/<dataset>/`, et ses coûts sont préfixés dans le manifeste.

### 9.3 Détection puis pose sur crop — IMPLÉMENTÉ

Deux modèles dans un même run : un détecteur **poolé** (une classe, images entières, labels sans keypoints) puis un modèle YOLO-pose entraîné sur des crops normalisés à la résolution du protocole (ADR-0024).

- Le modèle de pose est entraîné sur des crops issus de bboxes GT **bruitées** (`jitter_scale`, `jitter_shift`), jamais sur des cadrages parfaits : sinon décalage train/test garanti, puisqu'à l'inférence les cadrages viennent d'un détecteur. La validation, elle, utilise des cadrages nets — une métrique de validation bruitée ne servirait à rien.
- Une marge (`crop.padding`) entoure la bbox. Sans elle, tarses et antennes tombent hors du crop et deviennent irrécupérables quelle que soit la qualité du modèle. Les points hors cadre sont marqués `vis = 0` : ni appris comme des zéros, ni comptés comme des erreurs.
- La transformation crop → image est conservée et **toute prédiction est rétro-projetée** vers le repère de l'image d'origine avant écriture (contrat 3).
- L'évaluation bout-en-bout utilise les bboxes **prédites**. Le mode `pose_on_gt_boxes: true` écrit `bbox_source: gt` : diagnostic uniquement, jamais dans le même tableau que les approches bout-en-bout.

### 9.4 LoRA

- Les artefacts sauvegardés sont les **adaptateurs seuls** + une référence explicite au modèle de base (nom, version, hash). Un run LoRA non rechargeable sans le bon modèle de base est invalide.
- La config déclare explicitement : modules ciblés, rang, alpha, dropout, et **quels paramètres restent entraînables hors adaptateurs** (têtes, biais, normalisations). C'est souvent la vraie variable cachée d'une comparaison LoRA.

### 9.5 BatchNorm par groupe (domain-specific BN)

- Repose sur `meta.dataset_index` (§4.3). Les statistiques BN sont maintenues **par dataset** ; à l'inférence, le groupe est choisi par `dataset` de l'image.
- Piège à documenter : ce que fait l'approche quand le dataset est inconnu à l'inférence (moyenne des groupes ? groupe par défaut ?). Le comportement DOIT être explicite en config, pas implicite dans le code.
- Les batches mixtes multi-datasets doivent être supportés ; l'implémentation DOIT passer un test unitaire vérifiant qu'un batch mixte donne le même résultat que N batches purs.

### 9.6 Approches futures

Toute nouvelle approche (multi-tâches, distillation, pré-entraînement auto-supervisé, ensembles…) s'ajoute par §11 sans dérogation. Si une approche ne rentre pas dans le protocole `Approach`, **on modifie le protocole pour tout le monde, en bumpant sa version** — on ne crée pas de cas particulier.

---

## 10. Tests

Trois niveaux, tous obligatoires avant toute exécution longue :

1. **Tests de contrat** (`tests/contracts/`) : valident qu'un parquet produit respecte le schéma, les bornes de coordonnées, l'unicité des identifiants, la cohérence keypoint_schema ↔ dimension. Exécutés automatiquement à l'écriture de tout artefact en mode `debug`.
2. **Tests unitaires** : rétro-projection crop→image (aller-retour = identité à 1e-6 près), mapping local↔union, appariement, chaque métrique sur un cas calculé à la main, non-fuite (`fit` ne lit pas `test`), reproductibilité (deux runs même seed = mêmes prédictions).
3. **Smoke test** (`make smoke`) : chaque approche enregistrée est exécutée sur un fixture de 8 images et 1 fold, de `train` à `report`. **Une approche qui ne passe pas le smoke test n'est pas considérée comme implémentée.** Le fixture est commité dans `tests/fixtures/`.

CI : ruff + mypy (strict sur `contracts.py`, `registry.py`, `evaluation/`) + pytest + smoke.

---

## 11. Procédure : ajouter une approche

Exactement 6 artefacts, ni plus ni moins. Si vous devez toucher un 7ᵉ fichier existant, c'est un signal de conception à remonter.

1. `src/insectpose/approaches/<nom>.py` — classe décorée `@register_approach("<nom>")`, implémentant §4.2.
2. `configs/approach/<nom>.yaml` — hyperparamètres par défaut, `_target_` vers la classe.
3. `search_space` dans la classe (ou `tuning/search_spaces.py` si volumineux).
4. `tests/approaches/test_<nom>.py` — smoke + tests spécifiques (ex. §9.5).
5. `configs/experiment/exp_<lettre>_<nom>.yaml` — expérience figée pour le rapport.
6. Une entrée dans `DECISIONS.md` : ce que l'approche teste, ses hypothèses, ses limites connues.

---

## 12. Règles de génération pour les IA

À respecter par toute IA produisant du code sur ce dépôt.

**Obligations**

- Déclarer les dépendances lourdes via `availability()` : le smoke test ignore proprement une approche indisponible plutôt que d'échouer.
- Écrire en **anglais** tout texte destiné à un fichier produit (§8.3), en français le code et ses commentaires.
- Lire ce fichier et annoncer, avant d'écrire, quels contrats sont touchés.
- Écrire des signatures typées ; `contracts.py` fait foi pour les types de données.
- Toute fonction publique a une docstring indiquant : entrées, sorties, **effets de bord fichiers** (chemin exact écrit).
- Valider les entrées aux frontières de module (schéma parquet, présence de clés de config) et échouer tôt, bruyamment, avec un message actionnable.
- Produire, avec tout nouveau module, son test correspondant. Code sans test = non livré.
- Toute décision méthodologique non triviale prise en cours de route → ligne dans `DECISIONS.md`, pas un commentaire enterré dans le code.

**Interdictions**

- Pas de chemin en dur, pas de constante magique, pas de seuil littéral dans un `.py`.
- Pas de `try/except` silencieux, pas de `except Exception: pass`, pas de valeur de repli qui masque une donnée manquante.
- Pas de logique d'approche dans `training/`, `evaluation/`, `tuning/`, `reporting/` — aucun `if approach == ...` nulle part.
- Pas de calcul de métrique hors de `evaluation/metrics/`.
- Pas de mutation de `data/raw/`. Jamais.
- Pas de dépendance nouvelle sans justification et ajout à `pyproject.toml`.
- Pas de notebook comme source de vérité : un notebook appelle le package, il ne contient pas de logique.
- Pas de fichier « utils.py » fourre-tout : un module = une responsabilité nommable en une phrase.
- Pas de refactor opportuniste hors du périmètre demandé.

**Quand s'arrêter et demander**
Une IA DOIT interrompre la génération et poser la question si : un contrat devrait changer ; deux approches exigeraient un champ incompatible ; une métrique est ambiguë ; le schéma de keypoints d'un dataset est inconnu ; une décision affecterait la comparabilité entre approches. Inventer une convention pour continuer est la faute la plus coûteuse du projet.

**Gabarit de prompt de tâche recommandé**

```
Contexte : CONVENTIONS.md v1.0 (fourni intégralement).
Tâche : implémenter <X>.
Périmètre : fichiers autorisés à créer/modifier = [...]. Tout le reste est en lecture seule.
Contrats touchés : [aucun | n° ...].
Livrables : code + tests + entrée DECISIONS.md si décision prise.
Critère d'acceptation : `make smoke` passe pour l'approche <X>.
Si une règle de CONVENTIONS.md bloque : arrête-toi et explique.
```

---

## 13. Décisions de protocole (toutes tranchées)

Consignées dans `DECISIONS.md`. Elles sont **fermées** : les modifier invalide les résultats déjà produits.

| #        | Décision                | Valeur retenue                                                                      |
| -------- | ------------------------ | ----------------------------------------------------------------------------------- |
| ADR-0006 | Schéma de keypoints     | `insect42_v1`, 42 points, commun aux 4 datasets, union = identité                |
| ADR-0007 | Sigmas OKS               | `sigma = difficulty × 0.0025` (10→0.025 … 40→0.100)                           |
| ADR-0008 | Mesures morphométriques | 27 mesures + 9 paires symétriques, métriques de premier plan                      |
| ADR-0009 | Normalisation PCK        | `alpha × largeur du thorax`, référence `alpha = 0.25`, taux de repli publié |
| ADR-0010 | Métrique primaire       | `oks_ap`, seule clé d'évaluation librement surchargeable                        |
| ADR-0011 | Groupement anti-fuite    | une image = un specimen,`group_id = image_id`                                     |
| ADR-0012 | HPO                      | nichée : recherche sur folds internes, test externe jamais vu                      |
| ADR-0013 | Résolution d'entrée    | 640×640 pour toutes les approches, garde-fou strict                                |
| ADR-0014 | Dataset à l'inférence  | toujours déclaré ; un dataset inconnu est une erreur explicite                    |
| ADR-0015 | Suivi d'expériences     | manifestes +`master.parquet` uniquement                                           |

Restent ouverts sans bloquer : le budget d'HPO réellement soutenable (OPEN-09) et le traitement des keypoints systématiquement absents dans un dataset donné (OPEN-10).

---

*Fin du contrat. Toute évolution passe par un incrément de version de ce fichier et une entrée dans `DECISIONS.md`*

# CONVENTIONS.md — Règles d'architecture et de génération de code

**Projet :** estimation de pose sur 4 datasets d'insectes (Coleoptera, Diptera, Hymenoptera, Lepidoptera)
**Statut :** contrat normatif. Ce fichier fait autorité sur tout autre document du dépôt.
**Version du contrat :** 2.0 — décisions de protocole tranchées (ADR-0006 à 0015)

---

## 0. Comment utiliser ce fichier

Ce document est destiné à être fourni **en entier et en contexte** à toute IA générative (ou tout contributeur humain) chargée d'écrire du code dans ce dépôt.

Règle zéro : **toute génération de code doit citer, en commentaire d'en-tête du fichier produit, les sections de ce document qu'elle applique.** Si une instruction utilisateur contredit ce fichier, l'IA doit s'arrêter et signaler le conflit au lieu de trancher seule.

Vocabulaire normatif : **DOIT** / **NE DOIT PAS** = contrainte dure, non négociable. **DEVRAIT** = recommandation forte, dérogation possible si documentée dans `DECISIONS.md`. **PEUT** = libre.

---

## 1. Principes directeurs

1. **L'approche est un plugin, pas une branche de `if`.** Ajouter une 6ᵉ approche NE DOIT PAS modifier le code d'entraînement générique, d'évaluation, d'optimisation ou de reporting. Si vous devez modifier `evaluation/` pour ajouter une approche, l'abstraction est mauvaise : signalez-le.
2. **Les contrats de données sont l'API du projet.** Les approches ne communiquent jamais entre elles ni avec l'évaluateur par des objets Python : elles communiquent par des **fichiers au format figé** (§3). Cela permet d'entraîner avec Ultralytics, PyTorch pur, HuggingFace/PEFT, ou un modèle externe, sans que l'évaluateur ne le sache jamais.
3. **Une seule implémentation des métriques.** Aucune métrique NE DOIT être lue depuis les logs d'un framework tiers. Les métriques internes d'Ultralytics, de PyTorch Lightning ou d'un autre entraîneur servent **uniquement au monitoring**, jamais à la comparaison entre approches (§7.1).
4. **Séparation stricte : `fit` ≠ `predict` ≠ `evaluate` ≠ `aggregate`.** Quatre étapes, quatre artefacts, quatre points de reprise. On DOIT pouvoir ré-évaluer une expérience vieille de trois mois sans réentraîner.
5. **Tout est configuration ; rien n'est en dur.** Aucun chemin, hyperparamètre, seuil, taille d'image, nom de classe ou de keypoint NE DOIT apparaître littéralement dans un `.py`. Tout vient d'un YAML ou du `RunContext`.
6. **Reproductibilité par construction.** `run_id` déterministe, seeds explicites, config résolue sérialisée dans le dossier de run, versions de dépendances figées (§6.4).
7. **Coût de l'ignorance.** Toute approche DOIT être exécutable en mode `smoke` (2 epochs, 8 images, 1 fold) pour valider le branchement de bout en bout en < 2 minutes, avant tout entraînement réel.

---

## 2. Arborescence du dépôt

```
insectpose/
├── CONVENTIONS.md              # ce fichier — fait autorité
├── DECISIONS.md                # journal des choix méthodologiques (ADR, append-only)
├── README.md                   # démarrage rapide uniquement, pas de doctrine
├── pyproject.toml              # dépendances figées, config ruff/mypy/pytest
├── Makefile                    # raccourcis: make smoke / make tune / make eval / make report
│
├── configs/                    # composition Hydra — SEULE source de paramètres
│   ├── config.yaml             # config racine + defaults list
│   ├── paths.yaml              # racines de chemins (surchargées par machine)
│   ├── data/                   # coleoptera.yaml diptera.yaml ... pooled.yaml
│   ├── keypoints/              # schémas de keypoints par dataset + union (§3.1)
│   ├── approach/               # yolo_pooled.yaml yolo_per_dataset.yaml
│   │                           # detect_then_pose.yaml lora.yaml group_bn.yaml
│   ├── cv/                     # kfold5.yaml kfold5_grouped.yaml holdout.yaml
│   ├── eval/                   # default.yaml (métriques, seuils, sigmas OKS)
│   ├── tuning/                 # optuna_default.yaml + budgets par approche
│   └── experiment/             # compositions nommées et figées (§5.3)
│
├── data/
│   ├── raw/                    # IMMUABLE, jamais écrit par le code, jamais commité
│   ├── interim/                # sorties d'adaptateurs, régénérable
│   ├── processed/              # format canonique (§3.2), régénérable
│   └── splits/                 # assignations de folds versionnées et hashées (§3.3)
│
├── src/insectpose/
│   ├── contracts.py            # dataclasses/TypedDict des 5 contrats — INTOUCHABLE sans bump
│   ├── registry.py             # registre par nom (approches, métriques, adaptateurs)
│   ├── paths.py                # unique endroit qui construit des chemins
│   ├── context.py              # RunContext (run_id, seed, fold, dossiers, logger)
│   │
│   ├── data/
│   │   ├── schema.py           # validation du format canonique
│   │   ├── adapters/           # raw -> canonique, un module par source
│   │   ├── keypoints.py        # mapping par-dataset <-> espace union
│   │   ├── datamodule.py       # canonique -> batches (superset de champs, §4.3)
│   │   └── splits.py           # génération et lecture des folds
│   │
│   ├── approaches/
│   │   ├── base.py             # Protocol Approach + BaseApproach
│   │   ├── yolo_pooled.py
│   │   ├── yolo_per_dataset.py
│   │   ├── detect_then_pose.py
│   │   ├── lora.py
│   │   └── group_bn.py
│   │
│   ├── models/                 # briques réutilisables (backbones, têtes, adaptateurs LoRA, GroupBN)
│   ├── training/               # boucles génériques, callbacks, early stopping
│   ├── evaluation/
│   │   ├── metrics/            # une métrique = un module enregistré
│   │   ├── matching.py         # appariement pred<->gt (OKS/IoU), partagé
│   │   ├── evaluator.py        # predictions.parquet -> metrics.parquet
│   │   └── aggregate.py        # tous les runs -> results/master.parquet
│   ├── tuning/
│   │   ├── search_spaces.py    # espaces Optuna, un par approche
│   │   └── objective.py        # objectif générique (§6.3)
│   ├── reporting/              # tableaux, figures, tests statistiques
│   ├── cli.py                  # points d'entrée (§5.4)
│   └── utils/                  # seed, io, hashing, geometry, logging
│
├── runs/                       # artefacts d'exécution, non commités (§8)
├── results/                    # agrégats consolidés, parquet + figures
├── reports/                    # livrables (notebooks exportés, PDF, slides)
└── tests/                      # unitaires + contrat + smoke (§10)
```

**Règle d'or de l'arborescence :** un fichier `.py` NE DOIT PAS écrire hors de `runs/<run_id>/`, `data/interim/`, `data/processed/`, `data/splits/` et `results/`. Toute autre écriture est un bug.

---

## 3. Les cinq contrats

Ce sont les cinq formats figés qui rendent le projet modulaire. Chacun porte un champ `schema_version`. **Modifier un contrat DOIT se faire par incrément de version + lecteur rétrocompatible**, jamais par modification en place.

### 3.1 Contrat 0 — Schéma de keypoints (`configs/keypoints/insect42_v1.yaml`)

Les quatre datasets partagent **un seul schéma de 42 points** (ADR-0006). L'espace union est ce schéma lui-même : le mapping est l'identité, et le mécanisme d'union reste en place pour absorber une divergence future sans refonte.

```yaml
schema_version: 1
name: insect42_v1
status: VALIDATED
union_space: insect42_v1
sigma_from_difficulty: {scale: 0.0025}   # sigma = difficulty * scale (ADR-0007)
keypoints:
  - {name: thorax-left,  union: thorax-left,  difficulty: 30, flip: thorax-right}
  - {name: thorax-right, union: thorax-right, difficulty: 30, flip: thorax-left}
skeleton: [[0, 5], [0, 12], ...]         # 51 arêtes anatomiques
```

Règles :

- **L'ordre des 42 points est figé à vie** : il est encodé dans tous les artefacts produits. Ajouter un point = l'ajouter *en fin de liste* et bumper `schema_version`.
- Les tolérances OKS ne sont **pas** écrites en dur : `sigma = difficulty × scale`, où `difficulty` (10 à 40) est la difficulté de positionnement précis fournie par l'expert. Un point difficile à annoter est jugé avec plus d'indulgence, ce qui évite que la métrique soit dominée par le bruit d'annotation. Modifier `scale` change la définition de l'OKS : bumper `eval.version` et rejouer les runs.
- `flip` définit les paires de symétrie ; toute augmentation par miroir sans cette table est interdite. Les points de l'axe médian sont leur propre miroir.
- Un schéma marqué `status: PLACEHOLDER` est refusé quand `strict.require_validated_keypoints` est vrai (valeur par défaut).
- **Mesures morphométriques** (`configs/measurements/insect42_v1.yaml`, ADR-0008) : 27 mesures définies comme des polylignes de keypoints, plus 9 paires gauche/droite. C'est la grandeur réellement consommée en aval, donc une métrique de premier plan — pas une annexe.

### 3.2 Contrat 1 — Annotations canoniques (`data/processed/<dataset>/annotations.parquet`)

Une ligne = une **instance annotée**. Format unique quelle que soit la source d'origine (COCO, CVAT, CSV…).

| colonne                           | type             | description                                                                    |
| --------------------------------- | ---------------- | ------------------------------------------------------------------------------ |
| `schema_version`                | int              | 1                                                                              |
| `dataset`                       | str              | `coleoptera` \| `diptera` \| `hymenoptera` \| `lepidoptera`            |
| `image_id`                      | str              | identifiant**globalement unique** : `<dataset>/<nom_fichier_sans_ext>` |
| `image_path`                    | str              | chemin**relatif à `paths.data_root`**, jamais absolu                  |
| `image_width`, `image_height` | int              | pixels, image d'origine                                                        |
| `instance_id`                   | str              | `<image_id>#<n>`                                                             |
| `group_id`                      | str              | clé anti-fuite : spécimen, planche, session de capture (§6.1)               |
| `bbox_xywh`                     | list[float] (4)  | coordonnées**image d'origine**, pixels absolus                          |
| `kpts_xy`                       | list[float] (2K) | ordre du schéma local, pixels absolus, image d'origine                        |
| `kpts_vis`                      | list[int] (K)    | 0 absent / 1 occulté / 2 visible                                              |
| `area`                          | float            | aire du segment ou de la bbox                                                  |
| `keypoint_schema`               | str              | nom du schéma de §3.1                                                        |
| `split_source`                  | str              | `train` \| `test_officiel` \| `unknown` si un découpage amont existe    |

Règles :

- **Toutes les coordonnées, partout, dans tous les fichiers, sont exprimées dans le repère de l'image d'origine, en pixels absolus.** Aucun format normalisé, aucun `xyxy` relatif, aucune coordonnée dans un repère de crop ne doit jamais quitter un module.
- Les adaptateurs (`data/adapters/`) sont les **seuls** modules autorisés à connaître les formats sources. Un adaptateur ne fait que : lire → convertir → valider (`schema.py`) → écrire. Aucun filtrage, aucune augmentation, aucune décision méthodologique.
- Les instances invalides (keypoints hors image, bbox nulle) sont **conservées** avec un flag `qc_flags`, pas supprimées ; le filtrage est une décision de config, pas d'adaptateur.

### 3.3 Contrat 2 — Splits (`data/splits/<split_id>.parquet` + `.json`)

| colonne      | type                              |
| ------------ | --------------------------------- |
| `split_id` | str, ex.`kfold5_grouped_seed42` |
| `image_id` | str                               |
| `fold`     | int                               |
| `role`     | `train` \| `val` \| `test`  |

Règles :

- Les folds sont **générés une seule fois** et **partagés par toutes les approches**. Une approche NE DOIT JAMAIS créer ses propres splits.
- L'unité de découpage est `group_id`, pas `image_id` (§6.1).
- Le `.json` compagnon contient : seed, stratégie, stratification, comptages par dataset/fold, et un `content_hash` des annotations utilisées. **Si le hash des annotations change, les splits sont invalidés** et le pipeline DOIT refuser de tourner.

### 3.4 Contrat 3 — Prédictions (`runs/<run_id>/predictions/<split>_fold<k>.parquet`)

C'est **le** contrat qui rend les approches interchangeables. Une ligne = une instance prédite.

| colonne                                      | type             | description                                                                                                                                          |
| -------------------------------------------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                           | int              | 1                                                                                                                                                    |
| `run_id`, `fold`, `split`, `dataset` | str/int          |                                                                                                                                                      |
| `image_id`                                 | str              |                                                                                                                                                      |
| `pred_id`                                  | str              | unique                                                                                                                                               |
| `bbox_xywh`                                | list[float] (4)  | repère image d'origine ; obligatoire même pour une approche pose-only (alors = bbox GT ou bbox englobante des kpts, et`bbox_source` le précise) |
| `bbox_score`                               | float            | 1.0 si non applicable                                                                                                                                |
| `kpts_xy`                                  | list[float] (2K) | **repère image d'origine**, schéma local du dataset                                                                                          |
| `kpts_score`                               | list[float] (K)  |                                                                                                                                                      |
| `keypoint_schema`                          | str              | doit correspondre au schéma du dataset de l'image                                                                                                   |
| `bbox_source`                              | str              | `predicted` \| `gt` \| `derived`                                                                                                               |
| `inference_ms`                             | float            | temps par instance, pour la comparaison coût/perf                                                                                                   |

Règles :

- **Aucun seuil de score n'est appliqué à l'écriture.** On écrit toutes les prédictions au-dessus d'un seuil très bas (ex. 0.001) ; le seuillage est une opération d'évaluation, paramétrée en config. Sinon les courbes P/R sont tronquées et les approches deviennent incomparables.
- Toute approche opérant sur **crop** (pipeline détection→pose, §9.3) DOIT conserver la transformation affine crop→image et **rétro-projeter** avant écriture. Écrire des coordonnées dans le repère du crop est une erreur bloquante.
- Un modèle entraîné dans l'espace union DOIT projeter vers le schéma local avant écriture (§3.1).
- Les prédictions sur `test` d'un fold ne DOIVENT contenir que les images de ce fold.

### 3.5 Contrat 4 — Métriques (`runs/<run_id>/metrics.parquet`) et 5 — Manifeste (`runs/<run_id>/manifest.json`)

`metrics.parquet` — format long, jamais large :

| colonne                                       | description                                            |
| --------------------------------------------- | ------------------------------------------------------ |
| `run_id`, `approach`, `fold`, `split` |                                                        |
| `scope`                                     | `overall` \| `dataset:<nom>` \| `keypoint:<nom>` |
| `metric`                                    | nom canonique, ex.`pck@0.05_bboxdiag`                |
| `value`                                     | float                                                  |
| `n`                                         | taille de l'échantillon sous-jacent                   |

`manifest.json` : `run_id`, timestamp, `approach`, `split_id`, config Hydra **résolue** (pas les surcharges CLI), `content_hash` des données, commit git + état propre/sale du dépôt, versions des dépendances clés, seeds, chemins des artefacts produits, durées, ressources GPU, et `optuna_study`/`trial_number` si applicable. **Un run sans manifeste complet est exclu de l'agrégation.**

---

## 4. Interfaces et registre

### 4.1 Registre

Un décorateur unique, un espace de noms par famille :

```python
@register_approach("lora")            # approches
@register_metric("pck")               # métriques
@register_adapter("coleoptera_cvat")  # adaptateurs de données
```

Le nom enregistré DOIT être identique au nom du fichier YAML de config correspondant. Aucun `import` conditionnel, aucun `if approach == ...` ailleurs que dans le registre.

### 4.2 Protocole `Approach`

Toute approche DOIT implémenter exactement cette interface, ni plus ni moins côté pipeline :

```python
class Approach(Protocol):
    name: str

    def fit(self, data: FoldData, ctx: RunContext) -> None: ...
        # entraîne sur data.train, valide sur data.val ; écrit ses poids dans ctx.run_dir/weights/
        # NE DOIT PAS toucher à data.test

    def predict(self, images: ImageSet, ctx: RunContext) -> Path: ...
        # retourne le chemin d'un predictions parquet conforme au Contrat 3

    @classmethod
    def load(cls, run_dir: Path, cfg: DictConfig) -> "Approach": ...
        # reconstruit un modèle prédictif depuis les artefacts, sans réentraînement

    @classmethod
    def search_space(cls, trial: optuna.Trial) -> dict: ...
        # surcharges de config proposées à Optuna ; aucune logique d'entraînement ici
```

Règles :

- `fit` NE DOIT JAMAIS accéder à `data.test`. Un test unitaire vérifie cette propriété (§10).
- `predict` NE DOIT JAMAIS calculer de métrique.
- Une approche PEUT s'appuyer sur plusieurs sous-modèles (cf. détection+pose) : c'est son affaire interne, invisible du pipeline.
- Une approche « par dataset » (§9.2) reste **une seule** approche : elle encapsule N modèles et route selon `dataset`. Le pipeline ne doit pas voir la différence.

### 4.3 DataModule : superset de champs

Le batch produit par le datamodule DOIT toujours contenir le **superset** des champs utiles à toutes les approches, même si une approche donnée les ignore :

```
images, bboxes, keypoints, visibility, meta{image_id, instance_id, dataset,
dataset_index, group_id, orig_size, transform_matrix}
```

`dataset_index` est indispensable à l'approche BatchNorm par groupe (§9.5) ; `transform_matrix` à la rétro-projection. Les ajouter au coup par coup casse la modularité : ils sont là dès le départ.

---

## 5. Configuration

### 5.1 Outil

Hydra + OmegaConf. Composition par `defaults`, surcharge CLI par `clé=valeur`. Pas de `argparse` manuel, pas de dictionnaires de config codés en Python.

### 5.2 Règles

- Un fichier YAML par entité nommée ; le nom du fichier est l'identifiant.
- Toute clé DOIT avoir une valeur par défaut explicite ; interdiction du `cfg.get("x", 3)` disséminé dans le code.
- Les configs d'approche contiennent **uniquement** ce qui est spécifique à l'approche. Les paramètres communs (taille d'image, batch, epochs, seuils d'éval) vivent dans `config.yaml` et sont surchargeables.
- Interdiction d'interpolations Hydra qui traversent plus d'un niveau (`${a.b.c.d}` illisible) : préférer un champ explicite.
- La config **résolue** est écrite dans `runs/<run_id>/config.yaml` **avant** tout entraînement.

### 5.3 Expériences nommées

Toute exécution destinée au rapport final DOIT passer par un fichier `configs/experiment/*.yaml` figé et commité (ex. `exp_A_yolo_pooled_kfold5.yaml`). Les surcharges CLI ad hoc sont réservées à l'exploration et NE DOIVENT PAS produire de résultats cités dans le rapport.

### 5.4 CLI

Cinq verbes, pas plus :

```
python -m insectpose.cli prepare   data=coleoptera
python -m insectpose.cli split     cv=kfold5_grouped
python -m insectpose.cli train     experiment=exp_A cv.fold=0
python -m insectpose.cli predict   run_id=<...> split=test
python -m insectpose.cli evaluate  run_id=<...>
python -m insectpose.cli tune      experiment=exp_A tuning=optuna_default
python -m insectpose.cli report
```

`train` PEUT enchaîner `predict` + `evaluate` par commodité, mais chacun DOIT rester appelable indépendamment.

---

## 6. Protocole expérimental

### 6.1 Anti-fuite

- Le découpage se fait par `group_id`. Si un spécimen apparaît sur plusieurs images, toutes ses images sont dans le même fold. **Si le `group_id` n'est pas connu pour un dataset, la valeur par défaut est `image_id` et cette limitation DOIT être écrite dans `DECISIONS.md`.**
- Stratification par `dataset` (et par nombre d'instances par image si déséquilibré) obligatoire pour les folds poolés.
- Aucune statistique (moyenne/écart-type de normalisation, taille d'ancres, clustering de keypoints) NE DOIT être calculée sur autre chose que le `train` du fold courant.

### 6.2 Cross-validation

- Schéma par défaut : **K=5 folds groupés stratifiés**, seed fixe, `split_id` unique partagé par toutes les approches. Les mêmes folds pour tout le monde, sinon aucune comparaison n'est valide.
- Les approches « par dataset » (§9.2) utilisent **les mêmes folds**, simplement restreints à leur dataset. Ne jamais régénérer un découpage local.
- Une approche est comparée sur la **moyenne ± écart-type inter-folds**, et les résultats par fold sont conservés pour les tests appariés (§8.3).

### 6.3 Optimisation Optuna

- **Nichée par défaut** (ADR-0012). Pour chaque fold externe, la recherche tourne sur des folds **internes** construits à partir du seul train externe. Ces découpages internes (`<split_id>__outer<k>`) sont générés par `cli split` et versionnés exactement comme les folds externes. Les meilleurs hyperparamètres sont ensuite appliqués au fold externe entier. **Le test externe n'a jamais servi à choisir un hyperparamètre.** Un test automatique vérifie cette propriété.
- Mode dégradé `tune_once` : la recherche n'a lieu que sur les folds internes d'un seul fold externe, et le résultat est réutilisé pour tous les autres. Acceptable si documenté ; le budget de trials doit alors être identique entre approches.
- **Coût** : `n_folds × n_trials × inner_folds` entraînements par approche. À calibrer avant de lancer une approche lourde ; le budget effectif est enregistré dans chaque manifeste.
- L'objectif est **toujours la métrique primaire calculée par l'évaluateur partagé**, lue depuis `metrics.parquet` — jamais une loss de validation ni une métrique interne de framework.
- Un trial = un run complet avec son propre `run_id` et son manifeste ; les trials sont donc évaluables et auditables comme n'importe quel run. Ils n'exportent pas de figures qualitatives (bruit inutile).
- Stockage SQLite sous `runs/optuna/`, une étude par (approche, découpage, objectif, fold externe), reprise activée.
- Pruning `MedianPruner` par défaut ; une approche qui ne peut pas rapporter d'intermédiaire déclare `prunable: false`.
- **Budget équitable** : comparer 100 trials contre 10 invalide la conclusion.

### 6.4 Déterminisme

- Seed unique dans la config, dérivée par `seed_for(run_id, fold, purpose)` pour numpy / torch / python / dataloader workers.
- `torch.use_deterministic_algorithms(True)` en mode `debug` ; en mode `full` on autorise cudnn benchmark mais on l'enregistre dans le manifeste.
- Le non-déterminisme résiduel est absorbé par la répétition : toute conclusion finale DEVRAIT reposer sur ≥ 2 seeds pour l'approche gagnante.

---

## 7. Évaluation

### 7.1 Règle absolue

L'évaluateur prend **uniquement** : un `predictions.parquet` (Contrat 3), les annotations canoniques (Contrat 1), et `configs/eval/*.yaml`. Il ne charge aucun modèle, n'importe aucun module d'approche, et ignore totalement comment les prédictions ont été produites. **Si l'évaluateur doit savoir quelle approche l'a alimenté, le design est cassé.**

### 7.2 Jeu de métriques figé

Identique pour toutes les approches, calculé en `overall`, par `dataset:*`, par `keypoint:*` et par `measurement:*` :

- **Détection** (si `bbox_source == predicted`) : `det_ap@0.5`, `det_ap@[.5:.95]`.
- **Pose** : `oks_ap`, `oks_ap@0.5`, `oks_ar` (sigmas dérivés de la difficulté, ADR-0007) ; `pck@{0.125, 0.25, 0.5}_thorax_width` — un point est correct si son erreur est inférieure à `alpha × largeur du thorax` (ADR-0009), `alpha = 0.25` étant la référence du projet ; `nme_matched_only`, `kpt_coverage`, PCK par keypoint.
- **Échelle de référence** : `pck_normalizer_fallback_rate`. Quand les points de thorax ne sont pas annotés, la normalisation retombe sur la diagonale de bbox — et ce taux de repli est **publié**, jamais silencieux.
- **Mesures morphométriques** (ADR-0008) : `measurement_mape_median`, `measurement_mape_worst`, détail par mesure, et `symmetry_gap_median` / `symmetry_gap_p90` — l'écart gauche/droite des mesures prédites, calculable **sans vérité terrain**, donc utilisable comme contrôle qualité en production.
- **Bout-en-bout** : la métrique primaire pénalise les échecs de détection. Une pipeline qui ne détecte pas l'insecte n'a pas « 0 keypoint évalué », elle a un échec compté.
- **Coût** : latence par instance et p95, nombre de paramètres, VRAM, temps d'entraînement — métriques de premier ordre, pas des annexes.

**Métrique primaire du projet** : `oks_ap` (ADR-0010). C'est la **seule clé d'évaluation librement surchargeable** — elle ne modifie aucun calcul, seulement l'objectif d'Optuna et le classement des approches. Toutes les métriques étant calculées à chaque run, changer d'objectif n'oblige jamais à réévaluer :

```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```

### 7.3 Appariement

L'appariement prédiction↔GT (par OKS ou IoU, greedy par score décroissant) est implémenté **une seule fois** dans `evaluation/matching.py`. Aucune métrique ne réimplémente son propre appariement.

### 7.4 Comparaison des approches sur périmètre commun

Les approches n'ont pas le même périmètre naturel (une approche par dataset ne prédit rien hors de son dataset). Règle : **toute comparaison se fait sur l'union des images de test de tous les folds**, une approche restreinte étant évaluée comme la concaténation de ses N modèles. Un tableau de résultats DOIT indiquer le `n` sous-jacent de chaque cellule (§3.5) ; deux valeurs avec des `n` différents ne sont pas comparables et le rapport DOIT le signaler.

---

## 8. Runs, artefacts et résultats

### 8.1 `run_id`

```
<approach>__<data_scope>__<split_id>__fold<k>__<tag>__<hash8>
ex. lora__pooled__kfold5grouped_seed42__fold2__baseline__a3f91c07
```

`hash8` = 8 premiers caractères du hash de la config résolue + du `content_hash` des données. Deux runs identiques ont le même `run_id` : le pipeline DOIT alors sauter le run (idempotence) sauf `force=true`.

### 8.2 Contenu d'un dossier de run

```
runs/<run_id>/
├── manifest.json          # Contrat 5, écrit en dernier -> sa présence signale un run complet
├── config.yaml            # config résolue
├── weights/               # poids, checkpoints, adaptateurs LoRA
├── predictions/           # Contrat 3
├── metrics.parquet        # Contrat 4
├── logs/                  # stdout, courbes, tensorboard/mlflow
└── figures/               # visualisations qualitatives (§8.4)
```

`manifest.json` est écrit **en dernier**. Un dossier sans manifeste = run interrompu, ignoré par l'agrégation, supprimable sans discussion.

### 8.3 Langue des livrables

**Tout ce qui est écrit dans un fichier produit est en anglais** : titres, axes, légendes et annotations de figures, en-têtes et valeurs textuelles de tableaux, champs de manifestes et de rapports JSON, noms de fichiers. Les livrables circulent hors de l'équipe et finissent dans des publications ; une figure en français y est inutilisable.

Le code, les commentaires, les docstrings, les messages de log et la documentation interne (`CONVENTIONS.md`, `DECISIONS.md`, `README.md`) restent en français. La frontière est nette : ce qui sort dans `results/`, `runs/` ou `reports/` est en anglais, le reste non.

Corollaire : un nom de métrique, de scope ou de colonne est un identifiant, jamais une phrase à traduire. `oks_ap`, `dataset:coleoptera`, `measurement_mape_median` sont figés (§3.5) et ne changent pas de langue.

### 8.4 Agrégation et reporting

- `aggregate.py` scanne `runs/*/metrics.parquet` + manifestes → `results/master.parquet`. **C'est le seul chemin vers un tableau de résultats.** Aucune figure, aucun tableau du rapport ne DOIT être produit à partir d'un copier-coller de console.
- Les comparaisons entre approches DEVRAIENT utiliser des tests appariés par fold (Wilcoxon signé ou t apparié) avec correction pour comparaisons multiples, et rapporter des intervalles de confiance plutôt que des rangs bruts.
- `reporting/` produit : tableau principal (approche × dataset × métrique), courbes PCK, scatter coût vs performance, matrice d'erreurs par keypoint, échecs qualitatifs.

### 8.5 Qualitatif obligatoire

Chaque run DOIT exporter au moins 12 images de test annotées pred vs GT, incluant les 6 pires cas selon la métrique primaire. Un modèle n'est jamais validé sur des chiffres seuls.

---

## 9. Contraintes spécifiques par approche

Ces notes fixent les pièges connus de chaque famille. Elles n'ajoutent aucune interface : tout passe par §4.2.

### 9.1 YOLO poolé (une classe « insecte », tous datasets) — IMPLÉMENTÉ

Le schéma de keypoints étant commun aux 4 ordres (ADR-0006), le modèle prédit directement dans le schéma attendu : aucune reprojection union → local n'est nécessaire. Les points absents d'un dataset (ADR-0016) sortent en `vis = 0` dans les labels et sont masqués dans la loss, jamais appris comme des zéros.

Toute la logique risquée est isolée dans `data/yolo_export.py`, testée par aller-retour sans GPU :

- la bbox YOLO est **centrée**, le contrat 1 est en coin haut-gauche ;
- `flip_idx` est obligatoire dans `data.yaml` dès que `fliplr > 0`, sinon le miroir échange gauche et droite sans permuter les labels ;
- les noms de fichiers sont aplatis (`coleoptera__img000`), sinon deux datasets ayant un `img000.png` se recouvrent silencieusement.

Les fichiers YOLO sont un artefact **dérivé**, régénéré par fold sous `runs/<run_id>/yolo_dataset/`, jamais écrit dans `data/processed/`. `conf = 0.001` à l'inférence : le seuillage est une opération d'évaluation.

Matériel (ADR-0019) : `train.device: auto` prend le GPU 0 si CUDA est disponible ; AMP activée par défaut mais désactivée en `mode: debug` ; FP16 à l'inférence sur GPU. VRAM maximale, temps d'entraînement et nombre de paramètres entrent dans le manifeste — ce sont des métriques de coût de premier ordre, et l'agrégation alerte si des runs comparés viennent de matériels différents.

### 9.2 YOLO par dataset

Une seule classe `Approach` encapsulant 4 modèles, routés par `meta.dataset`. Utilise les mêmes folds restreints (§6.2). L'HPO PEUT être fait par dataset, mais le budget total de trials DOIT être annoncé et comparable à celui des autres approches (sinon l'avantage vient du budget, pas de la méthode).

### 9.3 Détection puis pose sur crop

- Le modèle de pose est entraîné sur des crops issus de bboxes **GT bruitées** (jitter d'échelle et de translation paramétré en config), jamais sur des bboxes GT parfaites : sinon décalage train/test garanti.
- La `transform_matrix` crop→image est conservée dans `meta` et appliquée à la rétro-projection (§3.4).
- L'évaluation bout-en-bout utilise les bboxes **prédites**. Une évaluation avec bboxes GT est autorisée en **diagnostic** uniquement, écrite avec `bbox_source: gt` et **jamais mise dans le même tableau** que les résultats bout-en-bout.

### 9.4 LoRA

- Les artefacts sauvegardés sont les **adaptateurs seuls** + une référence explicite au modèle de base (nom, version, hash). Un run LoRA non rechargeable sans le bon modèle de base est invalide.
- La config déclare explicitement : modules ciblés, rang, alpha, dropout, et **quels paramètres restent entraînables hors adaptateurs** (têtes, biais, normalisations). C'est souvent la vraie variable cachée d'une comparaison LoRA.

### 9.5 BatchNorm par groupe (domain-specific BN)

- Repose sur `meta.dataset_index` (§4.3). Les statistiques BN sont maintenues **par dataset** ; à l'inférence, le groupe est choisi par `dataset` de l'image.
- Piège à documenter : ce que fait l'approche quand le dataset est inconnu à l'inférence (moyenne des groupes ? groupe par défaut ?). Le comportement DOIT être explicite en config, pas implicite dans le code.
- Les batches mixtes multi-datasets doivent être supportés ; l'implémentation DOIT passer un test unitaire vérifiant qu'un batch mixte donne le même résultat que N batches purs.

### 9.6 Approches futures

Toute nouvelle approche (multi-tâches, distillation, pré-entraînement auto-supervisé, ensembles…) s'ajoute par §11 sans dérogation. Si une approche ne rentre pas dans le protocole `Approach`, **on modifie le protocole pour tout le monde, en bumpant sa version** — on ne crée pas de cas particulier.

---

## 10. Tests

Trois niveaux, tous obligatoires avant toute exécution longue :

1. **Tests de contrat** (`tests/contracts/`) : valident qu'un parquet produit respecte le schéma, les bornes de coordonnées, l'unicité des identifiants, la cohérence keypoint_schema ↔ dimension. Exécutés automatiquement à l'écriture de tout artefact en mode `debug`.
2. **Tests unitaires** : rétro-projection crop→image (aller-retour = identité à 1e-6 près), mapping local↔union, appariement, chaque métrique sur un cas calculé à la main, non-fuite (`fit` ne lit pas `test`), reproductibilité (deux runs même seed = mêmes prédictions).
3. **Smoke test** (`make smoke`) : chaque approche enregistrée est exécutée sur un fixture de 8 images et 1 fold, de `train` à `report`. **Une approche qui ne passe pas le smoke test n'est pas considérée comme implémentée.** Le fixture est commité dans `tests/fixtures/`.

CI : ruff + mypy (strict sur `contracts.py`, `registry.py`, `evaluation/`) + pytest + smoke.

---

## 11. Procédure : ajouter une approche

Exactement 6 artefacts, ni plus ni moins. Si vous devez toucher un 7ᵉ fichier existant, c'est un signal de conception à remonter.

1. `src/insectpose/approaches/<nom>.py` — classe décorée `@register_approach("<nom>")`, implémentant §4.2.
2. `configs/approach/<nom>.yaml` — hyperparamètres par défaut, `_target_` vers la classe.
3. `search_space` dans la classe (ou `tuning/search_spaces.py` si volumineux).
4. `tests/approaches/test_<nom>.py` — smoke + tests spécifiques (ex. §9.5).
5. `configs/experiment/exp_<lettre>_<nom>.yaml` — expérience figée pour le rapport.
6. Une entrée dans `DECISIONS.md` : ce que l'approche teste, ses hypothèses, ses limites connues.

---

## 12. Règles de génération pour les IA

À respecter par toute IA produisant du code sur ce dépôt.

**Obligations**

- Déclarer les dépendances lourdes via `availability()` : le smoke test ignore proprement une approche indisponible plutôt que d'échouer.
- Écrire en **anglais** tout texte destiné à un fichier produit (§8.3), en français le code et ses commentaires.
- Lire ce fichier et annoncer, avant d'écrire, quels contrats sont touchés.
- Écrire des signatures typées ; `contracts.py` fait foi pour les types de données.
- Toute fonction publique a une docstring indiquant : entrées, sorties, **effets de bord fichiers** (chemin exact écrit).
- Valider les entrées aux frontières de module (schéma parquet, présence de clés de config) et échouer tôt, bruyamment, avec un message actionnable.
- Produire, avec tout nouveau module, son test correspondant. Code sans test = non livré.
- Toute décision méthodologique non triviale prise en cours de route → ligne dans `DECISIONS.md`, pas un commentaire enterré dans le code.

**Interdictions**

- Pas de chemin en dur, pas de constante magique, pas de seuil littéral dans un `.py`.
- Pas de `try/except` silencieux, pas de `except Exception: pass`, pas de valeur de repli qui masque une donnée manquante.
- Pas de logique d'approche dans `training/`, `evaluation/`, `tuning/`, `reporting/` — aucun `if approach == ...` nulle part.
- Pas de calcul de métrique hors de `evaluation/metrics/`.
- Pas de mutation de `data/raw/`. Jamais.
- Pas de dépendance nouvelle sans justification et ajout à `pyproject.toml`.
- Pas de notebook comme source de vérité : un notebook appelle le package, il ne contient pas de logique.
- Pas de fichier « utils.py » fourre-tout : un module = une responsabilité nommable en une phrase.
- Pas de refactor opportuniste hors du périmètre demandé.

**Quand s'arrêter et demander**
Une IA DOIT interrompre la génération et poser la question si : un contrat devrait changer ; deux approches exigeraient un champ incompatible ; une métrique est ambiguë ; le schéma de keypoints d'un dataset est inconnu ; une décision affecterait la comparabilité entre approches. Inventer une convention pour continuer est la faute la plus coûteuse du projet.

**Gabarit de prompt de tâche recommandé**

```
Contexte : CONVENTIONS.md v1.0 (fourni intégralement).
Tâche : implémenter <X>.
Périmètre : fichiers autorisés à créer/modifier = [...]. Tout le reste est en lecture seule.
Contrats touchés : [aucun | n° ...].
Livrables : code + tests + entrée DECISIONS.md si décision prise.
Critère d'acceptation : `make smoke` passe pour l'approche <X>.
Si une règle de CONVENTIONS.md bloque : arrête-toi et explique.
```

---

## 13. Décisions de protocole (toutes tranchées)

Consignées dans `DECISIONS.md`. Elles sont **fermées** : les modifier invalide les résultats déjà produits.

| #        | Décision                | Valeur retenue                                                                      |
| -------- | ------------------------ | ----------------------------------------------------------------------------------- |
| ADR-0006 | Schéma de keypoints     | `insect42_v1`, 42 points, commun aux 4 datasets, union = identité                |
| ADR-0007 | Sigmas OKS               | `sigma = difficulty × 0.0025` (10→0.025 … 40→0.100)                           |
| ADR-0008 | Mesures morphométriques | 27 mesures + 9 paires symétriques, métriques de premier plan                      |
| ADR-0009 | Normalisation PCK        | `alpha × largeur du thorax`, référence `alpha = 0.25`, taux de repli publié |
| ADR-0010 | Métrique primaire       | `oks_ap`, seule clé d'évaluation librement surchargeable                        |
| ADR-0011 | Groupement anti-fuite    | une image = un specimen,`group_id = image_id`                                     |
| ADR-0012 | HPO                      | nichée : recherche sur folds internes, test externe jamais vu                      |
| ADR-0013 | Résolution d'entrée    | 640×640 pour toutes les approches, garde-fou strict                                |
| ADR-0014 | Dataset à l'inférence  | toujours déclaré ; un dataset inconnu est une erreur explicite                    |
| ADR-0015 | Suivi d'expériences     | manifestes +`master.parquet` uniquement                                           |

Restent ouverts sans bloquer : le budget d'HPO réellement soutenable (OPEN-09) et le traitement des keypoints systématiquement absents dans un dataset donné (OPEN-10).

---

*Fin du contrat. Toute évolution passe par un incrément de version de ce fichier et une entrée dans `DECISIONS.md`.*
