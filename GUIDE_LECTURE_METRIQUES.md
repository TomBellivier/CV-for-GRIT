# Guide de lecture des métriques et figures

À garder ouvert à côté de `results/`. Pour chaque indicateur : ce qu'il mesure, ce qui constitue un bon signal, et l'action à mener selon le cas.

---

## 0. Ordre de lecture, en trois minutes

Ne regardez pas tout. Dans cet ordre, chaque étape conditionne la suivante :

1. **`pck_normalizer_fallback_rate`** et **`kpt_coverage`** — si l'un des deux dérape, toutes les autres métriques sont biaisées. Rien d'autre ne mérite d'être lu avant.
2. **`folds_oks_ap.png`** — la dispersion inter-folds. Si les boîtes se chevauchent entre deux approches, l'écart de moyenne n'existe pas.
3. **`metric_oks_ap.png`** — la performance par dataset. C'est le classement.
4. **`metric_measurement_mape_median.png`** — l'erreur sur ce que le projet produit réellement.
5. Les figures de diagnostic (couverture, difficulté, confiance, symétrie) uniquement si l'une des quatre premières pose question.

---

## 1. Métriques de validité — à lire en premier

### `pck_normalizer_fallback_rate`
Part des instances où la largeur de thorax n'était pas mesurable (points `thorax-left`/`thorax-right` non annotés), et où le PCK est retombé sur la diagonale de bbox.

| Valeur | Lecture | Action |
|---|---|---|
| 0 | Toutes les instances normalisées de la même façon | Rien |
| < 0,05 | Négligeable | Mentionner en note de bas de tableau |
| > 0,1 | **Deux échelles de normalisation coexistent** | Les PCK ne sont plus comparables entre datasets. Vérifier l'annotation du thorax du dataset concerné, ou basculer `eval.pck.normalizer.type=bbox_diag` pour tout le monde (et bumper `eval.version`) |

Regardez-le **par dataset** : un taux global de 8 % peut cacher 30 % sur un seul ordre.

### `kpt_coverage`
Part des keypoints annotés appartenant à une instance effectivement appariée à une prédiction. C'est le taux de réussite **détection + appariement**, pas la qualité de la pose.

| Valeur | Lecture | Action |
|---|---|---|
| > 0,95 | La détection n'est pas le problème | Travailler la pose |
| 0,7–0,95 | Échecs de détection notables | Vérifier `det_ap@0.5` : si bas, c'est la détection ; si haut, c'est le seuil d'appariement OKS (`eval.match_oks_threshold`) qui rejette des poses trop imprécises |
| < 0,7 | **Le modèle rate souvent l'insecte** | Inutile d'optimiser la pose. Augmenter `approach.box`, vérifier les tailles de bbox, examiner les figures qualitatives du run |
| = 0 sur un dataset | Aucune prédiction ne passe le seuil `score_threshold_pointwise` (0,5) | Regarder la distribution de `bbox_score` dans les prédictions brutes : un modèle sous-entraîné sort typiquement 0,2–0,4 |

Cette métrique est le **garde-fou de `nme_matched_only`** : une NME excellente sur 40 % des instances ne vaut rien.

---

## 2. Métriques de pose

### `oks_ap` — métrique primaire
Moyenne de l'AP sur 10 seuils d'OKS. Intègre détection, appariement, précision des points, et la qualité du **classement** par score de confiance.

| Situation | Lecture | Action |
|---|---|---|
| `oks_ap` proche de `oks_ap@0.5` | Le modèle est soit bon soit très bon | Rien |
| `oks_ap` très inférieur à `oks_ap@0.5` | Détecte bien, localise grossièrement | Augmenter `approach.pose` (12 → 16–20), vérifier la courbe PCK aux seuils serrés |
| `oks_ar` nettement supérieur à `oks_ap` | Trouve les instances mais les classe mal, ou produit des doublons | Vérifier `max_det=1` (ADR-0017) et la calibration du score |
| Sous la baseline `mean_pose` | Le modèle fait pire que « pose moyenne dans la bbox GT » | Problème sérieux : convergence, labels mal formés, ordre de keypoints. `mean_pose` utilise les bbox GT : c'est un plancher, pas un concurrent |

### `pck@α_thorax_width`
Part des keypoints à moins de `α × largeur du thorax` de leur position vraie. Référence du projet : **α = 0,25**. Lisez-la sur `pck_curve.png` plutôt qu'à un seuil unique.

| Forme de la courbe | Lecture | Action |
|---|---|---|
| Montée rapide, plateau proche de 1 | Localisation précise et fiable | Rien |
| Montée lente et régulière | Erreur diffuse sur tous les points | Plus d'époques, `imgsz` plus grand (paramètre de protocole ADR-0013 : à changer pour toutes les approches) |
| **Plateau nettement sous 1 même à α = 0,5** | Un sous-ensemble de points est systématiquement raté, pas approximatif | `heatmap_keypoints_<dataset>.png` pour identifier lesquels, puis croiser avec la couverture |
| Deux régimes (coude marqué) | Population bimodale : bonnes instances et échecs francs | Examiner les 6 pires cas dans `runs/<id>/figures/` |

### `nme_matched_only`
Erreur moyenne normalisée, **uniquement sur les instances appariées**. Optimiste par construction. Ne la citez jamais sans `kpt_coverage` à côté : une NME de 0,08 avec une couverture de 0,6 est moins bonne qu'une NME de 0,12 avec une couverture de 0,98.

### `det_ap@0.5` / `det_ap@[.5:.95]`
Qualité du cadrage. N'apparaît que si `bbox_source == predicted` (donc pas pour `mean_pose`). Avec un insecte par image, `det_ap@0.5` devrait être très élevé ; sinon le problème est en amont de la pose.

### Détail par keypoint : `pck@0.25_thorax_width`, `nme`, `kpt_conf_mean`
Publiés sous les scopes `keypoint:<dataset>:<nom>`. Ce sont eux qui alimentent les figures de diagnostic et les heatmaps par point.

---

## 3. Métriques morphométriques — la finalité du projet

### `measurement_mape_median` / `measurement_mape_worst`
Erreur relative médiane sur les 27 mesures, et pire mesure. C'est **la grandeur réellement consommée en aval**.

| Situation | Lecture | Action |
|---|---|---|
| Bon `oks_ap`, mauvaise MAPE | Les points sont globalement bien placés mais les extrémités dérivent — or les mesures sont des distances entre extrémités | Regarder les scopes `measurement:*`, et le PCK des tips (ailes, antennes, tarses) |
| MAPE correcte, `oks_ap` moyen | Les erreurs se compensent le long des polylignes | Acceptable si l'usage aval ne demande que les mesures — à documenter explicitement |
| Une mesure très au-dessus des autres | Un point précis plombe une chaîne | Identifier via `heatmap_keypoints` ; repondérer la loss ou améliorer l'annotation de ce point |

Croisez toujours avec `coverage_measurements.parquet` : une mesure calculable sur 20 % des instances a une MAPE non représentative.

### `symmetry_gap_median` / `symmetry_gap_p90`
Écart relatif gauche/droite des mesures **prédites**. Aucune vérité terrain requise — c'est votre contrôle qualité en production.

| Situation | Lecture | Action |
|---|---|---|
| Médiane basse, p90 basse | Prédictions cohérentes | Utilisable comme filtre automatique en production |
| Médiane basse, **p90 haute** | Échecs localisés sur quelques individus | Le p90 devient un détecteur d'anomalie : signaler ces images pour relecture |
| Médiane haute | Biais systématique gauche/droite | Vérifier `flip_idx` dans le `data.yaml` du run — une table de symétrie fausse avec `fliplr > 0` produit exactement ce symptôme |

Sur `symmetry_pairs.png`, chaque panneau doit montrer un nuage serré sur la diagonale. Décalé au-dessus ou au-dessous : biais directionnel. Large mais centré : bruit.

---

## 4. Métriques de coût

`latency_ms_per_instance`, `latency_ms_p95`, plus `model_params`, `train_time_s`, `peak_vram_mb` dans les manifestes.

Ce ne sont pas des annexes : un modèle qui ne tient pas en mémoire ou ne tient pas la cadence n'est pas déployable, quel que soit son OKS. **Si deux approches sont à moins d'un écart-type inter-folds l'une de l'autre sur `oks_ap`, choisissez la moins coûteuse** — la différence de performance n'est pas établie, celle de coût l'est.

La latence n'est comparable qu'entre runs sur le même matériel ; l'agrégation vous avertit sinon.

**Champs propres à certaines approches**, à vérifier dans le manifeste :

| Champ | Approche | Ce qu'il doit valoir |
|---|---|---|
| `lora_trainable_ratio` | D | quelques pourcents. À 1,0, le gel n'a pas pris et ce n'est pas du LoRA |
| `lora_targets` | D | des convolutions de fin de cou, liste non vide |
| `group_norm_layers` | E | plusieurs dizaines sur un YOLO. À 0, l'approche est sans effet |
| `n_supervised_keypoints` | F | 26 (42 moins les 16 retirés) |
| `<dataset>_train_time_s` | B | un par dataset |
| `detector_train_time_s` / `pose_train_time_s` | C | les deux modèles |

---

## 5. Lecture des figures de diagnostic

### `pck_vs_coverage.png`
Chaque point est un keypoint dans un dataset : PCK obtenu contre taux d'annotation.

| Position | Lecture | Action |
|---|---|---|
| PCK bas, couverture basse | **Ce n'est pas un problème de modèle**, mais d'annotation | Ne pas optimiser ce point. Le signaler dans le rapport et en retirer les conclusions |
| PCK bas, couverture haute | Le modèle échoue sur un point bien supervisé | Vraie cible d'amélioration |
| PCK haut, couverture basse | Statistique fragile (petit `n`) | Ne rien en conclure |

C'est la figure qui évite l'erreur d'interprétation la plus fréquente du projet.

### `pck_vs_difficulty.png`
PCK par keypoint contre la difficulté déclarée par l'expert, avec le coefficient `r` dans le titre.

| `r` | Lecture | Action |
|---|---|---|
| Nettement négatif (< −0,4) | L'échelle experte prédit bien la difficulté réelle | Les sigmas OKS (ADR-0007) sont bien calibrés |
| Proche de 0 | **Les sigmas ne reflètent pas la difficulté réelle** | La métrique primaire elle-même est mal calibrée. Envisager de recalibrer `sigma_from_difficulty.scale` ou de passer à des sigmas empiriques — décision lourde : bumper `eval.version` et rejouer |
| Positif | Les points « difficiles » sont mieux prédits que les faciles | Suspecter une inversion dans la table de difficulté |

### `keypoint_confidence_vs_error.png`
Confiance moyenne contre erreur, par keypoint, un panneau par dataset, code couleur anatomique.

| Forme du nuage | Lecture | Action |
|---|---|---|
| Décroissant | **La confiance est exploitable comme filtre** en production | Définir un seuil de rejet à partir de cette courbe |
| Plat | La confiance ne dit rien de la qualité | Ne pas s'en servir pour filtrer ; signal de mauvaise calibration |
| Points à confiance haute **et** erreur haute | Échecs silencieux — le cas le plus dangereux en production | Les identifier nommément et les surveiller |
| Confiance concentrée sur une seule valeur | Le modèle ne discrimine pas (typique d'une baseline ou d'un modèle non convergé) | Vérifier l'entraînement |

### `training_curves.png`
Diagnostic uniquement — jamais une base de comparaison entre approches.

| Motif | Lecture | Action |
|---|---|---|
| Perte de validation qui remonte | Surapprentissage | Plus d'augmentation, moins d'époques, `patience` plus courte |
| Les deux pertes stagnent haut | Sous-apprentissage ou `lr0` inadapté | Élargir l'espace de recherche sur `lr0` |
| Arrêt très précoce | Le meilleur est atteint tôt | Réduire `train.epochs` : vous payez des époques inutiles sur chaque trial d'HPO |
| Courbes très différentes entre folds | Instabilité | Suspecter un fold déséquilibré ; vérifier `data/splits/<split_id>.json` |

### `folds_*.png`
Boxplot par approche, un point par fold. Une boîte large signifie que **votre conclusion est fragile**. Si deux boîtes se chevauchent, lisez `results/paired_tests.parquet` : le test apparié est plus sensible que l'inspection visuelle, car il compare fold à fold. Un fold aberrant chez toutes les approches indique un problème de données, pas de modèle.

---

## 6. Heatmaps de comparaison

### `heatmap_<metrique>.png` (modèles × datasets)
Les palettes sont **inversées** pour les métriques où bas = meilleur (NME, MAPE, symétrie, latence). La mention figure dans le titre.

| Motif | Lecture | Action |
|---|---|---|
| Une ligne uniformément claire | Un modèle domine partout | Candidat retenu |
| Damier | Chaque approche a son terrain | Argument fort pour les modèles par dataset (approche B) |
| Une colonne sombre chez tous | Un dataset est difficile pour tout le monde | Regarder sa couverture et ses effectifs avant d'incriminer les modèles |

### `heatmap_keypoints_<dataset>.png` (keypoints × modèles)
La question qu'elle tranche : une approche gagne-t-elle **partout** ou seulement sur les points faciles ?

Un modèle qui gagne 3 points d'`oks_ap` uniquement sur des points déjà à 0,95 n'apporte rien d'exploitable. Un modèle qui gagne sur les tips d'ailes et les tarses — ceux qui portent les mesures — vaut beaucoup plus, même à `oks_ap` égal.

---

## 7. Cas particuliers par approche

**B (`yolo_per_dataset`)** — même nombre d'époques pour tous les datasets (ADR-0023). Hymenoptera (192 images) voit cinq fois moins de pas d'optimisation que Coleoptera (935). Un écart entre ordres n'est donc **pas nécessairement** une différence de difficulté.

**C (`detect_then_pose`)** — deux modèles par fold, donc coût supérieur à budget de trials égal. Les runs `pose_on_gt_boxes: true` portent `bbox_source=gt` : diagnostic isolant la pose de la détection, jamais dans le tableau bout-en-bout.

**D (`lora`)** — lire d'abord `lora_trainable_ratio`. Une performance proche de A avec 3 % de paramètres entraînés est le résultat intéressant ; une performance proche de A avec 100 % ne dit rien.

**E (`group_bn`)** — comparer surtout par dataset. Le gain attendu, s'il existe, se voit sur les ordres les plus éloignés du reste du corpus, pas sur la moyenne.

**F (`yolo_pooled_reduced`)** — **ses métriques `overall` sont mécaniquement moins bonnes** : la vérité terrain contient les 16 points retirés et l'évaluation les compte. La seule comparaison valide :

```bash
python scripts/compare_models.py --exclude-keypoints leg hindwing
```

qui ajoute une ligne `MEAN (retained)` — le chiffre à comparer à celui de A sur les mêmes points.

---

## 8. Pièges de lecture

- **Comparer des runs de `split_id`, `eval_version`, `content_hash` ou matériel différents.** L'agrégation vous avertit ; ne passez pas outre.
- **Oublier `n`.** Une cellule à `n = 12` ne se compare pas à une cellule à `n = 400`.
- **Comparer `mean_pose` au bout-en-bout.** Bbox GT : plancher, pas concurrent.
- **Lire `val` pour conclure.** Les figures portent sur `test` ; un écart important val/test signale un surapprentissage aux folds internes.
- **Conclure sur un seul fold.** Sans dispersion, il n'y a pas de conclusion.
- **Comparer des approches à budgets d'HPO différents.** Vous mesureriez le budget.
- **Inclure les trials d'HPO.** Ils sont exclus automatiquement (`role_in_protocol`), mais restent dans `master.parquet` pour l'audit.

---

## 9. Table de décision rapide

| Symptôme | Cause la plus probable | Premier geste |
|---|---|---|
| `kpt_coverage` bas ou nul | Détection défaillante ou scores sous le seuil | `det_ap@0.5`, puis distribution de `bbox_score` |
| PCK plafonne sous 1 | Sous-ensemble de points systématiquement raté | `heatmap_keypoints` puis `pck_vs_coverage` |
| MAPE haute, OKS correct | Dérive des extrémités | PCK des tips (ailes, antennes, tarses) |
| `symmetry_gap` médiane haute | `flip_idx` erroné ou `fliplr` mal configuré | Vérifier le `data.yaml` du run |
| Confiance non informative | Mauvaise calibration | Ne pas filtrer sur le score ; vérifier la convergence |
| Boîtes qui se chevauchent | Pas de différence établie | Trancher au coût |
| `r` nul sur la difficulté | Sigmas OKS mal calibrés | Décision de protocole — consigner dans `DECISIONS.md` |
| Un dataset toujours en retrait | Données, pas modèle | Couverture, effectifs, qualité d'annotation |
| `lora_trainable_ratio` = 1,0 | Le gel n'a pas pris | Ligne `Parametres entrainables` dans les logs, ADR-0028 |
