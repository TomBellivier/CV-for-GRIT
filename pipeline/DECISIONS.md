# DECISIONS.md - journal des choix methodologiques

Append-only. Une entree = une decision. Format ADR allege.
Toute decision non triviale prise en ecrivant du code atterrit ici (CONVENTIONS.md §12).

---

## ADR-0001 - Contrats de fichiers plutot qu'API Python entre modules
**Date** : initialisation - **Statut** : accepte
**Contexte** : 5+ approches sur des frameworks heterogenes (Ultralytics, PyTorch, PEFT).
**Decision** : les approches communiquent avec le pipeline par fichiers au schema fige
(annotations / splits / predictions / metriques / manifeste). L'evaluateur ne charge aucun modele.
**Consequences** : re-evaluation possible sans reentrainement ; surcout d'I/O accepte ;
toute approche externe (modele tiers, prediction manuelle) est integrable sans code.

## ADR-0002 - Toutes les coordonnees dans le repere de l'image d'origine
**Date** : initialisation - **Statut** : accepte
**Contexte** : la pipeline detection->pose travaille sur crops ; YOLO travaille en normalise.
**Decision** : aucun format normalise ni repere de crop ne quitte un module. Conversion et
retro-projection a la frontiere, verifiees par test aller-retour (1e-6).
**Consequences** : evaluation unique et comparable pour toutes les approches.

## ADR-0003 - Un seul evaluateur, metriques jamais lues depuis un framework
**Date** : initialisation - **Statut** : accepte
**Decision** : les metriques d'Ultralytics et consorts servent au monitoring uniquement.
Seul `insectpose.evaluation` produit des chiffres citables.

## ADR-0004 - Export qualitatif obligatoire a chaque run
**Date** : initialisation - **Statut** : accepte
**Contexte** : un tableau de metriques ne montre pas *comment* un modele echoue.
**Decision** : chaque run exporte 12 figures pred vs GT dont les 6 pires cas selon l'OKS
par instance, plus un index JSON tracable. Une image manquante est une ERREUR bloquante
(`eval.qualitative.allow_missing_images=false`) : un export vide masquerait un chemin casse.
**Consequences** : `image_path` doit rester valide et relatif a `paths.data`.

## ADR-0005 - Metriques ponctuelles agregees par compteurs, pas par empilement
**Date** : initialisation - **Statut** : accepte
**Contexte** : les 4 ordres n'ont pas le meme nombre de keypoints ; en perimetre poole,
empiler des tableaux (N, K) de schemas differents est impossible.
**Decision** : PCK / NME agregent numerateurs et denominateurs par bloc de schema homogene.
**Consequences** : les scopes `overall` et `dataset:*` restent valides en multi-schemas ;
c'est une consequence directe de OPEN-01 et une raison de plus de la trancher tot.

## ADR-0006 - Schema de keypoints unique aux 4 datasets : insect42_v1
**Date** : specification expert - **Statut** : accepte (ferme OPEN-01)
**Contexte** : le risque principal du projet etait une divergence anatomique entre ordres.
**Decision** : les 4 datasets partagent un seul schema de 42 points (`configs/keypoints/
insect42_v1.yaml`), avec squelette de 51 aretes et table de symetrie gauche/droite complete.
L'espace union est ce schema lui-meme : le mapping est l'identite.
**Consequences** : le modele poole n'a aucun point a masquer, les comparaisons pooled vs
par-dataset portent uniquement sur la methode. Le mecanisme d'espace union reste en place
et teste, pour absorber sans refonte une divergence future. **L'ordre des 42 points est fige
a vie** : il est encode dans tous les artefacts produits.

## ADR-0007 - Sigmas OKS derives de la difficulte de positionnement
**Date** : specification expert - **Statut** : accepte (ferme OPEN-02, volet OKS)
**Decision** : `sigma = difficulty * 0.0025`, ou `difficulty` (10 a 40) est fournie par
l'expert par point. Correspondance : 10 -> 0.025, 20 -> 0.050, 30 -> 0.075, 40 -> 0.100,
soit la plage des sigmas COCO. La regle est declaree dans le schema, pas dans le code.
**Consequences** : un point difficile a annoter est juge avec plus d'indulgence, ce qui evite
que la metrique soit dominee par le bruit d'annotation. Changer `scale` change la definition
de l'OKS : bumper `eval.version` et rejouer les runs.

## ADR-0008 - Erreur sur les mesures morphometriques comme metrique de premier plan
**Date** : specification expert - **Statut** : accepte
**Contexte** : l'aval du projet consomme 27 mesures (longueurs, largeurs), pas des keypoints.
**Decision** : `configs/measurements/insect42_v1.yaml` definit les 27 mesures et les 9 paires
symetriques. Deux metriques : `measurement_mape_median` (erreur relative pred vs GT, par mesure
et globale) et `symmetry_gap_median/p90` (ecart gauche/droite des mesures PREDITES).
**Consequences** : `symmetry_gap` ne demande aucune verite terrain : c'est un controle qualite
utilisable en production sur des images non annotees. Une mesure n'est evaluee que si tous ses
points sont annotes visibles.

## ADR-0009 - PCK normalise par la largeur du thorax
**Date** : specification expert - **Statut** : accepte (ferme OPEN-02, volet PCK)
**Decision** : un point est correct si son erreur est inferieure a `alpha x largeur du thorax`,
la largeur etant la distance `thorax-left` <-> `thorax-right`. Valeur de reference du projet :
**alpha = 0.25** ; alphas 0.125 et 0.5 donnent la courbe.
**Consequences** : normalisation anatomique, insensible au cadrage de la bbox (contrairement a
la diagonale). Si les deux points de thorax ne sont pas annotes, repli sur la diagonale de bbox,
et le **taux de repli est publie** (`pck_normalizer_fallback_rate`) : une echelle de reference
silencieusement remplacee fausserait la comparaison.

## ADR-0010 - Metrique primaire : oks_ap, changeable sans invalider les runs
**Date** : specification expert - **Statut** : accepte (ferme OPEN-03)
**Decision** : `eval.primary_metric = oks_ap`. C'est **la seule cle d'evaluation librement
surchargeable** : elle ne modifie aucun calcul, seulement l'objectif d'Optuna et le classement.
```
python -m insectpose.cli train ... eval.primary_metric=measurement_mape_median \
                                   eval.primary_direction=minimize
```
**Consequences** : toutes les metriques sont calculees a chaque run, donc changer d'objectif
n'oblige jamais a reevaluer. Le manifeste enregistre l'objectif utilise ; l'agregation alerte
si des runs tunes sur des objectifs differents sont compares.

## ADR-0011 - Une image = un specimen : group_id = image_id
**Date** : specification expert - **Statut** : accepte (ferme OPEN-04)
**Decision** : aucun dataset ne contient plusieurs images d'un meme specimen ou d'une meme
planche. Le decoupage par groupe reste actif avec `group_id = image_id`.
**Consequences** : pas de fuite. Si un futur dataset apporte plusieurs vues par specimen, il
suffit de renseigner `data.adapter_options.group_id_field` : aucun code a modifier.

## ADR-0012 - HPO nichee (nested)
**Date** : specification expert - **Statut** : accepte (ferme OPEN-05)
**Decision** : pour chaque fold externe, la recherche d'hyperparametres tourne sur des folds
**internes** construits a partir du seul train externe (`<split_id>__outer<k>`, generes par
`cli split` et versionnes comme les folds externes). Les meilleurs hyperparametres sont ensuite
appliques au fold externe entier ; le test externe n'a jamais servi a choisir un hyperparametre.
**Consequences** : cout = `n_folds x n_trials x inner_folds` entrainements, soit 5 x 40 x 3 = 600
runs par approche aux valeurs par defaut. **A calibrer avant de lancer une approche lourde** :
reduire `n_trials`, `inner_folds`, ou basculer en `tune_once` (documente comme tel). Le budget
effectif est enregistre dans chaque manifeste, et il doit rester identique entre approches.

## ADR-0013 - Resolution d'entree commune 640x640
**Date** : specification expert - **Statut** : accepte (ferme OPEN-06)
**Decision** : `protocol.image_size = [640, 640]` pour toutes les approches. Le garde-fou
`strict.enforce_common_image_size` refuse toute divergence.
**Consequences** : la comparaison porte sur la methode, pas sur la resolution. Une exploration
a autre resolution reste possible en desactivant le garde-fou, mais ses resultats ne sont pas
citables dans le rapport.

## ADR-0014 - Le groupe d'insecte est toujours connu a l'inference
**Date** : specification expert - **Statut** : accepte (ferme OPEN-07)
**Decision** : l'utilisateur final declare l'ordre d'insecte des images traitees. Les approches
conditionnees par dataset (BatchNorm par groupe, modeles par dataset) peuvent donc s'appuyer sur
`meta.dataset` sans strategie de repli.
**Consequences** : un dataset absent ou inconnu a l'inference est une **erreur explicite**, pas
un cas a deviner. A implementer comme tel dans l'approche `group_bn`.

## ADR-0015 - Aucun suivi d'experiences complementaire
**Date** : specification expert - **Statut** : accepte (ferme OPEN-08)
**Decision** : pas de MLflow ni W&B. Les manifestes JSON et `results/master.parquet` sont la
seule source de verite.

## ADR-0016 - Keypoints absents selon l'ordre d'insecte : masques, jamais imputes
**Date** : specification expert - **Statut** : accepte (ferme OPEN-10)
**Contexte** : le schema `insect42_v1` est commun aux 4 ordres, mais certains points
n'existent pas chez tous (ailes, antennes selon les groupes). Ils portent `vis = 0`.
**Decision** : ces points sont **masques** partout, jamais remplaces par une valeur :
exclus de l'OKS et du PCK, masques dans la loss (label YOLO `0 0 0`), et les mesures qui en
dependent sont declarees non calculables pour ce dataset. `cli prepare` produit un rapport
de couverture (`data/processed/coverage_*.parquet` + `coverage_summary.json`) qui distingue
trois cas : absent (<= 1 % d'annotation), rare (< 50 %, PCK par point peu informatif) et present.
**Consequences** : un point absent de TOUS les datasets est signale en avertissement — le
modele le predirait sans aucune supervision, et il faudrait envisager de le retirer du schema.
Les scopes `keypoint:<dataset>:<point>` absents des resultats ne sont pas un bug : ils
signalent une absence d'annotation, et le `n` de chaque ligne permet de le verifier.

## ADR-0017 - Une image = un insecte
**Date** : specification expert - **Statut** : accepte
**Decision** : `data.single_instance_per_image = true`. La violation est une **erreur
bloquante a la preparation** des donnees, pas un avertissement.
**Consequences** : les approches a detection utilisent `max_det = 1`. Sans ce garde-fou, une
image multi-instances ferait mentir la detection top-1 en silence. La detection reste evaluee
(`det_ap@0.5`) : cadrer un seul insecte n'est pas trivial pour autant.

## ADR-0018 - yolo_pooled : premiere approche implementee
**Date** : implementation - **Statut** : accepte
**Decision** : Approche A = un seul modele YOLO-pose, une classe `insect`, les 4 datasets
confondus, 42 keypoints. Comme le schema est commun (ADR-0006), aucune reprojection union ->
local n'est necessaire. La logique risquee (conversion de coordonnees, format des labels,
table de symetrie) est isolee dans `data/yolo_export.py` et testee par aller-retour, sans GPU.
**Consequences** :
- `data.yaml` embarque `flip_idx` : sans lui, `fliplr` apprendrait une anatomie inversee.
  L'approche refuse `fliplr > 0` si le schema n'a aucune paire de symetrie.
- Les fichiers YOLO sont un artefact **derive**, regenere par fold sous `runs/<run_id>/`,
  jamais ecrit dans `data/processed/`.
- Les noms de fichiers sont aplatis (`coleoptera__img000`) : sans cela, deux datasets ayant
  un `img000.png` se recouvriraient silencieusement.
- `conf = 0.001` a l'inference : le seuillage est une operation d'evaluation, pas d'ecriture.
- Ultralytics ne remontant pas d'intermediaire exploitable, `prunable = false` : l'elagage
  Optuna se fait au niveau du fold, pas de l'epoque.
- Dependance lourde declaree via `availability()` : le smoke test **ignore proprement**
  l'approche si `ultralytics` est absent, au lieu d'echouer.

## ADR-0019 - Materiel : CUDA disponible, torch et ultralytics en dependances de premier rang
**Date** : specification environnement - **Statut** : accepte
**Contexte** : l'environnement cible dispose d'ultralytics et d'un GPU CUDA.
**Decision** :
- `torch`, `torchvision` et `ultralytics` passent dans `dependencies` (l'extra `[yolo]` est
  conserve vide, par compatibilite) ;
- `train.device: auto` se resout en GPU 0 si CUDA est disponible, sinon `cpu` ; une valeur
  explicite (`cpu`, `"0,1"`, `mps`) est toujours respectee ;
- precision mixte (`train.amp: true`) active par defaut, **desactivee d'office en
  `mode: debug`** ou la reproductibilite prime sur la vitesse ;
- `half: true` a l'inference YOLO sur GPU, ignore sur CPU ;
- la VRAM maximale (`peak_vram_mb`), le temps d'entrainement et le nombre de parametres sont
  enregistres dans le manifeste, comme metriques de cout de premier ordre.
**Consequences** : le materiel resolu (nom du GPU, capability, VRAM totale, versions CUDA/cuDNN)
entre dans `manifest.environment.device`, et l'agregation **alerte si des runs compares
proviennent de materiels differents** : les couts (latence, VRAM) ne seraient alors pas
comparables, meme si l'OKS l'est.
Le mecanisme `availability()` reste en place : une CI sans GPU ignore proprement l'approche
au lieu d'echouer. L'integration Ultralytics est verifiee par un double (`tests/approaches/
test_yolo_pooled_integration.py`) qui teste ce que nous controlons — conversion bbox centree
vers coin haut-gauche, arguments du protocole, copie des poids — sans exiger de GPU.

---

# DECISIONS OUVERTES

Aucune decision bloquante ouverte. Les points ci-dessous sont a trancher au fil de l'eau,
sans empecher le demarrage :

## OPEN-09 - Budget d'HPO reellement soutenable
Le cout nominal de l'HPO nichee (ADR-0012) est eleve. A calibrer sur la premiere approche
lourde, puis a figer a l'identique pour toutes les autres (equite du budget, §6.3).

## OPEN-11 - Poids de depart YOLO et taille de modele
`yolo11n-pose.pt` (nano) par defaut. Le choix de la taille (n/s/m/l) doit etre le meme pour
toutes les approches a base YOLO, sinon la comparaison porte sur la capacite du reseau.
A trancher au premier entrainement reel.
