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
**Decision** : les 4 datasets partagent un seul schema de 42 points (`configs/keypoints/ insect42_v1.yaml`), avec squelette de 51 aretes et table de symetrie gauche/droite complete.
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
- precision d'inference declaree en clair (`approach.inference_precision: fp16 | fp32`),
  traduite en `quantize` (Ultralytics >= 8.4) ou `half` (versions anterieures) ; en fp32
  aucun argument n'est passe, ce qui evite un avertissement de depreciation ;
- **inference en flux obligatoire** (`stream=True`) : sans cela Ultralytics conserve un
  objet `Results` par image, image d'origine comprise. Sur des photos de specimens en pleine
  resolution, quelques centaines d'images suffisent a saturer la RAM et le processus se fait
  tuer par l'OOM killer apres l'entrainement. Le double de test refuse desormais un appel
  non streame ;
- la VRAM maximale (`peak_vram_mb`), le temps d'entrainement et le nombre de parametres sont
  enregistres dans le manifeste, comme metriques de cout de premier ordre.
  **Consequences** : le materiel resolu (nom du GPU, capability, VRAM totale, versions CUDA/cuDNN)
  entre dans `manifest.environment.device`, et l'agregation **alerte si des runs compares
  proviennent de materiels differents** : les couts (latence, VRAM) ne seraient alors pas
  comparables, meme si l'OKS l'est.
  Le mecanisme `availability()` reste en place : une CI sans GPU ignore proprement l'approche
  au lieu d'echouer. L'integration Ultralytics est verifiee par un double (`tests/approaches/ test_yolo_pooled_integration.py`) qui teste ce que nous controlons — conversion bbox centree
  vers coin haut-gauche, arguments du protocole, copie des poids — sans exiger de GPU.

## ADR-0020 - Plafond de resolution des images source a l'export

**Date** : diagnostic terrain - **Statut** : ANNULE (remplace par ADR-0021)
**Annulation** : ce plafond avait ete introduit sur un diagnostic errone (voir plus bas).
La cause reelle de la saturation memoire etait le NOMBRE d'images passees en un appel, pas
leur resolution (ADR-0021). Le redimensionnement a donc ete retire : il ajoutait une
transformation de coordonnees a maintenir et a tester pour un benefice qui n'etait pas
celui recherche. Les images sont exportees telles quelles, par lien symbolique, et les
predictions restent nativement dans le repere de l'image d'origine.
Si le decodage devient un jour un goulot d'etranglement mesure, cette piste reste valable
et l'implementation est dans l'historique git.

**Contenu d'origine, conserve pour memoire** - **Statut initial** : accepte
**Contexte** : les datasets reels contiennent des photos de 10 a 50 MP (mediane 36 MP pour
les Lepidopteres). Une image de 36 MP decodee occupe ~108 Mo en uint8, x4 en float pour
l'augmentation, x4 en mosaique : l'entrainement saturait 31 Go de RAM et le processus etait
tue par l'OOM killer, avant meme la premiere epoque utile.
**Decision** : `protocol.export_max_side: 1280` plafonne le grand cote des images a l'export
YOLO (0 = aucun plafond). Consequences exploitees :

- les labels YOLO sont NORMALISES, donc **invariants au redimensionnement uniforme** :
  aucun label a recalculer, aucune conversion supplementaire a tester ;
- le facteur d'echelle est conserve par image dans `scales.json`, et les predictions sont
  **retro-projetees vers la resolution d'origine** avant ecriture (contrat 3) ;
- l'inference tourne aussi sur les copies reduites : decoder du 36 MP a la prediction
  couterait la meme RAM qu'a l'entrainement ;
- le decodage JPEG utilise le mode `draft` de PIL, qui decode directement a taille reduite.
  **Consequences** : `export_max_side` est un parametre de PROTOCOLE, enregistre dans le
  manifeste et **identique pour toutes les approches** — deux approches entrainees sur des
  resolutions sources differentes ne seraient pas comparables. 1280 px pour un modele qui
  travaille a 640 laisse une marge confortable ; a revoir si les keypoints fins (tarses,
  antennes) se degradent, en le documentant ici.
  En complement, le trainer Ultralytics (dataloaders, workers, buffers d'augmentation) est
  explicitement libere entre `fit` et `predict` : sans cela l'inference demarrait avec
  plusieurs Go deja occupes.

## ADR-0021 - Inference decoupee en lots

**Date** : diagnostic terrain - **Statut** : accepte
**Contexte** : `predict(source=<liste complete du fold>)` faisait croitre la RAM de ~1 Go
toutes les 5 secondes jusqu'a l'OOM, alors que le meme modele entraine groupe par groupe
passait sans probleme. `tracemalloc` a designe le chargeur d'Ultralytics
(`data/loaders.py`, `self.im0 = [...]`, `bs = len(im0)`) : **toutes les images du `source`
sont materialisees a la construction du chargeur, avant toute inference**. `stream=True`
n'y change rien, l'accumulation ayant lieu en amont.
**Decision** : l'inference parcourt les images par lots de `approach.predict_chunk_size`
(16 par defaut), avec liberation explicite entre deux lots. La conversion des `Results` est
isolee dans `_rows_from_results` pour qu'aucun d'eux ne survive a son lot.
**Consequences** : l'empreinte memoire de l'inference devient independante de la taille du
fold. Ce parametre n'affecte **aucun resultat**, seulement la memoire : il peut etre ajuste
librement, contrairement aux parametres de protocole.
**Note d'honnetete** : le diagnostic initial attribuait la saturation a la resolution des
images (ADR-0020). C'etait faux — l'utilisateur entrainait deja ces memes images sans
probleme, groupe par groupe. Le facteur discriminant etait le NOMBRE d'images par appel.
ADR-0020 reste utile (decodage et I/O plus rapides) mais n'etait pas la cause.

## ADR-0023 - Approche B : un modele YOLO-pose par dataset

**Date** : specification expert - **Statut** : accepte
**Decisions** :

- **Initialisation** : chaque modele repart des poids de base (COCO), pas du modele poule.
  A et B restent independantes ; la question posee est "un specialiste vaut-il un
  generaliste ?", et non "la specialisation apporte-t-elle quelque chose apres mutualisation ?".
- **Budget d'HPO** : total equivalent a celui de A. Consequence retenue : les hyperparametres
  sont **partages** par les N modeles, un trial les entrainant tous. Une recherche independante
  par dataset aurait quadruple le budget, et B aurait gagne par l'optimisation plutot que par
  la methode (§6.3).
- **Epoques** : identiques pour tous les datasets, quel que soit leur effectif.
  **Consequences** : avec 192 images (Hymenoptera) contre 935 (Coleoptera), le premier voit cinq
  fois moins de pas d'optimisation. **Un ecart de performance entre ordres n'est donc pas
  necessairement une difference de difficulte** : cette limite doit etre rappelee dans le rapport.
  Si elle devient genante, la variante "pas d'optimisation egalises" sera une nouvelle decision,
  pas un reglage. Techniquement, B est UNE approche encapsulant N modeles : le pipeline ne voit
  pas la difference, donc A et B sont evaluees exactement de la meme facon.

## ADR-0024 - Approche C : detection poulee puis pose sur crop

**Date** : specification expert - **Statut** : accepte
**Decisions** :

- **Detecteur** : unique et poule (une classe "insecte"), entraine DANS le run. Reutiliser les
  poids d'un run A aurait rendu C dependante de A et complique la gestion des folds.
- **Modele de pose** : YOLO-pose sur crops. Un top-down a heatmaps (HRNet, ViTPose) reste
  possible comme 6e approche si le gain de C sur A est net.
- **Resolution des crops** : 640x640, identique au protocole (ADR-0013). Une resolution plus
  faible aurait divise le cout, mais la comparaison avec A aurait alors porte en partie sur la
  resolution. Une variante 256 reste envisageable comme etude de cout, hors tableau principal.
- **Bruit de cadrage** : `jitter_scale=0.15`, `jitter_shift=0.10` a l'entrainement, aucun en
  validation ni a l'inference. **Marge de recadrage** : `padding=0.15`.
  **Consequences** : C entraine deux modeles par fold, donc son cout depasse celui de A et B a
  budget de trials egal - a mentionner dans la comparaison cout/performance. Les points tombant
  hors du crop sont masques (`vis = 0`), jamais appris comme des zeros. Le mode
  `pose_on_gt_boxes` isole la qualite de la pose de celle de la detection, mais ses resultats
  portent `bbox_source=gt` et ne figurent jamais dans le tableau bout-en-bout.

## ADR-0025 - Approche D : adaptateurs LoRA

**Date** : specification expert - **Statut** : accepte
**Decisions** : depart des poids COCO (pas du modele poule, pour garder D independante) ;
adaptateurs sur les convolutions du dernier segment du COU ; tetes entrainables, tout le
reste gele ; implementation via **peft** (`inject_adapter_in_model`, qui injecte en place
sans envelopper le modele).
**Consequences** :

- les index de blocs sont **calcules depuis la structure du modele**, jamais ecrits en dur :
  changer de taille de reseau (n/s/m/l) decalerait tout ;
- le manifeste enregistre le **nombre de parametres entrainables** et la liste des modules
  adaptes. C'est indispensable : "LoRA rang 8" ne designe rien tant qu'on ne sait pas ce qui
  reste degele a cote des adaptateurs, et deux configurations tres differentes se publient
  sous la meme etiquette ;
- les convolutions **groupees** (depthwise) sont ecartees des cibles : peft exige alors un
  rang divisible par `groups`, ce qui imposerait un rang de plusieurs dizaines pour un gain
  nul, une depthwise ne portant qu'une poignee de parametres. Le cou des architectures YOLO
  en contient ; le nombre d'exclusions est enregistre au manifeste (`lora_skipped_grouped`) ;
- si peft se revele inadapte aux `Conv2d` de cette architecture, l'alternative est un wrapper
  maison (~60 lignes) : ce serait une revision de cet ADR, pas un reglage.

## ADR-0026 - Approche E : BatchNorm par groupe d'insecte

**Date** : specification expert - **Statut** : accepte
**Decisions** : toutes les `BatchNorm2d`, statistiques ET parametres affines par groupe ;
entrainement complet depuis COCO (E reste independante de A) ; lots **mixtes**, le forward se
scindant par groupe puis recomposant dans l'ordre.
**Consequences** :

- chaque branche est initialisee depuis les statistiques du modele pre-entraine, jamais
  aleatoirement : la specialisation part d'un point commun au lieu de detruire les poids COCO ;
- le groupe vient du nom de fichier exporte (`<dataset>__<stem>`) a l'entrainement et d'une
  declaration explicite a l'inference. Un dataset inconnu leve une erreur (ADR-0014) ;
- l'inference regroupe les images par dataset. Ce n'est pas une optimisation : c'est la seule
  facon de declarer le groupe actif, l'information n'existant pas au niveau des couches ;
- l'equivalence lot mixte / N lots purs n'est PAS testee (choix assume : cout de calcul). Si un
  doute surgit sur la dynamique d'entrainement, c'est le premier test a ecrire.

## ADR-0027 - Approche F : variante sans pattes ni ailes posterieures

**Date** : demande expert - **Statut** : accepte
**Contexte** : les points de pattes et d'ailes posterieures sont les plus difficiles et les plus
mobiles. Question posee : leur retrait libere-t-il de la capacite au profit des autres ?
**Decision** : les 16 points concernes passent a `vis = 0` dans les labels d'ENTRAINEMENT et de
validation. Le schema reste `insect42_v1`, le test reste intact, et les predictions conservent
les 42 points (le contrat 3 impose le schema du dataset).
**Consequences - a rappeler dans tout rapport** : la verite terrain contient toujours ces points
et l'evaluation les compte. Les metriques `overall` de F sont donc **mecaniquement moins bonnes**
que celles de A et **ne sont pas comparables**. La seule comparaison valide porte sur les scopes
`keypoint:*` des points conserves :
`python scripts/compare_models.py --exclude-keypoints leg hindwing`, qui ajoute une ligne
`MEAN (retained)`. Confondre les deux lectures conduirait a conclure que F est mauvaise alors
qu'elle est simplement evaluee sur des points qu'elle n'a jamais appris.

## ADR-0028 - Patch du modele Ultralytics

**Date** : implementation - **Statut** : accepte
**Contexte** : D et E modifient le `nn.Module` construit par Ultralytics, qui ne prevoit ni
adaptateurs ni normalisation conditionnelle. Deux internes ont ete verifies sur la version
installee, et tous deux invalident l'approche naive :

- `on_pretrain_routine_start` se declenche AVANT la construction du modele,
  `on_pretrain_routine_end` APRES l'optimiseur et l'EMA. **Aucun callback ne convient** : un
  patch pose la serait soit perdu, soit absent de l'optimiseur ;
- la boucle de `freeze` remet `requires_grad=True` sur tout parametre gele dont le nom ne
  correspond pas a `args.freeze`. **Un simple `requires_grad=False` serait annule en silence.**
  **Decision** : passer un trainer derive (`train(trainer=...)`, supporte) ; appliquer le patch
  dans `get_model` et le gel dans `_build_train_pipeline`, juste avant l'optimiseur. Toute cette
  dependance aux internes est isolee dans `training/patching.py`.
  **Complements verifies a l'execution (revision)** : trois pieges supplementaires, tous
  constates sur un entrainement reel :
- le **validateur** possede son propre `preprocess` et ne passe pas par celui du trainer. Sans
  relais, la normalisation par groupe recevait les indices du dernier lot d'ENTRAINEMENT face a
  un lot de validation de taille differente. Le contexte est donc renseigne des deux cotes ;
- l'**evaluation finale** d'Ultralytics recharge le meilleur checkpoint et le FUSIONNE. Sur un
  modele patche, la fusion echoue (enveloppes LoRA) ou serait fausse (N jeux de statistiques
  ecrases en un). Elle est desactivee : ses metriques ne servent qu'au monitoring (§7.1) ;
- une classe construite DANS une fonction n'est pas picklable, et Ultralytics serialise le
  modele a chaque sauvegarde de checkpoint. `GroupBatchNorm2d` est donc publiee au niveau du
  module (identite corrigee + `__getattr__` de module) tout en gardant l'import de torch differe ;
- les enveloppes LoRA de peft n'exposent pas les attributs d'une convolution (`out_channels`).
  Les adaptateurs sont donc **fusionnes dans les poids de base** avant sauvegarde : le
  checkpoint redevient un YOLO standard, rechargeable et fusionnable, sans dependance a peft.
  Pour la normalisation par groupe, la fusion est simplement neutralisee a l'inference.

**Consequences** : un compte de parametres entrainables est journalise et enregistre au manifeste
a chaque run, et un compte nul leve une erreur. Si une future version d'Ultralytics change cet
ordre, ce chiffre le signale au lieu de laisser passer un entrainement silencieusement faux. La
logique de decision (quels modules, quels parametres, quel groupe) est ecrite en fonctions pures,
testables sans torch : c'est la partie qui casse en silence.

---

---

Aucune decision bloquante ouverte. Les points ci-dessous sont a trancher au fil de l'eau,
sans empecher le demarrage :

## OPEN-09 - Budget d'HPO reellement soutenable

Le cout nominal de l'HPO nichee (ADR-0012) est eleve. A calibrer sur la premiere approche
lourde, puis a figer a l'identique pour toutes les autres (equite du budget, §6.3).

## OPEN-11 - Poids de depart YOLO et taille de modele

`yolo11n-pose.pt` (nano) par defaut. Le choix de la taille (n/s/m/l) doit etre le meme pour
toutes les approches a base YOLO, sinon la comparaison porte sur la capacite du reseau.
A trancher au premier entrainement ree

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
**Decision** : les 4 datasets partagent un seul schema de 42 points (`configs/keypoints/ insect42_v1.yaml`), avec squelette de 51 aretes et table de symetrie gauche/droite complete.
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
- precision d'inference declaree en clair (`approach.inference_precision: fp16 | fp32`),
  traduite en `quantize` (Ultralytics >= 8.4) ou `half` (versions anterieures) ; en fp32
  aucun argument n'est passe, ce qui evite un avertissement de depreciation ;
- **inference en flux obligatoire** (`stream=True`) : sans cela Ultralytics conserve un
  objet `Results` par image, image d'origine comprise. Sur des photos de specimens en pleine
  resolution, quelques centaines d'images suffisent a saturer la RAM et le processus se fait
  tuer par l'OOM killer apres l'entrainement. Le double de test refuse desormais un appel
  non streame ;
- la VRAM maximale (`peak_vram_mb`), le temps d'entrainement et le nombre de parametres sont
  enregistres dans le manifeste, comme metriques de cout de premier ordre.
  **Consequences** : le materiel resolu (nom du GPU, capability, VRAM totale, versions CUDA/cuDNN)
  entre dans `manifest.environment.device`, et l'agregation **alerte si des runs compares
  proviennent de materiels differents** : les couts (latence, VRAM) ne seraient alors pas
  comparables, meme si l'OKS l'est.
  Le mecanisme `availability()` reste en place : une CI sans GPU ignore proprement l'approche
  au lieu d'echouer. L'integration Ultralytics est verifiee par un double (`tests/approaches/ test_yolo_pooled_integration.py`) qui teste ce que nous controlons — conversion bbox centree
  vers coin haut-gauche, arguments du protocole, copie des poids — sans exiger de GPU.

## ADR-0020 - Plafond de resolution des images source a l'export

**Date** : diagnostic terrain - **Statut** : ANNULE (remplace par ADR-0021)
**Annulation** : ce plafond avait ete introduit sur un diagnostic errone (voir plus bas).
La cause reelle de la saturation memoire etait le NOMBRE d'images passees en un appel, pas
leur resolution (ADR-0021). Le redimensionnement a donc ete retire : il ajoutait une
transformation de coordonnees a maintenir et a tester pour un benefice qui n'etait pas
celui recherche. Les images sont exportees telles quelles, par lien symbolique, et les
predictions restent nativement dans le repere de l'image d'origine.
Si le decodage devient un jour un goulot d'etranglement mesure, cette piste reste valable
et l'implementation est dans l'historique git.

**Contenu d'origine, conserve pour memoire** - **Statut initial** : accepte
**Contexte** : les datasets reels contiennent des photos de 10 a 50 MP (mediane 36 MP pour
les Lepidopteres). Une image de 36 MP decodee occupe ~108 Mo en uint8, x4 en float pour
l'augmentation, x4 en mosaique : l'entrainement saturait 31 Go de RAM et le processus etait
tue par l'OOM killer, avant meme la premiere epoque utile.
**Decision** : `protocol.export_max_side: 1280` plafonne le grand cote des images a l'export
YOLO (0 = aucun plafond). Consequences exploitees :

- les labels YOLO sont NORMALISES, donc **invariants au redimensionnement uniforme** :
  aucun label a recalculer, aucune conversion supplementaire a tester ;
- le facteur d'echelle est conserve par image dans `scales.json`, et les predictions sont
  **retro-projetees vers la resolution d'origine** avant ecriture (contrat 3) ;
- l'inference tourne aussi sur les copies reduites : decoder du 36 MP a la prediction
  couterait la meme RAM qu'a l'entrainement ;
- le decodage JPEG utilise le mode `draft` de PIL, qui decode directement a taille reduite.
  **Consequences** : `export_max_side` est un parametre de PROTOCOLE, enregistre dans le
  manifeste et **identique pour toutes les approches** — deux approches entrainees sur des
  resolutions sources differentes ne seraient pas comparables. 1280 px pour un modele qui
  travaille a 640 laisse une marge confortable ; a revoir si les keypoints fins (tarses,
  antennes) se degradent, en le documentant ici.
  En complement, le trainer Ultralytics (dataloaders, workers, buffers d'augmentation) est
  explicitement libere entre `fit` et `predict` : sans cela l'inference demarrait avec
  plusieurs Go deja occupes.

## ADR-0021 - Inference decoupee en lots

**Date** : diagnostic terrain - **Statut** : accepte
**Contexte** : `predict(source=<liste complete du fold>)` faisait croitre la RAM de ~1 Go
toutes les 5 secondes jusqu'a l'OOM, alors que le meme modele entraine groupe par groupe
passait sans probleme. `tracemalloc` a designe le chargeur d'Ultralytics
(`data/loaders.py`, `self.im0 = [...]`, `bs = len(im0)`) : **toutes les images du `source`
sont materialisees a la construction du chargeur, avant toute inference**. `stream=True`
n'y change rien, l'accumulation ayant lieu en amont.
**Decision** : l'inference parcourt les images par lots de `approach.predict_chunk_size`
(16 par defaut), avec liberation explicite entre deux lots. La conversion des `Results` est
isolee dans `_rows_from_results` pour qu'aucun d'eux ne survive a son lot.
**Consequences** : l'empreinte memoire de l'inference devient independante de la taille du
fold. Ce parametre n'affecte **aucun resultat**, seulement la memoire : il peut etre ajuste
librement, contrairement aux parametres de protocole.
**Note d'honnetete** : le diagnostic initial attribuait la saturation a la resolution des
images (ADR-0020). C'etait faux — l'utilisateur entrainait deja ces memes images sans
probleme, groupe par groupe. Le facteur discriminant etait le NOMBRE d'images par appel.
ADR-0020 reste utile (decodage et I/O plus rapides) mais n'etait pas la cause.

## ADR-0023 - Approche B : un modele YOLO-pose par dataset

**Date** : specification expert - **Statut** : accepte
**Decisions** :

- **Initialisation** : chaque modele repart des poids de base (COCO), pas du modele poule.
  A et B restent independantes ; la question posee est "un specialiste vaut-il un
  generaliste ?", et non "la specialisation apporte-t-elle quelque chose apres mutualisation ?".
- **Budget d'HPO** : total equivalent a celui de A. Consequence retenue : les hyperparametres
  sont **partages** par les N modeles, un trial les entrainant tous. Une recherche independante
  par dataset aurait quadruple le budget, et B aurait gagne par l'optimisation plutot que par
  la methode (§6.3).
- **Epoques** : identiques pour tous les datasets, quel que soit leur effectif.
  **Consequences** : avec 192 images (Hymenoptera) contre 935 (Coleoptera), le premier voit cinq
  fois moins de pas d'optimisation. **Un ecart de performance entre ordres n'est donc pas
  necessairement une difference de difficulte** : cette limite doit etre rappelee dans le rapport.
  Si elle devient genante, la variante "pas d'optimisation egalises" sera une nouvelle decision,
  pas un reglage. Techniquement, B est UNE approche encapsulant N modeles : le pipeline ne voit
  pas la difference, donc A et B sont evaluees exactement de la meme facon.

## ADR-0024 - Approche C : detection poulee puis pose sur crop

**Date** : specification expert - **Statut** : accepte
**Decisions** :

- **Detecteur** : unique et poule (une classe "insecte"), entraine DANS le run. Reutiliser les
  poids d'un run A aurait rendu C dependante de A et complique la gestion des folds.
- **Modele de pose** : YOLO-pose sur crops. Un top-down a heatmaps (HRNet, ViTPose) reste
  possible comme 6e approche si le gain de C sur A est net.
- **Resolution des crops** : 640x640, identique au protocole (ADR-0013). Une resolution plus
  faible aurait divise le cout, mais la comparaison avec A aurait alors porte en partie sur la
  resolution. Une variante 256 reste envisageable comme etude de cout, hors tableau principal.
- **Bruit de cadrage** : `jitter_scale=0.15`, `jitter_shift=0.10` a l'entrainement, aucun en
  validation ni a l'inference. **Marge de recadrage** : `padding=0.15`.
  **Consequences** : C entraine deux modeles par fold, donc son cout depasse celui de A et B a
  budget de trials egal - a mentionner dans la comparaison cout/performance. Les points tombant
  hors du crop sont masques (`vis = 0`), jamais appris comme des zeros. Le mode
  `pose_on_gt_boxes` isole la qualite de la pose de celle de la detection, mais ses resultats
  portent `bbox_source=gt` et ne figurent jamais dans le tableau bout-en-bout.

## ADR-0025 - Approche D : adaptateurs LoRA

**Date** : specification expert - **Statut** : accepte
**Decisions** : depart des poids COCO (pas du modele poule, pour garder D independante) ;
adaptateurs sur les convolutions du dernier segment du COU ; tetes entrainables, tout le
reste gele ; implementation via **peft** (`inject_adapter_in_model`, qui injecte en place
sans envelopper le modele).
**Consequences** :

- les index de blocs sont **calcules depuis la structure du modele**, jamais ecrits en dur :
  changer de taille de reseau (n/s/m/l) decalerait tout ;
- le manifeste enregistre le **nombre de parametres entrainables** et la liste des modules
  adaptes. C'est indispensable : "LoRA rang 8" ne designe rien tant qu'on ne sait pas ce qui
  reste degele a cote des adaptateurs, et deux configurations tres differentes se publient
  sous la meme etiquette ;
- si peft se revele inadapte aux `Conv2d` de cette architecture, l'alternative est un wrapper
  maison (~60 lignes) : ce serait une revision de cet ADR, pas un reglage.

## ADR-0026 - Approche E : BatchNorm par groupe d'insecte

**Date** : specification expert - **Statut** : accepte
**Decisions** : toutes les `BatchNorm2d`, statistiques ET parametres affines par groupe ;
entrainement complet depuis COCO (E reste independante de A) ; lots **mixtes**, le forward se
scindant par groupe puis recomposant dans l'ordre.
**Consequences** :

- chaque branche est initialisee depuis les statistiques du modele pre-entraine, jamais
  aleatoirement : la specialisation part d'un point commun au lieu de detruire les poids COCO ;
- le groupe vient du nom de fichier exporte (`<dataset>__<stem>`) a l'entrainement et d'une
  declaration explicite a l'inference. Un dataset inconnu leve une erreur (ADR-0014) ;
- l'inference regroupe les images par dataset. Ce n'est pas une optimisation : c'est la seule
  facon de declarer le groupe actif, l'information n'existant pas au niveau des couches ;
- l'equivalence lot mixte / N lots purs n'est PAS testee (choix assume : cout de calcul). Si un
  doute surgit sur la dynamique d'entrainement, c'est le premier test a ecrire.

## ADR-0027 - Approche F : variante sans pattes ni ailes posterieures

**Date** : demande expert - **Statut** : accepte
**Contexte** : les points de pattes et d'ailes posterieures sont les plus difficiles et les plus
mobiles. Question posee : leur retrait libere-t-il de la capacite au profit des autres ?
**Decision** : les 16 points concernes passent a `vis = 0` dans les labels d'ENTRAINEMENT et de
validation. Le schema reste `insect42_v1`, le test reste intact, et les predictions conservent
les 42 points (le contrat 3 impose le schema du dataset).
**Consequences - a rappeler dans tout rapport** : la verite terrain contient toujours ces points
et l'evaluation les compte. Les metriques `overall` de F sont donc **mecaniquement moins bonnes**
que celles de A et **ne sont pas comparables**. La seule comparaison valide porte sur les scopes
`keypoint:*` des points conserves :
`python scripts/compare_models.py --exclude-keypoints leg hindwing`, qui ajoute une ligne
`MEAN (retained)`. Confondre les deux lectures conduirait a conclure que F est mauvaise alors
qu'elle est simplement evaluee sur des points qu'elle n'a jamais appris.

## ADR-0028 - Patch du modele Ultralytics

**Date** : implementation - **Statut** : accepte
**Contexte** : D et E modifient le `nn.Module` construit par Ultralytics, qui ne prevoit ni
adaptateurs ni normalisation conditionnelle. Deux internes ont ete verifies sur la version
installee, et tous deux invalident l'approche naive :

- `on_pretrain_routine_start` se declenche AVANT la construction du modele,
  `on_pretrain_routine_end` APRES l'optimiseur et l'EMA. **Aucun callback ne convient** : un
  patch pose la serait soit perdu, soit absent de l'optimiseur ;
- la boucle de `freeze` remet `requires_grad=True` sur tout parametre gele dont le nom ne
  correspond pas a `args.freeze`. **Un simple `requires_grad=False` serait annule en silence.**
  **Decision** : passer un trainer derive (`train(trainer=...)`, supporte) ; appliquer le patch
  dans `get_model` et le gel dans `_build_train_pipeline`, juste avant l'optimiseur. Toute cette
  dependance aux internes est isolee dans `training/patching.py`.
  **Complements verifies a l'execution (revision)** : trois pieges supplementaires, tous
  constates sur un entrainement reel :
- le **validateur** possede son propre `preprocess` et ne passe pas par celui du trainer. Sans
  relais, la normalisation par groupe recevait les indices du dernier lot d'ENTRAINEMENT face a
  un lot de validation de taille differente. Le contexte est donc renseigne des deux cotes ;
- l'**evaluation finale** d'Ultralytics recharge le meilleur checkpoint et le FUSIONNE. Sur un
  modele patche, la fusion echoue (enveloppes LoRA) ou serait fausse (N jeux de statistiques
  ecrases en un). Elle est desactivee : ses metriques ne servent qu'au monitoring (§7.1) ;
- une classe construite DANS une fonction n'est pas picklable, et Ultralytics serialise le
  modele a chaque sauvegarde de checkpoint. `GroupBatchNorm2d` est donc publiee au niveau du
  module (identite corrigee + `__getattr__` de module) tout en gardant l'import de torch differe ;
- les enveloppes LoRA de peft n'exposent pas les attributs d'une convolution (`out_channels`).
  Les adaptateurs sont donc **fusionnes dans les poids de base** avant sauvegarde : le
  checkpoint redevient un YOLO standard, rechargeable et fusionnable, sans dependance a peft.
  Pour la normalisation par groupe, la fusion est simplement neutralisee a l'inference.

**Consequences** : un compte de parametres entrainables est journalise et enregistre au manifeste
a chaque run, et un compte nul leve une erreur. Si une future version d'Ultralytics change cet
ordre, ce chiffre le signale au lieu de laisser passer un entrainement silencieusement faux. La
logique de decision (quels modules, quels parametres, quel groupe) est ecrite en fonctions pures,
testables sans torch : c'est la partie qui casse en silence.

---

---

Aucune decision bloquante ouverte. Les points ci-dessous sont a trancher au fil de l'eau,
sans empecher le demarrage :

## OPEN-09 - Budget d'HPO reellement soutenable

Le cout nominal de l'HPO nichee (ADR-0012) est eleve. A calibrer sur la premiere approche
lourde, puis a figer a l'identique pour toutes les autres (equite du budget, §6.3).

## OPEN-11 - Poids de depart YOLO et taille de modele

`yolo11n-pose.pt` (nano) par defaut. Le choix de la taille (n/s/m/l) doit etre le meme pour
toutes les approches a base YOLO, sinon la comparaison porte sur la capacite du reseau.
A trancher au premier entrainement reel.
