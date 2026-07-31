# Approche 3 — Pipeline top-down : détection puis pose sur crop

## Principe

Deux modèles en série :

1. **Détecteur** (4 classes = 4 groupes) sur l'image entière. Localiser un gros
   insecte sur une macro est une tâche facile, qui converge vite.
2. **Modèle de pose** sur chaque boîte recadrée et élargie d'une marge.

Le gain vient de la **normalisation d'échelle et de position** : le modèle de
pose voit toujours un insecte centré, à taille comparable. Il n'a plus à
apprendre l'invariance d'échelle, seulement la morphologie. Sur 42 keypoints
dont une majorité de petites articulations de pattes, cela vaut typiquement
plusieurs points de PCK.

Bonus : le détecteur prédit aussi le taxon, sans coût supplémentaire.

## Fichiers

| Fichier | Rôle |
|---|---|
| `prepare_two_stage_dataset.py` | Construit le dataset de détection et le dataset de crops |
| `train_two_stage.py` | Entraîne les deux étages |
| `eval_two_stage.py` | Évalue le pipeline complet sur les images **originales** |

## Utilisation

```bash
python prepare_two_stage_dataset.py \
    --data-config groups.yaml \
    --out-dir two_stage_data \
    --margin 1.25 --link

python train_two_stage.py \
    --det-data  two_stage_data/det/det.yaml \
    --pose-data two_stage_data/pose_crops/pose.yaml \
    --det-model yolo26n.pt --pose-model yolo26n-pose.pt \
    --det-epochs 100 --pose-epochs 150 --pose-imgsz 256 \
    --margin 1.25 \
    --out-dir two_stage_weights --runs-dir runs_two_stage

python eval_two_stage.py \
    --manifest two_stage_weights/two_stage_manifest.json \
    --data-config groups.yaml \
    --out-dir pose_results
```

## La marge doit être la même partout

`--margin` définit la géométrie du crop. Elle est utilisée à trois endroits :
préparation des données, entraînement (implicitement, via les crops) et
inférence. Le manifeste la propage automatiquement de `train_two_stage.py` vers
`eval_two_stage.py`, mais si vous préparez les données avec une marge et
évaluez avec une autre, le modèle de pose voit des cadrages qu'il n'a jamais
rencontrés et les résultats s'effondrent sans message d'erreur.

## `--pose-imgsz`

`prepare_two_stage_dataset.py` affiche en fin d'exécution la distribution des
tailles de crop et **suggère une valeur** : le multiple de 32 couvrant environ
90 % des crops sans sur-échantillonnage. Suivez-la. Un `pose-imgsz` de 256 sur
des crops de 150 px gaspille du calcul ; à 640 c'est du pur gaspillage.

## Évaluation — ce qui est mesuré

L'évaluation tourne sur les **images de validation originales de chaque groupe**
et compare aux **labels originaux**, pas aux crops. C'est délibéré : noter le
modèle de pose sur ses propres crops le comparerait à quelque chose que les
autres approches n'ont jamais vu, et masquerait silencieusement chaque échec de
détection. Ici, une détection manquée coûte les keypoints correspondants — ce
qui est le comportement honnête.

Conséquences pratiques :

- `--pose-conf` vaut 0.01 par défaut. Chaque crop contient exactement un
  insecte, et l'instance la mieux notée est retenue ; filtrer sur la confiance
  ne ferait que perdre des keypoints.
- `--filter-by-class` est **désactivé** par défaut : le pipeline est noté sans
  qu'on lui souffle le taxon. L'activer mesure le cas où le groupe est connu
  d'avance.
- `training_time_sec` porte le **total des deux étages**, identique pour les
  quatre lignes : les deux modèles sont entraînés une seule fois pour tous les
  groupes. Les temps séparés figurent dans la feuille `metadata`.
- `map_source` vaut toujours `custom` (voir le README principal, section 2).

## Keypoints perdus au recadrage

Un keypoint repoussé hors du crop par la marge est marqué **invisible** plutôt
que rabattu sur un bord — le rabattre apprendrait au modèle une position fausse.
Le script indique combien de keypoints sont concernés. Si ce nombre est élevé,
augmentez `--margin`.
