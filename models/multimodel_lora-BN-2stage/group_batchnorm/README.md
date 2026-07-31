# Approche 2 — BatchNorm spécifiques par groupe

## Principe

Toutes les convolutions sont partagées. Seule la normalisation est dupliquée :
paramètres affines (`gamma`, `beta`) et statistiques courantes (`running_mean`,
`running_var`), un jeu par groupe. Sur votre architecture cela représente
**moins de 1 % des paramètres par groupe** — le script affiche le chiffre exact.

L'intuition : une large part de l'écart entre deux domaines visuels se ramène à
un décalage et un changement d'échelle par canal des distributions d'activation.
Réestimer la normalisation absorbe cet écart sans toucher aux filtres.

C'est la seule méthode de spécialisation que 200 images d'hyménoptères peuvent
réellement soutenir : quelques milliers de paramètres libres, pas quelques
millions.

## Fichiers

| Fichier | Rôle |
|---|---|
| `group_bn.py` | `GroupBatchNorm2d`, conversion, gel, extraction/chargement, vérification du partage |
| `train_group_bn.py` | Réestime les BN sur chaque groupe, base gelée |
| `eval_group_bn.py` | Assemble les 4 banques dans un modèle unique et évalue |

## Utilisation

```bash
python train_group_bn.py \
    --base-weights runs_base/base/weights/best.pt \
    --data-config groups.yaml \
    --epochs 40 --lr0 0.002 \
    --out-dir gbn_weights --runs-dir runs_gbn \
    --verify-shared

python eval_group_bn.py \
    --manifest gbn_weights/gbn_manifest.json \
    --out-dir pose_results
```

## Points à comprendre

**Peu d'époques suffisent.** Il n'y a que quelques milliers de paramètres libres
par groupe ; 40 époques est un plafond confortable, pas un minimum. Si la courbe
plafonne à l'époque 15, arrêtez-vous là.

**Les statistiques courantes ne sont pas des paramètres.** `running_mean` et
`running_var` sont des *buffers* : ils se mettent à jour à chaque passe avant en
mode entraînement, indépendamment de `requires_grad`. C'est voulu — leur
réestimation sur les images du groupe constitue la moitié de l'effet recherché.
Vous obtiendriez déjà un gain en ne faisant qu'une passe avant sur les données
du groupe, sans aucune rétropropagation (c'est l'idée originale d'AdaBN).

**`--verify-shared` n'est pas décoratif.** Il recharge les quatre checkpoints et
compare bit à bit tous les tenseurs hors-BN. Si la vérification échoue, le gel
n'a pas pris effet et vous avez en réalité entraîné quatre modèles complets
indépendants — l'expérience ne mesure alors plus ce qu'elle prétend mesurer. Le
résultat est reporté dans la ligne `shared_weights_verified` de la feuille
`metadata`.

**`--also-train`** permet de laisser une partie supplémentaire entraînable, par
exemple `--also-train model.23.cv4,model.23.one2one_cv4` pour dégeler la branche
keypoints. C'est un hybride entre cette approche et une tête séparée ; à
n'essayer qu'après avoir mesuré la version pure.

## Variante encore plus légère

Si même 40 époques par groupe sont trop coûteuses, `--epochs 1 --lr0 0.0`
réalise une **AdaBN pure** : les gradients sont nuls, seules les statistiques
courantes se réestiment sur les images du groupe. C'est presque gratuit et
souvent déjà rentable.
