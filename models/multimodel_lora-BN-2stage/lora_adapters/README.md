# Approche 1 — Adaptateurs LoRA par groupe

## Principe

Chaque convolution `W` du modèle de base est **gelée**, et une correction de
rang faible est apprise en parallèle :

```
y = conv(x, W) + (alpha / r) · conv(conv(x, A), B)
```

`A` a la forme `[r, C_in, k, k]`, `B` la forme `[C_out, r, 1, 1]` et est
**initialisée à zéro** — le modèle adapté démarre donc numériquement identique
au modèle de base. On apprend `r · (C_in·k² + C_out)` paramètres au lieu de
`C_out · C_in · k²`.

Une banque d'adaptateurs par groupe. Changer de groupe est une affectation de
chaîne de caractères : les poids partagés ne sont jamais rechargés.

## Fichiers

| Fichier | Rôle |
|---|---|
| `lora.py` | `LoRAConv2d`, injection, gel, extraction/chargement des banques |
| `train_lora.py` | Entraîne une banque par groupe sur le modèle de base gelé |
| `eval_lora.py` | Installe les 4 banques sur un seul backbone et évalue |

## Utilisation

```bash
python train_lora.py \
    --base-weights runs_base/base/weights/best.pt \
    --data-config groups.yaml \
    --rank 8 --targets neck_head \
    --epochs 80 --batch 16 --lr0 0.001 \
    --out-dir lora_weights --runs-dir runs_lora

python eval_lora.py \
    --manifest lora_weights/lora_manifest.json \
    --out-dir pose_results
```

## Paramètres qui comptent

**`--targets`** — où injecter. `neck_head` (couches 11 à 23) est le défaut :
c'est là que se joue la spécialisation morphologique, et le backbone (0–10)
encode des primitives visuelles communes aux quatre groupes. Autres valeurs :
`all`, `neck`, `head`, ou une liste explicite `13,16,19,22,23`.

**`--rank`** — 8 est un bon départ. Avec 160 images d'hyménoptères, monter
au-delà de 16 sera contre-productif. Le script affiche le nombre exact de
paramètres par groupe au lancement : surveillez-le.

**`--lr0`** — 0.001, soit dix fois moins que le fine-tuning complet. LoRA
supporte en général un LR **plus élevé** que le fine-tuning complet parce que
peu de paramètres bougent ; si la convergence est lente, montez à 0.003.

Les convolutions **depthwise** (`groups != 1`) sont ignorées par défaut : une
factorisation de rang faible d'un noyau depthwise n'a guère de sens, et la
matrice `A` dense qu'elle imposerait coûterait plus cher que la couche adaptée.
`--include-grouped` pour forcer.

## Détail d'implémentation à connaître

`YOLO.train()` **reconstruit le modèle** à partir de son yaml avant
l'entraînement, ce qui détruirait toute injection faite en amont. Deux hooks
contournent le problème :

- `get_model` — injecte les adaptateurs juste après la construction ;
- `build_optimizer` — gèle la base juste avant la création de l'optimiseur,
  c'est-à-dire **après** la boucle d'Ultralytics qui réactive `requires_grad`
  sur tout ce qui n'est pas dans sa liste `freeze`.

Un callback `on_train_start` sert de filet de sécurité si une version future
d'Ultralytics déplace ces appels. Si `train_lora.py` se termine sur
`no adapter tensors recovered`, c'est que ce mécanisme a cédé : vérifiez la
version d'Ultralytics installée.

Les banques sont sauvegardées **séparément du modèle** (`lora_<groupe>.pt`,
quelques centaines de Ko) plutôt que dans le checkpoint picklé, ce qui évite
toute dépendance au dépicklage de classes personnalisées.
