from pathlib import Path

root = Path("./data/raw/hymenoptera").resolve()          # résout le symlink
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}
EXPECTED = {5 + 42 * 2, 5 + 42 * 3}                    # 89 ou 131 champs
print(root)

cibles = []
for label in sorted(root.rglob("*.txt")):
    if len(label.read_text().split()) in EXPECTED:
        continue
    images = [p for p in (label.parents[2] / "images" / label.parent.name).iterdir()
              if p.stem == label.stem and p.suffix in EXTS]
    cibles.append((label, images))

print(f"{len(cibles)} label(s) non conforme(s)\n")
for label, images in cibles:
    print(f"  {len(label.read_text().split()):4d} champs | {label}")
    for img in images:
        print(f"                  | {img}")
    if not images:
        print("                  | AUCUNE IMAGE TROUVEE")