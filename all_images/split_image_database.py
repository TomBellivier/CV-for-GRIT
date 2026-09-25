import os
import shutil
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".svg"}


def is_numeric_folder(name: str) -> bool:
    """Retourne True si le nom de dossier est entièrement numérique (ex: '01', '002')."""
    return name.isdigit()


def collect_images(folder: str) -> list[str]:
    """Retourne la liste des fichiers image dans le dossier (non récursif)."""
    images = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(filename)
    return images


def organize_images(folder: str, n: int, dry_run: bool = False) -> None:
    """
    Déplace les images du dossier dans des sous-dossiers numérotés,
    avec au maximum N images par sous-dossier.

    Args:
        folder  : Chemin vers le dossier contenant les images.
        n       : Nombre maximum d'images par sous-dossier.
        dry_run : Si True, affiche les actions sans les effectuer.
    """
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        raise ValueError(f"Le chemin spécifié n'est pas un dossier valide : {folder}")

    if n <= 0:
        raise ValueError("N doit être un entier strictement positif.")

    images = collect_images(folder)

    if not images:
        print("Aucune image trouvée dans le dossier.")
        return

    total = len(images)
    num_folders = (total + n - 1) // n  # arrondi supérieur
    padding = len(str(num_folders))     # pour numéroter avec des zéros (01, 02…)

    print(f"Dossier source  : {folder}")
    print(f"Images trouvées : {total}")
    print(f"Sous-dossiers   : {num_folders}  ({n} images max / dossier)")
    if dry_run:
        print("Mode simulation (--dry-run) : aucun fichier ne sera déplacé.\n")

    for index, filename in enumerate(images):
        folder_num = (index // n) + 1
        subfolder_name = str(folder_num).zfill(padding)
        subfolder_path = os.path.join(folder, subfolder_name)
        src = os.path.join(folder, filename)
        dst = os.path.join(subfolder_path, filename)

        print(f"  [{index + 1:>{len(str(total))}}] {filename}  →  {subfolder_name}/")

        if not dry_run:
            os.makedirs(subfolder_path, exist_ok=True)
            shutil.move(src, dst)

    if not dry_run:
        print("\nOrganisation terminée.")


def reverse_organize(folder: str, dry_run: bool = False) -> None:
    """
    Annule l'organisation : remonte toutes les images des sous-dossiers
    numérotés vers le dossier parent, puis supprime les sous-dossiers vides.

    Args:
        folder  : Chemin vers le dossier racine.
        dry_run : Si True, affiche les actions sans les effectuer.
    """
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        raise ValueError(f"Le chemin spécifié n'est pas un dossier valide : {folder}")

    # Repère les sous-dossiers numérotés
    numeric_subfolders = sorted([
        entry.name for entry in os.scandir(folder)
        if entry.is_dir() and is_numeric_folder(entry.name)
    ])

    if not numeric_subfolders:
        print("Aucun sous-dossier numéroté trouvé. Rien à annuler.")
        return

    print(f"Dossier cible   : {folder}")
    print(f"Sous-dossiers   : {', '.join(numeric_subfolders)}")
    if dry_run:
        print("Mode simulation (--dry-run) : aucun fichier ne sera déplacé.\n")

    moved = 0
    conflicts = 0

    for subfolder_name in numeric_subfolders:
        subfolder_path = os.path.join(folder, subfolder_name)
        images = collect_images(subfolder_path)

        for filename in images:
            src = os.path.join(subfolder_path, filename)
            dst = os.path.join(folder, filename)

            if os.path.exists(dst):
                print(f"Conflit ignoré : {filename} existe déjà dans le dossier racine.")
                conflicts += 1
                continue

            print(f"  {subfolder_name}/{filename}  →  {filename}")
            if not dry_run:
                shutil.move(src, dst)
            moved += 1

        # Supprime le sous-dossier s'il est vide après déplacement
        if not dry_run:
            remaining = os.listdir(subfolder_path)
            if not remaining:
                os.rmdir(subfolder_path)
                print(f"Sous-dossier supprimé : {subfolder_name}/")
            else:
                print(f"Sous-dossier non supprimé ({len(remaining)} fichier(s) restant(s)) : {subfolder_name}/")

    if not dry_run:
        print(f"\nAnnulation terminée — {moved} image(s) remontée(s){f', {conflicts} conflit(s) ignoré(s)' if conflicts else ''}.")
    else:
        print(f"\nSimulation : {moved} image(s) seraient remontées{f', {conflicts} conflit(s) ignoré(s)' if conflicts else ''}.")


def main():
    parser = argparse.ArgumentParser(
        description="Organise les images d'un dossier en sous-dossiers numérotés, ou annule l'opération."
    )
    parser.add_argument(
        "folder",
        help="Chemin vers le dossier contenant les images."
    )
    parser.add_argument(
        "n",
        type=int,
        nargs="?",         # optionnel si --reverse est utilisé
        default=None,
        help="Nombre maximum d'images par sous-dossier (non requis avec --reverse)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simule les déplacements sans modifier les fichiers."
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Annule l'organisation : remonte les images des sous-dossiers numérotés vers le dossier parent."
    )

    args = parser.parse_args()

    if args.reverse:
        reverse_organize(args.folder, dry_run=args.dry_run)
    else:
        if args.n is None:
            parser.error("L'argument 'n' est requis sauf en mode --reverse.")
        organize_images(args.folder, args.n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()