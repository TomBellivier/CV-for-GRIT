import os
import shutil
import argparse

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".svg"}


def is_numeric_folder(name: str) -> bool:
    """Return True if the folder name is entirely numeric (e.g. '01', '002')."""
    return name.isdigit()


def collect_images(folder: str) -> list[str]:
    """Return the list of image files in the folder (non recursive)."""
    images = []
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            images.append(filename)
    return images


def organize_images(folder: str, n: int, dry_run: bool = False) -> None:
    """
    Move the images of the folder into numbered subfolders, with at most N images
    per subfolder.

    Args:
        folder  : Path to the folder containing the images.
        n       : Maximum number of images per subfolder.
        dry_run : If True, print the actions without performing them.
    """
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        raise ValueError(f"The given path is not a valid folder: {folder}")

    if n <= 0:
        raise ValueError("N must be a strictly positive integer.")

    images = collect_images(folder)

    if not images:
        print("No image found in the folder.")
        return

    total = len(images)
    num_folders = (total + n - 1) // n  # rounded up
    padding = len(str(num_folders))     # to number with leading zeros (01, 02…)

    print(f"Source folder   : {folder}")
    print(f"Images found    : {total}")
    print(f"Subfolders      : {num_folders}  ({n} images max / folder)")
    if dry_run:
        print("Simulation mode (--dry-run): no file will be moved.\n")

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
        print("\nOrganisation done.")


def reverse_organize(folder: str, dry_run: bool = False) -> None:
    """
    Undo the organisation: move every image of the numbered subfolders back to the
    parent folder, then delete the empty subfolders.

    Args:
        folder  : Path to the root folder.
        dry_run : If True, print the actions without performing them.
    """
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        raise ValueError(f"The given path is not a valid folder: {folder}")

    # Find the numbered subfolders
    numeric_subfolders = sorted([
        entry.name for entry in os.scandir(folder)
        if entry.is_dir() and is_numeric_folder(entry.name)
    ])

    if not numeric_subfolders:
        print("No numbered subfolder found. Nothing to undo.")
        return

    print(f"Target folder   : {folder}")
    print(f"Subfolders      : {', '.join(numeric_subfolders)}")
    if dry_run:
        print("Simulation mode (--dry-run): no file will be moved.\n")

    moved = 0
    conflicts = 0

    for subfolder_name in numeric_subfolders:
        subfolder_path = os.path.join(folder, subfolder_name)
        images = collect_images(subfolder_path)

        for filename in images:
            src = os.path.join(subfolder_path, filename)
            dst = os.path.join(folder, filename)

            if os.path.exists(dst):
                print(f"Conflict skipped: {filename} already exists in the root folder.")
                conflicts += 1
                continue

            print(f"  {subfolder_name}/{filename}  →  {filename}")
            if not dry_run:
                shutil.move(src, dst)
            moved += 1

        # Delete the subfolder if it is empty after the move
        if not dry_run:
            remaining = os.listdir(subfolder_path)
            if not remaining:
                os.rmdir(subfolder_path)
                print(f"Subfolder deleted: {subfolder_name}/")
            else:
                print(f"Subfolder not deleted ({len(remaining)} file(s) left): {subfolder_name}/")

    if not dry_run:
        print(f"\nUndo done — {moved} image(s) moved back{f', {conflicts} conflict(s) skipped' if conflicts else ''}.")
    else:
        print(f"\nSimulation: {moved} image(s) would be moved back{f', {conflicts} conflict(s) skipped' if conflicts else ''}.")


def main():
    parser = argparse.ArgumentParser(
        description="Organise the images of a folder into numbered subfolders, or undo the operation."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing the images."
    )
    parser.add_argument(
        "n",
        type=int,
        nargs="?",         # optional if --reverse is used
        default=None,
        help="Maximum number of images per subfolder (not required with --reverse)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the moves without modifying the files."
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Undo the organisation: move the images of the numbered subfolders back to the parent folder."
    )

    args = parser.parse_args()

    if args.reverse:
        reverse_organize(args.folder, dry_run=args.dry_run)
    else:
        if args.n is None:
            parser.error("The 'n' argument is required except in --reverse mode.")
        organize_images(args.folder, args.n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
