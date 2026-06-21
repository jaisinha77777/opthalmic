"""
Fetch a real, labelled glaucoma fundus dataset and build a uniform manifest.

The training script (scripts/train_fundus.py) consumes a single file:
    data/fundus/labels.csv   with columns:  filepath, glaucoma[0/1], cdr[optional]

This script gets real images onto disk and writes that manifest. Because the public
datasets are distributed under different licences and layouts, two acquisition paths
are supported:

  1) Kaggle CLI (recommended -- needs a free Kaggle account + API token at
     ~/.kaggle/kaggle.json, see https://www.kaggle.com/docs/api):

        python scripts/fetch_fundus_dataset.py --source kaggle \
            --dataset arnavjain1/glaucoma-datasets

  2) A manually downloaded archive (REFUGE, G1020, RIM-ONE DL, ACRIMA, ...):

        python scripts/fetch_fundus_dataset.py --zip C:/Downloads/G1020.zip

After images are on disk under data/fundus/, the manifest is built automatically by
scanning for either (a) a dataset CSV with glaucoma/CDR columns, or (b) class
subfolders named like 'glaucoma'/'normal'.

Recommended datasets (binary glaucoma; some include vertical CDR ground truth):
  - REFUGE   1200 imgs, disc/cup segmentation + labels
  - G1020    1020 imgs, binary labels + vertical CDR
  - RIM-ONE DL ~485 imgs, normal/glaucoma (credential-free GitHub mirror exists)
  - ACRIMA   705 imgs, binary labels
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FUNDUS_DIR = ROOT / "data" / "fundus"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GLAUCOMA_HINTS = ("glaucoma", "glauc", "_g_", "pos", "abnormal")
NORMAL_HINTS = ("normal", "healthy", "_n_", "neg", "control")


def _kaggle_download(dataset: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {dataset} -> {out}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out), "--unzip"],
            check=True,
        )
    except FileNotFoundError:
        sys.exit("kaggle CLI not found. Install with `pip install kaggle` and place "
                 "your API token at ~/.kaggle/kaggle.json (see script header).")
    except subprocess.CalledProcessError as e:
        sys.exit(f"kaggle download failed ({e}). Check the dataset slug and your token.")


def _unzip(zip_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    print(f"[zip] extracting {zip_path} -> {out}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)


def _label_from_path(p: Path) -> int | None:
    """Infer 0/1 glaucoma label from any folder/file name hint along the path."""
    s = "/".join(part.lower() for part in p.parts)
    # ACRIMA convention: glaucoma files contain "_g_", every other ACRIMA image
    # is a healthy control (e.g. Im001_g_ACRIMA.jpg vs Im001_ACRIMA.jpg).
    if "_acrima" in s or "acrima" in s:
        return 1 if "_g_" in s else 0
    if any(h in s for h in GLAUCOMA_HINTS):
        return 1
    if any(h in s for h in NORMAL_HINTS):
        return 0
    return None


def _find_existing_csv(root: Path) -> Path | None:
    """Look for a dataset-provided CSV that already carries labels."""
    for csv_path in root.rglob("*.csv"):
        if csv_path.name == "labels.csv":
            continue
        try:
            header = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0].lower()
        except Exception:
            continue
        if ("glaucoma" in header or "label" in header or "binary" in header) and \
           ("image" in header or "id" in header or "file" in header or "name" in header):
            return csv_path
    return None


def _build_manifest_from_csv(csv_path: Path, root: Path) -> int:
    """Translate a dataset CSV into our uniform labels.csv. Best-effort column matching."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    img_col = pick("imageid", "image", "filename", "file", "name", "id")
    lab_col = pick("glaucoma", "label", "binary", "binarylabels", "class")
    cdr_col = pick("cdr", "vertical_cdr", "vcdr", "cup_disc_ratio")
    if img_col is None or lab_col is None:
        return 0

    # Index every image on disk by basename for path resolution.
    by_name = {p.name.lower(): p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT}

    rows = []
    for _, r in df.iterrows():
        name = str(r[img_col]).strip()
        key = name.lower()
        path = by_name.get(key) or by_name.get(key + ".jpg") or by_name.get(key + ".png")
        if path is None:
            stem = Path(key).stem
            path = next((p for n, p in by_name.items() if Path(n).stem == stem), None)
        if path is None:
            continue
        try:
            label = int(float(r[lab_col]) > 0.5)
        except Exception:
            label = _label_from_path(path)
            if label is None:
                continue
        cdr = ""
        if cdr_col is not None:
            try:
                cdr = round(float(r[cdr_col]), 3)
            except Exception:
                cdr = ""
        rows.append((str(path.relative_to(ROOT)), label, cdr))
    return _write_manifest(rows)


def _build_manifest_from_folders(root: Path) -> int:
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXT:
            continue
        label = _label_from_path(p)
        if label is None:
            continue
        rows.append((str(p.relative_to(ROOT)), label, ""))
    return _write_manifest(rows)


def _write_manifest(rows: list) -> int:
    if not rows:
        return 0
    out = FUNDUS_DIR / "labels.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "glaucoma", "cdr"])
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description="Fetch + organize a glaucoma fundus dataset.")
    ap.add_argument("--source", choices=["kaggle"], help="download via the Kaggle CLI")
    ap.add_argument("--dataset", help="kaggle dataset slug, e.g. arnavjain1/glaucoma-datasets")
    ap.add_argument("--zip", dest="zip_path", help="path to a manually downloaded archive")
    ap.add_argument("--organize-only", action="store_true",
                    help="skip download; just (re)build labels.csv from data/fundus/")
    args = ap.parse_args()

    FUNDUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        _unzip(Path(args.zip_path), FUNDUS_DIR)
    elif args.source == "kaggle":
        if not args.dataset:
            sys.exit("--dataset is required with --source kaggle")
        _kaggle_download(args.dataset, FUNDUS_DIR)
    elif not args.organize_only:
        sys.exit("Provide --zip <path>, --source kaggle --dataset <slug>, or --organize-only.")

    print("[manifest] scanning images and labels ...")
    csv_path = _find_existing_csv(FUNDUS_DIR)
    n = _build_manifest_from_csv(csv_path, FUNDUS_DIR) if csv_path else 0
    if n == 0:
        n = _build_manifest_from_folders(FUNDUS_DIR)

    if n == 0:
        sys.exit("Could not label any images. Ensure data/fundus/ contains a dataset CSV "
                 "with glaucoma labels, or class subfolders named 'glaucoma'/'normal'.")
    print(f"[done] wrote {FUNDUS_DIR / 'labels.csv'} with {n} labelled images.")
    print("Next: python scripts/train_fundus.py")


if __name__ == "__main__":
    main()
