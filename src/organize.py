"""Rename raw phone photos and build data/labels.csv.  Owner: ME

The only manual step: drag each photo into its landmark's folder,
    data/raw/<landmark>/          normal photos
    data/raw/<landmark>/pano/     panorama photos (optional)
Then, from the repo root:
    python -m src.organize

What it does, per landmark folder:
- reads when each photo was taken (EXIF first, then the PXL_ file name)
- sorts new photos by that time and renames them
  <landmark>_<YYYY-MM-DD>_<NNN>.jpg   (panorama: ..._pano001.jpg)
- writes one row per photo to data/labels.csv: date, split (test if the
  date is in config.TEST_DATES, pano for panorama photos, else train)
  and lighting from the hour (sunny / sunset)

Safe to run after every shoot: renamed photos are kept, new ones get
the next numbers, and anything you typed in labels.csv (distance, angle)
stays.
"""
import re
from datetime import datetime, timedelta

import pandas as pd
from PIL import Image

from src import config

EXTS = {".jpg", ".jpeg", ".png"}
COLUMNS = ["filename", "landmark", "date", "split", "distance", "angle", "lighting"]
RENAMED = re.compile(r"^(?P<landmark>.+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<idx>(pano)?\d{3})\.(jpg|png)$")
PIXEL = re.compile(r"PXL_(\d{8})_(\d{6})")


def taken_at(path):
    """When the photo was taken, in local time."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            stamp = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        if stamp:
            return datetime.strptime(stamp.strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    m = PIXEL.search(path.name)
    if m:
        utc = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return utc + timedelta(hours=config.UTC_OFFSET_HOURS)
    print(f"[organize] no date found, using file time: {path.name}")
    return datetime.fromtimestamp(path.stat().st_mtime)


def lighting(hour):
    """sunny before config.SUNSET_FROM_HOUR, sunset from then on.
    Type another value (e.g. cloudy) into labels.csv by hand if needed."""
    return "sunny" if hour < config.SUNSET_FROM_HOUR else "sunset"


def split_for(date, is_pano):
    if is_pano:
        return "pano"
    return "test" if date in config.TEST_DATES else "train"


def _photos(folder):
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS]


def _next_index(folder, pano):
    used = [int(m["idx"].replace("pano", "")) for p in _photos(folder)
            if (m := RENAMED.match(p.name)) and m["idx"].startswith("pano") == pano]
    return max(used, default=0) + 1


def _rename_new(landmark, src, dst_folder, pano):
    """Rename photos in src that are not renamed yet; return their rows."""
    new = [p for p in _photos(src) if not RENAMED.match(p.name)]
    rows = []
    n = _next_index(dst_folder, pano)
    for t, p in sorted((taken_at(p), p) for p in new):
        date = t.date().isoformat()
        ext = ".png" if p.suffix.lower() == ".png" else ".jpg"
        idx = f"pano{n:03d}" if pano else f"{n:03d}"
        name = f"{landmark}_{date}_{idx}{ext}"
        target = dst_folder / name
        if target.exists():
            raise FileExistsError(f"{target} already exists, not overwriting")
        p.rename(target)
        rows.append({"filename": name, "landmark": landmark, "date": date,
                     "split": split_for(date, pano), "distance": "", "angle": "",
                     "lighting": lighting(t.hour)})
        n += 1
    return rows


def organize():
    """Rename new photos in data/raw/ and rewrite data/labels.csv.

    Returns:
        pandas.DataFrame, the new labels table
    """
    if config.LABELS_CSV.exists():
        old = pd.read_csv(config.LABELS_CSV, dtype=str).fillna("")
    else:
        old = pd.DataFrame(columns=COLUMNS)

    new_rows = []
    for landmark in config.LANDMARKS:
        folder = config.RAW_DIR / landmark
        if not folder.is_dir():
            print(f"[organize] missing folder: {folder}")
            continue
        new_rows += _rename_new(landmark, folder, folder, pano=False)
        new_rows += _rename_new(landmark, folder / "pano", folder, pano=True)

    # Every renamed photo on disk gets exactly one row; keep what was typed.
    labels = pd.concat([old, pd.DataFrame(new_rows, columns=COLUMNS)], ignore_index=True)
    labels = labels.drop_duplicates("filename", keep="first")
    on_disk = {}
    for landmark in config.LANDMARKS:
        for p in _photos(config.RAW_DIR / landmark):
            if m := RENAMED.match(p.name):
                on_disk[p.name] = m
    missing = sorted(set(labels["filename"]) - set(on_disk))
    if missing:
        print(f"[organize] dropping {len(missing)} rows whose photo is gone, e.g. {missing[0]}")
        labels = labels[labels["filename"].isin(on_disk)]
    for name in sorted(set(on_disk) - set(labels["filename"])):
        m = on_disk[name]
        labels.loc[len(labels)] = {"filename": name, "landmark": m["landmark"], "date": m["date"],
                                   "split": "", "distance": "", "angle": "", "lighting": ""}

    # Split always follows the date, so changing TEST_DATES re-splits.
    is_pano = labels["filename"].str.contains("_pano")
    labels["split"] = [split_for(d, p) for d, p in zip(labels["date"], is_pano)]
    labels = labels.sort_values(["landmark", "filename"])[COLUMNS].reset_index(drop=True)
    labels.to_csv(config.LABELS_CSV, index=False)

    print(f"[organize] renamed {len(new_rows)} new photos; labels.csv has {len(labels)} rows")
    print(labels.groupby(["landmark", "split"]).size().unstack(fill_value=0).to_string())
    return labels


if __name__ == "__main__":
    organize()
