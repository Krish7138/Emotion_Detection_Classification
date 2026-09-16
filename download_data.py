# -*- coding: utf-8 -*-
"""Fetch the SemEval-2018 Task 1 (E-c, English) corpus into data/raw/.

The original notebook read the corpus from a Kaggle dataset path that no longer
resolves, so this downloads the official release directly from the shared task
authors and writes the three files under the names the pipeline expects.

    python download_data.py

Produces:
    data/raw/SemEval2018-Task1-train.txt   6,838 rows
    data/raw/SemEval2018-Task1-dev.txt       886 rows
    data/raw/SemEval2018-Task1-test.txt    3,259 rows

No extra packages are needed - only the standard library.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

URL = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"
BASE = "SemEval2018-Task1-all-data/English/E-c/"

# source file inside the zip  ->  name the notebooks expect,  expected row count
FILES = {
    "2018-E-c-En-train.txt":     ("SemEval2018-Task1-train.txt", 6838),
    "2018-E-c-En-dev.txt":       ("SemEval2018-Task1-dev.txt",    886),
    "2018-E-c-En-test-gold.txt": ("SemEval2018-Task1-test.txt",  3259),
}

# The host rejects requests without a browser user agent.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "*/*",
    "Referer": "https://saifmohammad.com/",
}


def main():
    print("SemEval-2018 Task 1 - Affect in Tweets (E-c, English)")
    print("source:", URL)
    print("\nDownloading (~6 MB) ...")

    try:
        req = urllib.request.Request(URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=180) as resp:
            blob = resp.read()
    except Exception as exc:
        print(f"\nDownload failed: {exc}\n")
        print("Download the zip manually and extract the three files from")
        print(f"  {BASE}")
        print("into data/raw/, renaming them as listed in the README. Mirrors:")
        print("  * https://competitions.codalab.org/competitions/17751")
        print("  * https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1")
        return 1

    print(f"received {len(blob) / 1e6:.1f} MB")

    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        print("\nThe download was not a valid zip - the host may have returned an "
              "error page. Try again, or download it manually in a browser.")
        return 1

    RAW.mkdir(parents=True, exist_ok=True)
    names = set(zf.namelist())
    ok = True

    print()
    for src, (dest_name, expected) in FILES.items():
        member = BASE + src
        if member not in names:
            print(f"  {src:28s} NOT FOUND in the archive")
            ok = False
            continue

        text = zf.read(member).decode("utf-8")
        rows = sum(1 for line in text.split("\n") if line.strip()) - 1  # minus header

        dest = RAW / dest_name
        dest.write_text(text, encoding="utf-8")

        flag = "OK" if rows == expected else f"WARNING expected {expected}"
        print(f"  {dest_name:32s} {rows:>5,} rows   {flag}")
        if rows != expected:
            ok = False

    print()
    if ok:
        print("Saved to", RAW)
        print("\nNext: run notebooks/01_data_load.ipynb")
        return 0
    print("Finished with warnings - check the row counts above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
