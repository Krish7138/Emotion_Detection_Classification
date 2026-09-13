# `01_data_load.ipynb` - Explanation

**Stage 1 of 5.** Load the corpus, standardise its schema, normalise the labels,
prove the data is intact, and write a clean interim copy that every later stage
reads.

| | |
|---|---|
| **Input** | `data/raw/SemEval2018-Task1-{train,dev,test}.txt` (tab separated) |
| **Output** | `data/interim/{train,dev,test}_raw.csv` |
| **Figures** | none - this stage plots nothing |
| **Next** | `02_eda.ipynb` |
| **Runtime** | a few seconds, CPU only |

---

## Why this notebook exists

The three released files are not quite usable as they arrive. Columns are
capitalised inconsistently (`ID`, `Tweet`, then eleven emotion names), and some
label cells hold the string `NONE` rather than `0`, which makes the label block a
mixed-type object array instead of a binary matrix. Every downstream notebook
would otherwise repeat the same fix-ups, and any drift between those copies would
silently change the results.

So this stage does exactly one thing: turn three raw files into one stable
schema, and refuse to continue if anything is wrong.

**It deliberately does not touch the text.** No cleaning, no lowercasing, no
emoji handling. That is `03`'s job. Keeping the text pristine here is what lets
`02_eda.ipynb` profile the corpus *as it actually arrived* — if this notebook
stripped URLs, the EDA could never tell you how many tweets contained one.

---

## Walk-through

### Cell 1 - the shared header

Every notebook in the project opens with the same block. It detects Kaggle
(`/kaggle/working` exists) and picks the project root accordingly, otherwise it
walks up one level from `notebooks/`. It then creates the five output
directories and defines the two constants the whole project shares:

```python
EMOTION_LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "love",
                  "optimism", "pessimism", "sadness", "surprise", "trust"]
TRACKS = [("no_emoji",   "text_no_emoji",   "Without Emoji"),
          ("with_emoji", "text_with_emoji", "With Emoji")]
```

`TRACKS` is unused here but is kept identical across all four notebooks so the
header can be copied verbatim - that is also what makes `kaggle_run_all.ipynb`
possible by simple concatenation.

`display_path()` prints paths relative to the project root, so machine-specific
parent folders never leak into saved notebook output.

### 1.1 — Locate the source files

`find_file()` searches the candidate roots **recursively** (`rglob`). That
recursion exists for Kaggle, which nests attached datasets under
`/kaggle/input/<dataset-slug>/...` at a depth you cannot predict. Locally it just
finds them in `data/raw/`.

If any of the three is missing the notebook raises `FileNotFoundError` with
instructions rather than failing later with a confusing pandas error.

### 1.2 - Load

Read with `sep="\t"`. Observed shapes:

```
train  (6838, 13)
dev     (886, 13)
test   (3259, 13)
```

13 columns = `ID` + `Tweet` + 11 emotions. These row counts match the original
`krish.ipynb` exactly, which is the check that `download_data.py` fetched the
same corpus the reported results came from.

### 1.3 - Standardise column names

Lowercase everything, rename `Tweet` → `text`. Then an assertion confirms all
eleven emotion columns are present in all three splits. A missing column here
would otherwise surface much later as a silent `KeyError` in the middle of a
training run.

### 1.4 — Normalise the emotion labels

```python
df[col] = df[col].replace("NONE", 0)
df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
```

`NONE` → `0`, then a numeric cast with anything uncoercible also falling to `0`.
The assertion that follows is the important part: every label cell must be
exactly `0` or `1`. If a stray value survived, the class weights computed in `03`
and every F1 in `04` would be wrong.

### 1.5 — Integrity checks

Produces this table:

| split | rows | null text | empty text | duplicate ids | duplicate text | rows with no label | labels per tweet (mean) |
|---|---|---|---|---|---|---|---|
| train | 6838 | 0 | 0 | 0 | 0 | 204 | 2.35 |
| dev | 886 | 0 | 0 | 0 | 0 | 14 | 2.44 |
| test | 3259 | 0 | 0 | 0 | 0 | 75 | 2.41 |

Two things to read off it:

* **No nulls, no empty strings, no duplicate IDs, no duplicate text.** The
  corpus is clean; asserted, not merely printed.
* **`rows with no label` is non-zero and that is correct.** 204 training tweets
  carry none of the eleven emotions — an annotator judged them to convey no
  listed emotion. They are **kept**, not dropped. Dropping them would teach the
  model that every tweet must express something, and would change the negative
  base rate the class weights are computed from.
* **Mean labels per tweet ≈ 2.4**, so this is a genuine multi-label problem, not
  a multi-class one dressed up. `02` quantifies that properly.

### 1.6 - Save the interim copy

Writes `id`, `text` and the eleven label columns to
`data/interim/{split}_raw.csv` in UTF-8. UTF-8 matters — the emoji have to
survive the round trip or the entire experiment collapses to a single track.

Resulting sizes: `train_raw.csv` 899 KB, `test_raw.csv` 427 KB, `dev_raw.csv`
116 KB.

---

## Outputs in full

| Path | Rows | What it is |
|---|---|---|
| `data/interim/train_raw.csv` | 6,838 | training split, schema-normalised, text untouched |
| `data/interim/dev_raw.csv` | 886 | development split - every reported metric comes from this |
| `data/interim/test_raw.csv` | 3,259 | held-out test split, deliberately unused until `04` §4.9 |

No figures.

---

## Things worth knowing

**The split sizes are lopsided.** Dev is 886 rows — 8.1% of the corpus. Every
headline number in the study is measured on those 886 rows, which is the main
reason `04` reports the effect as indicative rather than confirmed.

**Read the CSVs back with UTF-8.** They contain emoji. On Windows,
`pd.read_csv(path)` without an explicit encoding can pick up cp1252 and mangle
them. `03` reads them straight back with pandas' UTF-8 default, which is correct
on all three platforms the project runs on.

**Re-running is safe.** The notebook is a pure function of `data/raw/` — it
overwrites `data/interim/` and holds no state.

**If an assertion fires, stop.** Each one guards an invariant that later
notebooks assume without re-checking. Fixing it here is cheap; discovering it
after a 75-minute GPU training run is not.
