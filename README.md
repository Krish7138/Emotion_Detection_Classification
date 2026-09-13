# Does the Emoji Matter?

Multi-label emotion classification of tweets, and a controlled test of whether
**removing** emoji or **converting them to text** changes what the model predicts.

MSc Computing Research Project. The original single notebook (`krish.ipynb`) is
split into a five-stage pipeline.

---

## The five stages

| # | File | What it does |
|---|---|---|
| 1 | `notebooks/01_data_load.ipynb` | Load the corpus, standardise columns, normalise labels, integrity checks |
| 2 | `notebooks/02_eda.ipynb` | Profile the raw corpus: label imbalance, co-occurrence, text noise, emoji density |
| 3 | `notebooks/03_model_train.ipynb` | Preprocessing (the two tracks), both architectures, tuning, **saves models + thresholds** |
| 4 | `notebooks/04_evaluation.ipynb` | Score all four conditions, ΔF1, confusion matrices, ROC - writes tables and figures |
| 5 | `app/streamlit_app.py` | The UI: a text box, the trained model behind it |

Plus `download_data.py`, which fetches the corpus.

Each stage reads what the previous one wrote, so **run them in order**. Stage 3 is
the only one the UI strictly needs.

**Notebook documentation.** [`notebooks/README.md`](notebooks/README.md) lists what
every notebook writes - outputs and figures - alongside the headline results, and
each notebook has a companion section-by-section walk-through:
[01](notebooks/01_data_load_explanation.md) ·
[02](notebooks/02_eda_explanation.md) ·
[03](notebooks/03_model_train_explanation.md) ·
[04](notebooks/04_evaluation_explanation.md).

## Quick start

```bash
pip install -r requirements.txt
```

Fetch the corpus (downloads ~6 MB, standard library only):

```bash
python download_data.py
```

That writes the three files the pipeline expects:

```
data/raw/SemEval2018-Task1-train.txt   6,838 rows
data/raw/SemEval2018-Task1-dev.txt       886 rows
data/raw/SemEval2018-Task1-test.txt    3,259 rows
```

Run notebooks 01 → 02 → 03 → 04, then start the UI:

```bash
streamlit run app/streamlit_app.py
```

That is the whole sequence. The app needs **no copying, no extra scripts and no
manual threshold step** - notebook 03 writes its artefacts directly into the
folders the app already reads.

## How the data flows

```
data/raw/          the three SemEval .txt files          (you provide)
   |  01
data/interim/      {train,dev,test}_raw.csv
   |  02  ->  figures/, results/eda_*.csv
   |  03
data/processed/    {train,dev,test}_cleaned.csv          <- both tracks
models/            4 x .keras                            <- read by the UI
results/           thresholds.json, best_configs.json    <- read by the UI
   |  04
results/           results_table.csv, emoji_impact.csv, per_emotion_impact.csv,
                   per_emotion_auc.csv, mean_auc.csv, confusion_counts.csv
figures/           overall_performance, emoji_impact, per_emotion_impact,
                   precision_recall, 4 x confusion_*, 4 x roc_*
```

## The experimental design in one paragraph

One preprocessing pipeline is built and forked at exactly one operation.
**Track A** deletes every emoji; **Track B** replaces each with its Unicode
description (`😂` → `face with tears of joy`). Everything upstream and downstream
is identical, so the only systematic difference between the two corpora is whether
emoji content survives as text. Each track then trains both architectures under
an identical protocol, giving four comparable conditions and a clean difference
measure, `ΔF1 = F1(with emoji) - F1(without emoji)`.

## Where the data comes from

`download_data.py` pulls the official release published by the shared task
authors:

<https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip>

and extracts the three English E-c files, renaming them to the names above:

| In the zip | Written as |
|---|---|
| `English/E-c/2018-E-c-En-train.txt` | `SemEval2018-Task1-train.txt` |
| `English/E-c/2018-E-c-En-dev.txt` | `SemEval2018-Task1-dev.txt` |
| `English/E-c/2018-E-c-En-test-gold.txt` | `SemEval2018-Task1-test.txt` |

Row counts (6,838 / 886 / 3,259) match the original notebook exactly, as does the
first training row - this is the same corpus the reported results were produced
from.

If that host is unreachable, the same data is mirrored at:

* <https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2018_task_1> (config `subtask5.english`)
* <https://competitions.codalab.org/competitions/17751> (official task page, needs registration)

> The Kaggle dataset path hard-coded in the original `krish.ipynb`
> (`priyankshekhda/semeval2018-task1-*`) no longer resolves, which is why the
> download script exists.

## Running on Kaggle

`kaggle_run_all.ipynb` is all four stages merged into one notebook, built by
concatenating the four files above so the code cannot drift from them.

Upload it, then in the right-hand panel set **Accelerator = GPU** and attach two
datasets via **+ Add Input**:

| Dataset | Why |
|---|---|
| `semeval2018-task1-ec-english` | the corpus |
| `emoji-demoji-wheels` | `emoji` + `demoji` wheels |

The wheels dataset exists because Kaggle grants notebook **internet only to
phone-verified accounts**. Without it `pip install demoji` fails with a DNS
error, and without internet the corpus cannot be downloaded either - hence
shipping both as datasets. If your account *is* verified you can switch Internet
on instead and skip both.

Then **Save Version → Save & Run All**. A full GPU run takes roughly 75 minutes,
most of it the BiLSTM. Everything is written to `/kaggle/working` and bundled as
`emoji_project_outputs.zip` in the **Output** tab; unzip it into the project root
and the folders line up exactly.

> If a Kaggle session ends, the output is **not** lost provided the notebook was
> run with *Save Version*. It stays under that version's Output tab.

## Things worth knowing

**The attention model is not pretrained BERT.** It is a Transformer encoder
trained from scratch. The checkpoints are named `bert_*.keras` for continuity with
the original notebook, but the report and the UI both label it accurately. This
is also why its absolute scores sit below published state of the art.

**The decision thresholds are not optional.** The models are trained with a
positive-weighted loss, so the threshold that maximises Micro-F1 is not 0.5 and is
swept per condition instead. Together with the loss weighting, that is what makes
rare emotions (*trust*, *surprise*, *pessimism*) predictable at all. In the shipped
run all four conditions selected **0.55** - the top of the swept range
(`0.10 … 0.55`), which suggests the true optimum lies above it and was never
tested; widening the sweep needs no retraining. Notebook 03 saves them to
`results/thresholds.json`; the
original notebook computed them but never wrote them out, so they were lost when
the kernel ended. `app/recompute_thresholds.py` remains as a fallback for models
that arrive without them.

**Preprocessing must not drift.** `app/preprocessing.py` is a deliberate copy of
the pipeline defined in notebook 03, because the models were fitted on that exact
output. Notebook 03 also writes `results/pipeline_fixture.json` - a few
input/output pairs the serving copy can be checked against.

**Reported metrics are development-split figures.** The test split is prepared but
deliberately left unconsumed so it stays a genuine holdout. Section 4.9 of notebook
04 runs it behind an explicit opt-in flag.

**Emoji density bounds everything.** Only a minority of tweets contain emoji, so
the two tracks are byte-identical on most rows. A small aggregate effect is
expected even when the effect on emoji-bearing tweets is large - which is why
results are also reported per emotion.


**Version drift is real, not theoretical.** The shipped models were saved by
Kaggle's Keras **3.13.2**. Loading them on Keras 3.11 fails with

```
Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}
```

because the newer Keras writes a field the older one rejects. `requirements.txt`
therefore pins `keras>=3.13,<3.14`. If you retrain locally on an older Keras the
models will load fine there, but will not load on a newer one.
