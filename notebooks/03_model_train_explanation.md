# `03_model_train.ipynb` - Explanation

**Stage 3 of 5.** Build the two parallel corpora the whole study rests on, then
train both architectures on each of them. This is the notebook that produces
everything the Streamlit app serves.

| | |
|---|---|
| **Input** | `data/interim/{train,dev,test}_raw.csv` |
| **Output** | `data/processed/*_cleaned.csv`, `models/*.keras` (×4), `results/thresholds.json`, `results/best_configs.json`, `results/hyperparameter_search.csv`, `results/pipeline_fixture.json` |
| **Figures** | none - this stage plots nothing |
| **Next** | `04_evaluation.ipynb` |
| **Runtime** | ~75 minutes on a Kaggle GPU, most of it the BiLSTM. Hours on CPU. |

> **The attention model is not pretrained BERT.** It is a Transformer encoder
> trained from scratch. The checkpoints are named `bert_*.keras` for continuity
> with the original `krish.ipynb`, but nothing pretrained is loaded. This is also
> why its absolute scores sit below published state of the art, and it must be
> described accurately in the write-up.

---

## The experimental design

One preprocessing pipeline is built and **forked at exactly one operation**:

* **Track A - `text_no_emoji`** - every emoji is deleted (`emoji.replace_emoji`).
* **Track B - `text_with_emoji`** - every emoji is replaced by its Unicode
  description (`demoji.replace_with_desc`), so `😂` becomes `face with tears of joy`.

Everything upstream and downstream of that fork is identical: same cleaning, same
normalisation, same tokeniser, same lemmatiser, same seed, same architectures,
same hyper-parameter grid, same epoch budget, same early stopping, same threshold
sweep. The **only** systematic difference between the two corpora is whether
emoji content survives as text.

That gives four comparable conditions:

| | Track A (no emoji) | Track B (with emoji) |
|---|---|---|
| **BiLSTM** | `lstm_no_emoji.keras` | `lstm_with_emoji.keras` |
| **Transformer** | `bert_no_emoji.keras` | `bert_with_emoji.keras` |

and a clean difference measure, `ΔF1 = F1(with emoji) - F1(without emoji)`, which
`04` computes.

---

## Walk-through

### Cells 1-3 - setup

`emoji` and `demoji` are imported with a `!pip install` fallback for Kaggle.
NLTK data (`punkt`, `wordnet`, `omw-1.4`) is downloaded quietly.

Global settings:

```python
SEED = 42                # numpy + tf.keras.utils.set_random_seed
MAX_TOKENS      = 20000  # vocabulary cap, per track
SEQUENCE_LENGTH = 128    # far above the corpus tail (mean 16 words)
SEARCH_EPOCHS   = 5      # phase 1
FINAL_EPOCHS    = 10     # phase 2
```

> **On the NLTK downloads.** In the saved Kaggle run these three downloads failed
> with a DNS error (`Temporary failure in name resolution`) because the notebook
> had no internet. `WordNetLemmatizer` then falls back to returning tokens
> unchanged for most inputs. The run completed and the models are valid, but this
> means lemmatisation was largely inert in the shipped artefacts. If you re-run
> locally with internet, lemmatisation *will* fire - which changes the vocabulary
> slightly and means the retrained models will not be bit-identical to the shipped
> ones. Both tracks are affected identically, so the comparison remains fair
> either way.

### 3.1 - The preprocessing pipeline

Five stages, of which stage 2 is the fork:

| Stage | Function | What it does |
|---|---|---|
| 1 | `clean_text` | strip URLs, strip `@mentions`, drop the `#` but keep the word, collapse whitespace |
| 2a | `handle_emoji_no` | **Track A** - delete every emoji |
| 2b | `handle_emoji_yes` | **Track B** - replace each emoji with its description |
| 3 | `normalize_text` | lowercase, replace non-alphanumerics with spaces |
| 4 | `tokenize_text` | NLTK `TweetTokenizer`, dropping bare punctuation |
| 5 | `lemmatize_tokens` | slang expansion (`u` → `you`, `cant` → `cannot`, …) then WordNet lemmatisation |

`build_text(text, emoji_mode)` chains all five and returns a space-joined string.

> These functions are the **single source of truth** for what the models were
> fitted on. `app/preprocessing.py` is a deliberate copy for serving. Change one
> without the other and every prediction the UI makes becomes invalid — the model
> would receive text in a distribution it never saw during training.

### Worked example - where the tracks diverge

Run on a constructed sentence, so no corpus text is reproduced:

| stage | Track A | Track B |
|---|---|---|
| 0. input | `I'm so happy today 😂😭 u cant believe it!! #joy @friend http://x.com` | *(same)* |
| 1. noise removal | `I'm so happy today 😂😭 u cant believe it!! joy` | *(same)* |
| 2. **FORK** | `I'm so happy today  u cant believe it!! joy` | `I'm so happy today  face with tears of joy  loudly crying face  u cant believe it!! joy` |
| 3. normalisation | `i m so happy today u cant believe it joy` | `i m so happy today face with tears of joy loudly crying face u cant believe it joy` |
| 5. slang + lemma | `i m so happy today you cannot believe it joy` | `i m so happy today face with tear of joy loudly cry face you cannot believe it joy` |

Track B recovered six extra tokens from two emoji: `cry`, `face`, `loudly`, `of`,
`tear`, `with`. On a tweet averaging 16 words, that is a substantial addition -
and note that half of them (`face`, `of`, `with`) carry no emotional content and
simply dilute the sequence. That trade-off is exactly what the experiment
measures.

### 3.2 - Apply to every split, then check the fork actually worked

Both tracks are computed for all three splits and stored side by side in the same
DataFrame. Two failure modes are then checked explicitly:

| split | mean len A | mean len B | empty A | empty B | rows differing | % differing |
|---|---|---|---|---|---|---|
| train | 82.4 | 86.4 | 0 | 0 | 825 | 12.1 |
| dev | 81.5 | 89.6 | 0 | 0 | 230 | 26.0 |
| test | 81.4 | 89.6 | 0 | 0 | 853 | 26.2 |

* **`empty A` / `empty B` are zero.** No row was reduced to an empty string.
  (Emoji-only tweets would show up here - there are none in this corpus.)
* **`% differing` tracks `02`'s emoji density**, as it must. It runs slightly
  higher (12.1% vs 11.3% on train) because this pipeline uses the `emoji` library
  while `02` used a narrower regex - see the note in `02`'s explanation. If this
  column were ever `0`, the fork would have silently done nothing and the entire
  study would be measuring noise.
* **Track B is consistently longer** (86.4 vs 82.4 characters on train), which is
  the descriptions being written in.

The processed corpora are then written to `data/processed/{split}_cleaned.csv`
with both text columns plus the labels, and `results/pipeline_fixture.json`
records three input/output triplets so the serving copy in `app/` can be checked
for drift.

### 3.3 - Class imbalance

`02` established a 7.3:1 spread. Each label is weighted by its own
negative-to-positive ratio:

```python
pos_weight = (len(train) - pos_counts) / pos_counts
```

| emotion | positives | pos_weight |
|---|---|---|
| trust | 357 | 18.15 |
| surprise | 361 | 17.94 |
| love | 700 | 8.77 |
| pessimism | 795 | 7.60 |
| anticipation | 978 | 5.99 |
| fear | 1,242 | 4.51 |
| optimism | 1,984 | 2.45 |
| sadness | 2,008 | 2.41 |
| joy | 2,477 | 1.76 |
| anger | 2,544 | 1.69 |
| disgust | 2,602 | 1.63 |

`weighted_bce` applies these through
`tf.nn.weighted_cross_entropy_with_logits`. A missed `trust` positive costs 18×
what a false alarm does, which is what makes the rare emotions predictable at
all - and also why every condition ends up recalling far more than it precisely
predicts (`04` §4.7).

### 3.4 - Data pipeline and model builders

`make_vectorizer` adapts a `TextVectorization` layer **per track**, with
`standardize=None` because normalisation already happened in the pipeline. This
matters: Track B's vocabulary contains description words (`tears`, `loudly`,
`weary`) that Track A's simply does not, so the two tracks genuinely see
different vocabularies - that is the manipulation working, not a leak.

The vectoriser is baked **into the saved model** (the input is `dtype=tf.string`),
which is why the app can feed raw preprocessed strings straight to
`model.predict` with no separate tokeniser artefact to keep in sync.

**BiLSTM arm** - embedding → BiLSTM(`lstm_units`, sequences) → BiLSTM(`lstm_units//2`)
→ dropout → dense(11).

**Transformer arm** - embedding → multi-head self-attention → residual +
layer-norm → feed-forward → residual + layer-norm → global average pool →
dropout → dense(11). One encoder block, trained from scratch. **No positional
encoding is added**, so with global average pooling this arm is close to
order-insensitive - a known simplification, and part of why it underperforms the
BiLSTM here.

Both heads emit **raw logits**, not probabilities. Every consumer must apply
`tf.nn.sigmoid` itself - `04` does, and so does `app/inference.py`.

### 3.5 - Prediction and threshold selection

```python
def best_threshold(y_true, y_prob):
    for t in np.arange(0.1, 0.6, 0.05):   # 0.10 … 0.55
        ...keep the t maximising Micro-F1
```

The threshold is swept on the development split and the Micro-F1 maximiser is
adopted - applied identically in all four conditions, so it cannot favour one
track over the other.

> ⚠️ **The sweep is truncated, and it binds.** `np.arange(0.1, 0.6, 0.05)` stops
> at 0.55, and **all four conditions selected exactly 0.55** - the top of the
> range. That is the signature of an optimum lying at or beyond the boundary, so
> the true Micro-F1 maximiser may well be above 0.55 and was never tested.
> Widening the sweep (say to 0.95) is the single cheapest improvement available:
> it needs no retraining, only a re-scoring pass. Note also that this contradicts
> the claim in the project README that the tuned thresholds "land well below 0.5"
> - the saved `results/thresholds.json` is `0.55` for all four. The comparison
> between tracks is unaffected, since the same value applies to both.

### 3.6 - Hyper-parameter grids

Three configurations per architecture, spanning capacity and learning rate in
opposite directions:

| | cfg1 (baseline) | cfg2 (more capacity, lower LR) | cfg3 (less capacity, higher LR) |
|---|---|---|---|
| **BiLSTM** | embed 128, units 128, lr 1e-3, drop 0.2, batch 64 | embed 256, units 256, lr 5e-4, drop 0.3, batch 32 | embed 128, units 64, lr 2e-3, drop 0.1, batch 64 |
| **Transformer** | embed 128, heads 4, key 32, ff 256, lr 1e-3, drop 0.2, batch 64 | embed 256, heads 8, key 32, ff 512, lr 5e-4, drop 0.3, batch 32 | embed 128, heads 2, key 64, ff 256, lr 2e-3, drop 0.1, batch 64 |

The same budget is spent on all four conditions, which is what keeps the
comparison fair.

### 3.7 - Two-phase training

**Phase 1 (`search`)** trains each of the three configurations for up to 5 epochs,
tunes a threshold, scores dev Micro-F1, and keeps the winner.

**Phase 2** retrains that winner from scratch for up to 10 epochs.

Early stopping with `restore_best_weights=True` is active in both phases
(patience 2, then 3), so the model that gets saved is always the one at minimum
validation loss, never the one at the final epoch.

The search log, written to `results/hyperparameter_search.csv`:

| config | micro_f1 | macro_f1 | label_acc | model | track |
|---|---|---|---|---|---|
| cfg1 | 52.31 | 46.39 | 73.15 | LSTM | Without Emoji |
| cfg2 | 54.49 | 48.88 | 75.71 | LSTM | Without Emoji |
| **cfg3** | **54.66** | 49.06 | 75.64 | LSTM | Without Emoji |
| cfg1 | 51.42 | 45.61 | 73.06 | LSTM | With Emoji |
| cfg2 | 53.97 | 48.03 | 74.75 | LSTM | With Emoji |
| **cfg3** | **54.17** | 48.60 | 74.54 | LSTM | With Emoji |
| **cfg1** | **54.81** | 47.77 | 76.16 | BERT | Without Emoji |
| cfg2 | 49.38 | 44.07 | 73.26 | BERT | Without Emoji |
| cfg3 | 47.43 | 45.92 | 66.27 | BERT | Without Emoji |
| cfg1 | 47.85 | 44.42 | 70.32 | BERT | With Emoji |
| cfg2 | 47.59 | 43.73 | 70.51 | BERT | With Emoji |
| **cfg3** | **50.25** | 46.70 | 72.88 | BERT | With Emoji |

**The spread across configurations is 7.38 points** (47.43 → 54.81). Hold on to
that number: `04` §4.8 compares it against the largest emoji effect measured
(3.71 points) and concludes the aggregate effect is smaller than
configuration-driven variation.

Note also that the Transformer's two tracks picked **different** winning
configurations (cfg1 without emoji, cfg3 with). That is the protocol working as
designed - each condition gets its own best setup - but it does mean the
Transformer's ΔF1 confounds "emoji helped" with "cfg3 happened to suit this
track", which is worth stating when the number is quoted.

Final Phase-2 results:

```
BiLSTM      | Without Emoji  -> threshold 0.55 | dev Micro-F1 53.63%
BiLSTM      | With Emoji     -> threshold 0.55 | dev Micro-F1 55.65%
Transformer | Without Emoji  -> threshold 0.55 | dev Micro-F1 47.29%
Transformer | With Emoji     -> threshold 0.55 | dev Micro-F1 51.00%
```

### 3.8 - Persist everything

The original `krish.ipynb` computed the tuned thresholds but **never wrote them to
disk**, so they vanished when the kernel ended - and without them the served
model falls back to 0.5 and stops predicting rare emotions almost entirely. That
is the bug this section fixes.

---

## Outputs in full

| Path | Size / contents | Read by |
|---|---|---|
| `data/processed/train_cleaned.csv` | 6,838 rows × (text, text_no_emoji, text_with_emoji, 11 labels) | `04` |
| `data/processed/dev_cleaned.csv` | 886 rows, same schema | `04` - every reported metric |
| `data/processed/test_cleaned.csv` | 3,259 rows, same schema | `04` §4.9 only, opt-in |
| `models/lstm_no_emoji.keras` | 19.8 MB | `04`, app |
| `models/lstm_with_emoji.keras` | 19.9 MB | `04`, app |
| `models/bert_no_emoji.keras` | 19.7 MB | `04`, app |
| `models/bert_with_emoji.keras` | 19.8 MB | `04`, app |
| `results/thresholds.json` | 4 keys, all `0.55` | `04`, app |
| `results/best_configs.json` | winning config per condition | reference, app |
| `results/hyperparameter_search.csv` | 12 rows - the full search log | `04` §4.8 |
| `results/pipeline_fixture.json` | 3 input/output triplets | drift check for `app/preprocessing.py` |

No figures.