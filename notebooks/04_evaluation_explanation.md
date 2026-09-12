# `04_evaluation.ipynb` — Explanation

**Stage 4 of 5.** Score the four trained conditions and answer the two research
questions. This is where the study's findings are actually produced.

| | |
|---|---|
| **Input** | `models/*.keras` (×4), `results/thresholds.json`, `results/hyperparameter_search.csv`, `data/processed/dev_cleaned.csv` |
| **Output** | `results/results_table.csv`, `emoji_impact.csv`, `per_emotion_impact.csv`, `per_emotion_auc.csv`, `mean_auc.csv`, `confusion_counts.csv` (+ `test_results_table.csv` if §4.9 is enabled) |
| **Figures** | `overall_performance.png`, `emoji_impact.png`, `per_emotion_impact.png`, `precision_recall.png`, 4 × `confusion_*.png`, 4 × `roc_*.png` — 12 in total |
| **Next** | the Streamlit UI in `app/` |
| **Runtime** | a couple of minutes — inference only, no training |

**Research questions**

* **RQ1** — to what extent does emoji handling affect performance?
  Measured as `ΔF1 = F1(with emoji) − F1(without emoji)`.
* **RQ2** — does that effect depend on the architecture and on the specific emotion?

Everything is computed on the **development split** (886 rows). The test split
was prepared but deliberately left unconsumed so it stays a genuine holdout;
§4.9 runs it behind an explicit opt-in flag, currently `False`.

---

## Walk-through

### 4.1 — Score all four conditions

Models are loaded with `compile=False` (the custom training loss is irrelevant to
inference) and the sigmoid is applied here, because the models emit **raw
logits**. Predictions are then thresholded with the per-condition value from
`results/thresholds.json` — all four are `0.55`.

Each condition is scored on **its own** text column: the no-emoji models see
`text_no_emoji`, the with-emoji models see `text_with_emoji`. Crossing those
would measure a train/serve mismatch instead of the emoji effect.

### 4.2 — Overall metrics → `results/results_table.csv`, `figures/overall_performance.png`

| Model | Track | Micro F1 | Macro F1 | Label Acc | Subset Acc | Precision | Recall | Thr |
|---|---|---|---|---|---|---|---|---|
| BiLSTM | Without Emoji | 53.63 | 47.86 | 74.02 | 5.53 | 44.38 | 67.75 | 0.55 |
| **BiLSTM** | **With Emoji** | **55.65** | **48.86** | 75.85 | 9.48 | 46.93 | 68.35 | 0.55 |
| Transformer | Without Emoji | 47.29 | 44.36 | 70.19 | 1.81 | 38.90 | 60.30 | 0.55 |
| Transformer | With Emoji | 51.00 | 47.30 | 72.63 | 3.39 | 42.29 | 64.23 | 0.55 |

How to read this:

* **Plain accuracy is misleading and is included only to show why.** Roughly
  three quarters of all label decisions are negative, so a model predicting
  nothing at all would score around 78% label accuracy while being useless.
  Micro-F1 is the primary measure.
* **Subset accuracy is brutal** (1.81–9.48%) — it demands all eleven labels be
  simultaneously correct on a tweet averaging 2.35 positives. It is reported for
  completeness, not as a headline.
* **The mean Micro–Macro gap is 4.80 points.** Micro weights every decision
  equally and so is dominated by frequent emotions; Macro averages per-label F1
  and so gives `trust` the same voice as `disgust`. The gap reads out how unevenly
  performance is spread — and 4.8 points says it is concentrated in the frequent
  emotions.
* **The BiLSTM beats the Transformer in both tracks.** That is expected here: the
  attention arm is a single encoder block trained from scratch with no positional
  encoding, on 6,838 examples. It is not pretrained BERT.

### 4.3 — RQ1: the aggregate emoji effect → `results/emoji_impact.csv`, `figures/emoji_impact.png`

| Model | ΔMicro-F1 | ΔMacro-F1 | Direction |
|---|---|---|---|
| BiLSTM | **+2.03** | +0.99 | emoji helps |
| Transformer | **+3.71** | +2.94 | emoji helps |

**Both architectures moved in the same direction**: converting emoji to text
helped. The notebook contains a branch that would flag opposite signs as
architecture-dependence; it does not fire here.

So the aggregate answer to RQ1 is *"a positive effect of 2–4 Micro-F1 points"* —
but that claim has to be read against §4.8 before it is stated as a finding.

### 4.4 — RQ2 part one: per-emotion effect → `results/per_emotion_impact.csv`, `figures/per_emotion_impact.png`

**This is the study's most robust finding.** An aggregate figure can average a
large gain on one label against a large loss on another and report approximately
nothing, so ΔF1 is broken down by label:

| Emotion | BiLSTM w/o | BiLSTM with | Δ BiLSTM | Transf. w/o | Transf. with | Δ Transf. | Mean Δ |
|---|---|---|---|---|---|---|---|
| sadness | 54.6 | 58.0 | +3.35 | 44.8 | 56.6 | **+11.80** | **+7.58** |
| fear | 51.0 | 59.1 | **+8.09** | 46.8 | 50.6 | +3.81 | +5.95 |
| optimism | 65.3 | 67.2 | +1.87 | 63.1 | 65.6 | +2.54 | +2.21 |
| joy | 75.4 | 78.3 | +2.94 | 72.9 | 73.7 | +0.84 | +1.89 |
| love | 50.6 | 50.5 | −0.09 | 54.8 | 58.2 | +3.38 | +1.64 |
| disgust | 65.7 | 66.5 | +0.72 | 63.0 | 64.7 | +1.71 | +1.21 |
| pessimism | 31.4 | 31.4 | −0.01 | 31.2 | 33.5 | +2.29 | +1.14 |
| trust | 17.3 | 17.0 | −0.37 | 12.7 | 15.3 | +2.52 | +1.07 |
| anticipation | 31.3 | 27.2 | **−4.13** | 25.9 | 30.0 | +4.10 | −0.01 |
| anger | 68.4 | 69.3 | +0.85 | 64.3 | 63.3 | −0.96 | −0.05 |
| surprise | 15.3 | 13.0 | −2.28 | 8.4 | 8.6 | +0.27 | −1.01 |

```
per-emotion effect ranges from -4.13 to +11.80 (15.9 points of spread)
aggregate effects were only 3.71 points
```

**The aggregate figure is the near-cancellation of much larger, opposing
per-label movements — not a weak uniform trend.** The spread is more than four
times the largest aggregate effect.

Two cross-checks against `02`:

* `sadness`, `joy` and `love` — the three emotions `02` showed to be
  over-represented among emoji-bearing tweets — all gain. `sadness` gains most.
* `anger` and `disgust` — the two `02` showed emoji users *avoid* — barely move.
  `anger` is the one emotion where the two architectures disagree in sign.

That is the emoji-label skew from `02` showing up in the results exactly where it
was predicted to. It is the strongest evidence in the study that the effect is
mechanistic rather than noise.

`anticipation` is the largest disagreement between architectures: −4.13 for the
BiLSTM against +4.10 for the Transformer. Effects at that level for a single label
on 886 rows should not be over-interpreted from a single seed.

### 4.5 — Confusion matrices → `results/confusion_counts.csv`, 4 × `figures/confusion_*.png`

F1 hides *which* error is being made. One 3×4 grid per condition, eleven
per-label 2×2 matrices each, plus the raw TP/FP/FN/TN counts as CSV.

The section then ranks by false positives per true positive, and the answer is
stark. The worst case is **Transformer / Without Emoji on `surprise`: 34 TP
against 745 FP** — on a split of 886 rows, the model fires `surprise` on 88% of
tweets while only 35 actually carry it. `trust` and `pessimism` show the same
pattern more mildly.

This is the `pos_weight` of 17.94 doing exactly what it was told to do: a missed
positive costs eighteen times a false alarm, so the model buys recall at almost
any precision. The rare emotions are "predictable" in the sense that they are
predicted — but with precision low enough that those particular outputs should not
be trusted individually. Reporting Macro-F1 alongside Micro-F1 is what keeps this
visible in the headline numbers.

### 4.6 — ROC-AUC, a threshold-free check → `results/per_emotion_auc.csv`, `mean_auc.csv`, 4 × `figures/roc_*.png`

Every metric so far depends on the tuned threshold — and §3.5 noted that the
threshold sweep hit the top of its range, so this check matters more than usual.
AUC does not depend on where the cut is placed; it measures whether the model
*ranks* positives above negatives at all.

| model | track | mean AUC |
|---|---|---|
| BiLSTM | With Emoji | **0.788** |
| BiLSTM | Without Emoji | 0.786 |
| Transformer | With Emoji | **0.777** |
| Transformer | Without Emoji | 0.754 |

```
BiLSTM       ΔAUC +0.002 | ΔMicro-F1 +2.03  -> AGREE
Transformer  ΔAUC +0.023 | ΔMicro-F1 +3.71  -> AGREE
```

**Both agree in direction**, so the emoji effect is a real change in the learned
representation rather than an artefact of threshold placement.

But note the magnitudes. The Transformer's ΔAUC of +0.023 is a substantive
ranking improvement. The BiLSTM's **+0.002 is essentially zero** — its ranking
barely changed at all, even though its Micro-F1 rose 2.03 points. The honest
reading is that the emoji effect is well supported for the Transformer and only
weakly supported for the BiLSTM, where most of the F1 gain came from where the
decision boundary happened to fall rather than from better discrimination. A
single directional "AGREE" flag understates that difference.

Per-emotion AUC also confirms the difficulty ordering seen everywhere else:
`anticipation` (0.584–0.666) and `surprise` (0.676–0.739) are the hardest labels
under any metric.

### 4.7 — The precision–recall trade-off → `figures/precision_recall.png`

```
recall exceeds precision by 21.4 to 23.4 points in every condition
```

The weighted loss deliberately pushes the models to predict positives liberally.
That is the right trade for screening or exploratory use — surfacing candidate
emotional content for a human to review — and the wrong one wherever each positive
prediction triggers a costly action. Stating it explicitly is the point of the
section; it is a design choice, not a defect, but it constrains deployment.

### 4.8 — How much weight will these numbers bear?

The most important section in the notebook. Three limits:

**1. The effect is smaller than configuration-driven variation.**

```
Spread across hyper-parameter configurations : 7.38 points
Largest emoji effect measured                : 3.71 points
```

Each condition was trained **once, with a single seed**, so no estimate of
run-to-run variance exists. Simply picking a different configuration moves
Micro-F1 by twice as much as the entire emoji manipulation does. **The aggregate
result is therefore indicative, not confirmed.** The per-emotion effects (up to
11.8 points) are far larger relative to that noise floor and are the safer claim.

**2. The development split did triple duty.** Model selection, threshold
selection and final reporting all used the same 886 rows, so absolute values are
optimistic by construction. The bias applies equally to both tracks, so the
*comparison* between them survives — but the absolute Micro-F1 figures should not
be quoted as generalisation estimates.

**3. Emoji density dilutes the aggregate.** Only a minority of tweets contain
emoji, so the aggregate effect is averaged over rows the manipulation cannot
touch. On the dev split specifically that is about 75% of rows (see `02`).

### 4.9 — Optional: the held-out test split

```python
RUN_TEST_EVALUATION = False   # <- set True to consume the holdout
```

Currently **`False`** — the test split remains untouched, and `test_results_table.csv`
does not exist. Running it converts the indicative development results into
confirmatory ones and needs no retraining, taking a couple of minutes. Thresholds
stay fixed from the development split rather than being re-tuned, which is what
makes it a fair test.

Set it to `True` once, when ready to report. A drop against the development
figures is expected and healthy — those figures were optimistic by construction
(limit 2 above). This is the single highest-value thing left to run in the
project, and it is worth noting that the test split is 3,259 rows against dev's
886, so it would also cut the sampling noise substantially.

---

## Answers to the research questions

**RQ1 — to what extent does emoji handling affect performance?**
Converting emoji to their text descriptions improved Micro-F1 in both
architectures: **+2.03** points for the BiLSTM and **+3.71** for the Transformer.
Both are smaller than the 7.38-point spread across hyper-parameter configurations,
and each condition was trained once, so this is an **indicative** result rather
than a confirmed one. ROC-AUC agrees in direction for both, but only substantively
for the Transformer (+0.023 against the BiLSTM's +0.002).

**RQ2 — does the effect depend on architecture and emotion?**
**On emotion, decisively yes.** Per-emotion ΔF1 spans 15.9 points, from −4.13
(`anticipation`, BiLSTM) to +11.80 (`sadness`, Transformer) — more than four
times the largest aggregate effect. The emotions that gain are exactly those `02`
identified as over-represented among emoji-bearing tweets.

**On architecture, partly.** Both architectures moved the same way in aggregate,
so the coarse claim holds across both. But the magnitudes differ nearly twofold,
the AUC evidence supports one far more than the other, and the two disagree in
sign on `anger` and `anticipation`.

Together these point to one conclusion: **emoji handling is a label-dependent
preprocessing decision that should be validated on the target task, not applied
as a blanket default.**

---

## Outputs in full

| Path | Contents |
|---|---|
| `results/results_table.csv` | 4 rows — the headline metrics per condition |
| `results/emoji_impact.csv` | 2 rows — ΔMicro-F1 / ΔMacro-F1 per architecture (RQ1) |
| `results/per_emotion_impact.csv` | 11 rows — ΔF1 per emotion per architecture (RQ2) |
| `results/per_emotion_auc.csv` | 44 rows — AUC per emotion per condition |
| `results/mean_auc.csv` | 4 rows — mean AUC per condition |
| `results/confusion_counts.csv` | 44 rows — TP/FP/FN/TN per emotion per condition |
| `results/test_results_table.csv` | *only if §4.9 is enabled* |

| Figure | Section | Shows |
|---|---|---|
| `figures/overall_performance.png` | 4.2 | Micro-F1 / Macro-F1 / label accuracy, four conditions |
| `figures/emoji_impact.png` | 4.3 | ΔMicro-F1 per architecture (RQ1) |
| `figures/per_emotion_impact.png` | 4.4 | ΔF1 per emotion, both architectures (RQ2) |
| `figures/confusion_lstm_no_emoji.png` | 4.5 | 11 per-label confusion matrices |
| `figures/confusion_lstm_with_emoji.png` | 4.5 | " |
| `figures/confusion_bert_no_emoji.png` | 4.5 | " |
| `figures/confusion_bert_with_emoji.png` | 4.5 | " |
| `figures/roc_lstm_no_emoji.png` | 4.6 | 11 ROC curves with per-label AUC |
| `figures/roc_lstm_with_emoji.png` | 4.6 | " |
| `figures/roc_bert_no_emoji.png` | 4.6 | " |
| `figures/roc_bert_with_emoji.png` | 4.6 | " |
| `figures/precision_recall.png` | 4.7 | micro-precision vs micro-recall, four conditions |

---

## Things worth knowing

**This notebook trains nothing.** It only loads and scores, so it is cheap to
re-run and safe to iterate on. If a figure needs restyling for the write-up,
change it here — never retrain.

**It will fail loudly if `03` has not been run.** Missing checkpoints raise
`FileNotFoundError` with the instruction to run `03_model_train.ipynb` first.

**`BERT` in the keys means the from-scratch Transformer.** `MODEL_LABEL` maps it
to `"Transformer"` for every displayed table and figure title, so the saved output
is labelled accurately even though the file names and dictionary keys are not.

**The app does not depend on this notebook.** `app/streamlit_app.py` reads
`models/` and `results/thresholds.json`, both written by `03`. Everything here is
for the write-up.
