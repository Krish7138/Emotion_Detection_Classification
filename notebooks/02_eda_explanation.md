# `02_eda.ipynb` - Explanation

**Stage 2 of 5.** Profile the corpus *before* any text is modified. Every figure
here exists to justify - or fail to justify - a decision made in `03` or a caveat
stated in `04`.

| | |
|---|---|
| **Input** | `data/interim/{train,dev,test}_raw.csv` |
| **Output** | `results/eda_text_quality.csv`, `results/eda_emoji_density.csv` |
| **Figures** | `label_distribution.png`, `label_cardinality.png`, `label_cooccurrence.png`, `text_length.png`, `emoji_label_skew.png` |
| **Next** | `03_model_train.ipynb` |
| **Runtime** | under a minute, CPU only |

---

## Why this notebook exists

This is not decorative EDA. Three questions drive it, and each one has a
consequence:

1. **How imbalanced are the labels?** → decides whether a weighted loss is needed.
2. **What noise does the text carry?** → decides which preprocessing stages are justified.
3. **How many tweets actually contain emoji?** → this is the **ceiling on the entire
   study**, because the two tracks are byte-identical on every tweet without one.

It profiles the **training split** specifically, because that is what the model
learns from and what the class weights are computed from. Emoji density is
reported for all three splits.

---

## Walk-through

### 2.1 - Split sizes

| split | tweets | share % |
|---|---|---|
| train | 6,838 | 62.3 |
| dev | 886 | 8.1 |
| test | 3,259 | 29.7 |

Note the shape of this: the development split is the smallest of the three, yet
it carries all model selection, all threshold tuning and all reported metrics.
That is a real limitation, and `04` §4.8 states it explicitly.

### 2.2 - Label distribution → `figures/label_distribution.png`

Counts positive instances per emotion in the training split.

```
most frequent  : disgust (2,602)
least frequent : trust   (357)
imbalance ratio: 7.3 : 1
```

**Consequence.** A 7.3:1 spread means an unweighted binary cross-entropy is
minimised most efficiently by predicting "no" for `trust` on every single row -
that alone scores about 95% accuracy on that label. The notebook prints the
verdict directly:

> `-> severe imbalance: a class-weighted loss and threshold tuning are justified`

`03` acts on this in two places: a per-label `pos_weight` (§3.3) and a swept
decision threshold rather than a fixed 0.5 (§3.5).

### 2.3 - Multi-label structure → `figures/label_cardinality.png`

How many emotions ride on one tweet?

```
mean labels per tweet : 2.35
tweets with 0 labels  : 204
tweets with 2 or more : 5,652 (82.7%)
```

**Consequence.** 82.7% of tweets carry two or more emotions, so this cannot be
reduced to single-label classification, and plain accuracy is meaningless as a
headline. `04` reports Micro-F1 and Macro-F1 with independent per-label decisions
instead.

### 2.4 - Label co-occurrence → `figures/label_cooccurrence.png`

A heatmap of `P(column emotion | row emotion)`, diagonal blanked out. The
strongest pairs:

| given a | then b | P(b given a) |
|---|---|---|
| love | joy | 0.936 |
| anger | disgust | 0.811 |
| disgust | anger | 0.793 |
| trust | optimism | 0.779 |
| optimism | joy | 0.754 |
| trust | joy | 0.697 |

**Consequence.** These are near-deterministic implications: 94% of tweets labelled
`love` are also labelled `joy`. Two things follow. First, they explain confusions
that appear in `04`'s confusion matrices - a model firing `joy` on a `love` tweet
is following the data, not malfunctioning. Second, the architecture used here
predicts each label with an independent output unit and so cannot represent
`love → joy` directly; it has to relearn it from the text every time. That is
known headroom, not a bug.

### 2.5 - Text noise → `figures/text_length.png`, `results/eda_text_quality.csv`

| property | value | % of split | justifies |
|---|---|---|---|
| mean characters | 95.2 | - | sequence length of 128 tokens is ample |
| mean words | 16.1 | - | short texts: every token matters |
| contains a URL | 0 | 0.0 | URL stripping |
| contains @mention | 3,061 | 44.8 | mention stripping + de-identification |
| contains #hashtag | 2,998 | 43.8 | keep the word, drop the `#` |
| contains emoji | 770 | 11.3 | the manipulated variable of this study |

**Consequences, one per row:**

* **Mean 16 words.** A 128-token sequence length sits far above the tail, so
  effectively nothing is truncated. Also worth noticing: with only 16 tokens to
  work with, adding two or three emoji-description tokens is a *large* relative
  change to a tweet's content - which is why the per-tweet effect can be big even
  when the corpus-wide effect is small.
* **Zero URLs.** The release already stripped them. The URL regex in `03` is
  therefore dead code on this corpus - it is retained because
  `app/preprocessing.py` must mirror `03` exactly, and the app *does* see user
  text containing URLs.
* **44.8% contain an @mention.** This is the preprocessing stage that does real
  work, and it doubles as de-identification: removing handles removes the most
  obvious personal identifier in the corpus.
* **43.8% contain a hashtag.** Hashtags are frequently the emotional payload
  (`#joy`), so the `#` symbol is dropped and the word kept - deleting the whole
  token would throw away signal.

> ⚠️ **Caveat on the saved CSV.** The `% of split` column is computed as
> `100 × value / len(train)` for *every* row, so it is only meaningful for the
> four count rows. The `1.4` and `0.2` against "mean characters" and "mean words"
> are an artefact of that formula - ignore them and cite the raw `value` for
> those two.

### 2.6 - Emoji density → `results/eda_emoji_density.csv`

**This is the single most important number in the project.**

| split | tweets | with emoji | % with emoji |
|---|---|---|---|
| train | 6,838 | 770 | 11.3 |
| dev | 886 | 220 | **24.8** |
| test | 3,259 | 799 | 24.5 |

The two preprocessing tracks differ *only* on rows containing at least one emoji.
On every other row they emit byte-identical text. So this percentage is a hard
ceiling on any aggregate difference the study can measure:

> the aggregate difference between the two tracks is diluted by the 88.7% of
> training rows the manipulation cannot touch.

**Read the split difference carefully.** Emoji density is not uniform: 11.3% of
training rows carry emoji, but roughly 25% of dev and test rows do. So the
*training* signal is thin while the *evaluation* set is comparatively rich in
emoji-bearing rows. The dilution argument is real, but on the split the metrics
actually come from it is about 75%, not 89%. This asymmetry is a property of the
official release rather than anything the notebook does - but it should be stated
whenever the 11.3% figure is quoted as the study's ceiling.

### Which emoji appear?

277 distinct emoji in the training split. The head of the distribution:

| emoji | tweets | description |
|---|---|---|
| 😂 | 119 | face with tears of joy |
| 😭 | 53 | loudly crying face |
| 🙄 | 39 | face with rolling eyes |
| 😩 | 38 | weary face |
| 😊 | 24 | smiling face with smiling eyes |

The distribution is steep - one emoji accounts for roughly 15% of all
emoji-bearing tweets. Track B converts these to text (`face with tears of joy`),
so those exact description words are what the model gets a chance to learn from,
and only the head of this distribution appears often enough to be learnable at
all.

### Do emoji-bearing tweets carry different emotions? → `figures/emoji_label_skew.png`

Label prevalence among emoji-bearing tweets minus prevalence among the rest, in
percentage points:

| emotion | with emoji | without | difference |
|---|---|---|---|
| joy | 49.6 | 34.5 | **+15.1** |
| love | 17.1 | 9.4 | **+7.8** |
| sadness | 33.9 | 28.8 | +5.1 |
| pessimism | 12.9 | 11.5 | +1.4 |
| surprise | 6.5 | 5.1 | +1.4 |
| optimism | 28.8 | 29.0 | -0.2 |
| fear | 16.5 | 18.4 | -1.9 |
| trust | 3.5 | 5.4 | -1.9 |
| anticipation | 12.3 | 14.6 | -2.2 |
| anger | 33.5 | 37.7 | -4.2 |
| disgust | 33.2 | 38.7 | -5.4 |

**Consequence.** The emoji subset is emotionally skewed, and strongly so. Emoji
usage clusters on `joy`, `love` and `sadness`, and is *avoided* on `disgust` and
`anger`. This is a prediction made before any model is trained: the manipulation
can only plausibly move labels it has coverage of. `04` bears it out - `sadness`
moved most (mean ΔF1 +7.58) and `joy` gained (+1.89), while `anger` (-0.05) and
`surprise` (-1.01) barely moved or went backwards.

---

## Outputs in full

| Path | What it is |
|---|---|
| `results/eda_text_quality.csv` | the six-row noise profile (see the caveat on `% of split`) |
| `results/eda_emoji_density.csv` | emoji-bearing tweet counts per split |

| Figure | Section | Shows |
|---|---|---|
| `figures/label_distribution.png` | 2.2 | positives per emotion, 7.3:1 imbalance |
| `figures/label_cardinality.png` | 2.3 | how many emotions per tweet |
| `figures/label_cooccurrence.png` | 2.4 | P(b given a) heatmap across the eleven labels |
| `figures/text_length.png` | 2.5 | tweet character-length histogram |
| `figures/emoji_label_skew.png` | 2.6 | which emotions the emoji subset over-represents |

