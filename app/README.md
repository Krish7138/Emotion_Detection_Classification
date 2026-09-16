# Emoji Impact on Emotion Recognition - Streamlit deliverable

Interactive front end for the MSc Computing Research Project *"Does the Emoji Matter?"*.

Rather than a generic emotion classifier, the app is built around the project's
research question: every input is run down **both** preprocessing tracks -
emoji removed (Track A) and emoji converted to their Unicode descriptions
(Track B) - through the same trained model, so the user can see exactly what
the emoji changed.

---

## 1. Setup

There is none, beyond running the training notebook once.

`notebooks/03_model_train.ipynb` writes its artefacts straight into the folders
this app reads, so nothing is copied and no extra script is run:

| Written by notebook 03 | Read by the app |
|---|---|
| `models/*.keras` (four files) | the classifier |
| `results/thresholds.json` | the decision thresholds |
| `data/processed/dev_cleaned.csv` | the fallback threshold recovery |

```bash
pip install -r ../requirements.txt
python ../download_data.py        # fetches the corpus into data/raw/
# then run notebooks 01 -> 02 -> 03
```

## 2. Run

From the project root:

```bash
streamlit run app/streamlit_app.py
```

### If the models came from somewhere else

`recompute_thresholds.py` remains as a fallback for checkpoints that arrive
without `thresholds.json` - for example downloaded from an older Kaggle run:

```bash
python recompute_thresholds.py
```

The thresholds are not optional. The models are trained with a positive-weighted
loss, so the value maximising Micro-F1 sits well below 0.5, and that is what makes
rare emotions (*trust*, *surprise*, *pessimism*) predictable at all. On the 0.5
default the app would predict almost nothing for those labels and would contradict
the dissertation.

## 3. Pages

**Classify** - paste any comment, sentence or single word and it is classified
into the eleven emotions. The headline result is the detected emotion set with
confidence for each, backed by a ranked table of all eleven showing which passed
the decision threshold. Below that, the same input is compared across both
preprocessing tracks: a *"what the emoji changed"* panel listing labels gained
or lost, a per-emotion probability chart with each track's threshold drawn on
it, and the full stage-by-stage preprocessing trace.

Because the model is **multi-label**, a comment can legitimately carry several
emotions at once, so the result is a set rather than one category. Two cases are
handled explicitly: when no emotion clears the threshold the three strongest
signals are shown for reference, and when a track is left with empty text after
preprocessing (an emoji-only input strips Track A to nothing) no prediction is
attempted at all - see the note below.

**Model performance** - corpus statistics, final results for all four
conditions, ΔF1 by architecture, the per-emotion breakdown and mean ROC-AUC.
All figures are reproduced from the executed notebook so the app and the
dissertation agree exactly. Works without the model files.

**Evaluation feedback** - collects user judgements (accuracy rating, which
track read better, usefulness, free comment) and appends them to
`feedback/user_feedback.csv`, downloadable as CSV. This supplies the
user-evaluation evidence the handbook requires for an Applied project.

## 4. Files

```
config.py                 labels, paths, model registry, reference results
preprocessing.py          copy of the pipeline defined in notebook 03
inference.py              model loading, sigmoid, thresholds, both-track predict
recompute_thresholds.py   fallback threshold recovery (usually unnecessary)
streamlit_app.py          the three-page UI
feedback/                 user_feedback.csv, written by the Evaluation page
```

Models, results and processed data live at the **project root**, not under
`app/`, because the notebooks write them there.

## 5. Notes for maintainers

**Do not "improve" `preprocessing.py`.** The models were fitted on the exact
output of the notebook's pipeline. Any drift in the regexes, the slang map or
the tokeniser flags silently invalidates every prediction. The port is verified
against notebook 03's own worked example (also stored as
`results/pipeline_fixture.json`): for the input
`I'm so happy today 😂😭 u cant believe it!! #joy @friend http://x.com`,
Track B must produce
`i m so happy today face with tear of joy loudly cry face you cannot believe it joy`.

**Serving contract.** The models take raw strings - `Input(shape=(), dtype=tf.string)`
with `TextVectorization` and its 20k vocabulary inside the `.keras` file, so no
separate vectoriser artefact is needed. They emit **raw logits** from a bare
`Dense(11)`, so the sigmoid is applied in `inference.py`. They are loaded with
`compile=False` because the training loss (`weighted_bce`) is a custom function
that inference does not need.

Pass strings as `tf.constant(..., dtype=tf.string)`; Keras 3 rejects numpy `<U`
string arrays on a `tf.string` input layer.

**Naming.** The notebook calls the attention model `bert` and the checkpoint
filenames keep that name, but it is a Transformer encoder **trained from
scratch**, not pretrained BERT. The UI labels it accurately.

**Empty input after preprocessing.** An emoji-only comment such as `😂😂😂`
leaves Track A with an empty string once emoji are removed. Feeding that to a
model whose `Embedding` uses `mask_zero=True` produces a fully masked sequence,
so the pooling step divides by zero and the model returns **NaN**. `predict_one`
therefore short-circuits: it never calls the model on empty text, returns zeros
with an `empty` flag, and the UI reports "no prediction possible" instead of
showing meaningless numbers. Probabilities are also passed through
`np.nan_to_num` as a second guard.

**Version drift.** Training used TensorFlow 2.19 / Keras 3 / Python 3.12. Keras 3
checkpoints load fine on TF 2.20, but if you hit a deserialisation error, pin
`tensorflow==2.19.0`.

## 6. Honesty notes carried into the UI

The app displays these caveats because the dissertation does:

- All metrics are **development-split** figures; the test split was preserved unused.
- The Transformer was trained from scratch, so absolute scores sit below published state of the art.
- Only **11.3%** of training tweets contained emoji, which bounds the size of any emoji effect.
- The aggregate ΔF1 values (−1.12 / +1.63) are within the range run-to-run variance could produce, and are indicative rather than confirmed.
