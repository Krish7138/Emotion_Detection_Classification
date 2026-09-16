# -*- coding: utf-8 -*-
"""Model loading and prediction.

Deliberately free of Streamlit imports so that recompute_thresholds.py can
reuse it from the command line.

Serving contract (see krish.ipynb):
  * models take raw strings  -> Input(shape=(), dtype=tf.string); the
    TextVectorization layer and its 20k vocabulary live inside the .keras file
  * models emit RAW LOGITS   -> Dense(11) with no activation, so sigmoid is
    applied here, matching the notebook's predict()
  * models were compiled with the custom weighted_bce loss, so they are loaded
    with compile=False; inference never needs the loss or its POS_WEIGHT
"""
import json
import os
from functools import lru_cache

import numpy as np

import config
from preprocessing import build_text

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def model_path(model_key, track):
    return config.MODEL_DIR / config.MODELS[model_key]["files"][track]


def available_models():
    """Which of the four (model_key, track) checkpoints are present on disk."""
    out = {}
    for mk in config.MODELS:
        for track in ("no_emoji", "with_emoji"):
            out[(mk, track)] = model_path(mk, track).exists()
    return out


def all_models_present():
    return all(available_models().values())


def missing_models():
    return [config.MODELS[mk]["files"][t]
            for (mk, t), ok in available_models().items() if not ok]


@lru_cache(maxsize=8)
def load_model(model_key, track):
    """Load one checkpoint. Cached, because loading is the slow part."""
    import tensorflow as tf
    path = model_path(model_key, track)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Download the four .keras files from the Kaggle output into app/models/."
        )
    return tf.keras.models.load_model(str(path), compile=False)


def predict_proba(texts, model_key, track):
    """Raw texts -> (n, 11) probability array. Applies the pipeline + sigmoid."""
    import tensorflow as tf
    if isinstance(texts, str):
        texts = [texts]
    mode = "no" if track == "no_emoji" else "yes"
    cleaned = [build_text(t, mode) for t in texts]
    model = load_model(model_key, track)
    # tf.constant, not a numpy <U array: Keras 3 rejects numpy string dtypes
    # on a tf.string input layer.
    logits = model.predict(tf.constant(cleaned, dtype=tf.string), verbose=0)
    return tf.nn.sigmoid(logits).numpy()


def predict_pre_cleaned(cleaned_texts, model_key, track):
    """For already-preprocessed text (e.g. dev_cleaned.csv columns)."""
    import tensorflow as tf
    model = load_model(model_key, track)
    texts = [str(t) for t in cleaned_texts]
    logits = model.predict(tf.constant(texts, dtype=tf.string),
                           batch_size=64, verbose=0)
    return tf.nn.sigmoid(logits).numpy()


# --- thresholds -------------------------------------------------------------

def load_thresholds():
    """Return {'LSTM_no_emoji': 0.30, ...} or None if not yet recovered."""
    if not config.THRESHOLD_JSON.exists():
        return None
    with open(config.THRESHOLD_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def threshold_for(thresholds, model_key, track, default=0.5):
    if not thresholds:
        return default
    return float(thresholds.get(f"{model_key}_{track}", default))


def predict_one(text, model_key, thresholds=None):
    """Run one raw input down BOTH tracks; this is the core of the app.

    Returns a dict keyed by track with probabilities, the applied threshold,
    the predicted label set, the exact model input and an `empty` flag.

    The empty case is real and must be short-circuited: an emoji-only input
    leaves Track A with nothing after preprocessing, and feeding an empty
    string to a model whose Embedding uses mask_zero=True produces an
    entirely masked sequence, so the pooling step divides by zero and the
    model returns NaN. Calling the model at all there would be meaningless.
    """
    n = len(config.EMOTION_LABELS)
    out = {}
    for track in ("no_emoji", "with_emoji"):
        mode = "no" if track == "no_emoji" else "yes"
        model_input = build_text(text, mode)
        thr = threshold_for(thresholds, model_key, track)

        if not model_input.strip():
            probs = np.zeros(n, dtype=np.float32)
            empty = True
        else:
            probs = predict_pre_cleaned([model_input], model_key, track)[0]
            # belt and braces: never let a NaN reach the UI
            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            empty = False

        out[track] = {
            "probs": probs,
            "threshold": thr,
            "labels": ([] if empty else
                       [lab for lab, p in zip(config.EMOTION_LABELS, probs)
                        if p >= thr]),
            "model_input": model_input,
            "empty": empty,
        }

    a = set(out["no_emoji"]["labels"])
    b = set(out["with_emoji"]["labels"])
    out["diff"] = {
        "gained": sorted(b - a),   # emoji made these appear
        "lost": sorted(a - b),     # emoji made these disappear
        "shared": sorted(a & b),
        "changed": bool(a ^ b),
        # a comparison is only meaningful when both tracks saw real text
        "comparable": not (out["no_emoji"]["empty"] or out["with_emoji"]["empty"]),
    }
    return out
