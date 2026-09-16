# -*- coding: utf-8 -*-
"""Shared configuration, paths and reference results.

Every constant here mirrors the training notebook (krish.ipynb) so that the
served models see exactly the inputs they were trained on.
"""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT = APP_DIR.parent

# The app reads the artefacts the notebooks write, in place. Nothing is copied,
# so once 03_model_train.ipynb has run, `streamlit run app/streamlit_app.py`
# works with no further setup.
MODEL_DIR = PROJECT / "models"
DATA_DIR = PROJECT / "data" / "processed"
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"

ASSETS_DIR = APP_DIR / "assets"
FEEDBACK_DIR = APP_DIR / "feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "user_feedback.csv"
THRESHOLD_JSON = RESULTS_DIR / "thresholds.json"
DEV_CSV = DATA_DIR / "dev_cleaned.csv"

# --- from notebook cell 5 ---------------------------------------------------
EMOTION_LABELS = [
    "anger", "anticipation", "disgust", "fear", "joy", "love",
    "optimism", "pessimism", "sadness", "surprise", "trust",
]

# (track_key, cleaned-text column, display label)
TRACKS = [
    ("no_emoji", "text_no_emoji", "Without Emoji"),
    ("with_emoji", "text_with_emoji", "With Emoji"),
]

MAX_TOKENS = 20000
SEQUENCE_LENGTH = 128

# Threshold sweep used in the notebook's best_threshold()
THRESHOLD_GRID = [round(0.10 + 0.05 * i, 2) for i in range(10)]  # 0.10 .. 0.55

# --- model registry ---------------------------------------------------------
# The notebook names the attention model "bert", but it is a Transformer encoder
# trained from scratch, NOT pretrained BERT. Files keep the notebook's names;
# the UI uses the accurate label.
MODELS = {
    "LSTM": {
        "label": "BiLSTM",
        "long_label": "Bidirectional LSTM",
        "files": {"no_emoji": "lstm_no_emoji.keras",
                  "with_emoji": "lstm_with_emoji.keras"},
    },
    "BERT": {
        "label": "Transformer",
        "long_label": "Transformer encoder (trained from scratch)",
        "files": {"no_emoji": "bert_no_emoji.keras",
                  "with_emoji": "bert_with_emoji.keras"},
    },
}

TRACK_LABELS = {"no_emoji": "Without Emoji (Track A)",
                "with_emoji": "With Emoji (Track B)"}

# --- reference results, taken from the executed notebook outputs -------------
# Used by the Performance page so the app and the dissertation agree exactly.
FINAL_RESULTS = [
    # model, track, config, micro_f1, macro_f1, label_acc, precision, recall
    ("LSTM", "no_emoji", "cfg3", 54.64, 49.47, 75.42, 46.24, 66.77),
    ("LSTM", "with_emoji", "cfg3", 53.52, 48.04, 73.57, 43.86, 68.63),
    ("BERT", "no_emoji", "cfg1", 50.63, 43.98, 75.49, 45.74, 56.69),
    ("BERT", "with_emoji", "cfg3", 52.26, 45.58, 75.91, 46.61, 59.46),
]

EMOJI_IMPACT = {  # ΔMicro-F1, ΔMacro-F1 in percentage points
    "LSTM": (-1.12, -1.43),
    "BERT": (+1.63, +1.60),
}

# per-emotion F1: (LSTM_no, LSTM_yes, BERT_no, BERT_yes) and the notebook deltas
PER_EMOTION_F1 = {
    "sadness":      (57.4, 56.9, 39.0, 60.0),
    "joy":          (73.5, 76.1, 71.7, 74.9),
    "disgust":      (66.7, 67.3, 58.6, 62.9),
    "pessimism":    (31.4, 28.7, 32.6, 36.9),
    "fear":         (59.8, 54.4, 48.3, 53.7),
    "trust":        (19.5, 16.9, 11.3, 12.8),
    "love":         (52.6, 49.7, 54.2, 54.8),
    "optimism":     (67.3, 65.2, 63.2, 62.6),
    "anger":        (68.2, 65.4, 63.0, 60.3),
    "anticipation": (28.8, 33.5, 24.0, 11.2),
    "surprise":     (19.2, 14.3, 17.9, 11.3),
}
PER_EMOTION_DELTA = {  # (ΔLSTM, ΔTransformer) straight from notebook cell 48
    "sadness": (-0.45, 21.01), "joy": (2.61, 3.18), "disgust": (0.61, 4.28),
    "pessimism": (-2.72, 4.35), "fear": (-5.31, 5.38), "trust": (-2.59, 1.46),
    "love": (-2.83, 0.61), "optimism": (-2.04, -0.55), "anger": (-2.82, -2.77),
    "anticipation": (4.65, -12.78), "surprise": (-4.89, -6.55),
}

MEAN_AUC = {("LSTM", "no_emoji"): 0.798, ("LSTM", "with_emoji"): 0.794,
            ("BERT", "no_emoji"): 0.759, ("BERT", "with_emoji"): 0.775}

CORPUS_STATS = {
    "train": 6838, "dev": 886, "test": 3259,
    "emoji_pct": 11.3, "emoji_count": 770,
}

EXAMPLES = [
    "I finally got the job I wanted 😂🎉 cant believe it",
    "stuck in traffic again 😡 this city is broken",
    "not sure how I feel about the news today 😐 kind of worried honestly",
    "my cat knocked over the plant 😭 but honestly it was funny",
    "waiting to hear back and its killing me 😰 fingers crossed",
    "absolutely gutted about the result today",  # no emoji: tracks agree
    "😂😂😂",  # emoji only: Track A is left with nothing at all
]

DISCLAIMER = (
    "All reported metrics are development-split figures; the test split was preserved unused. "
    "The attention model is a Transformer encoder trained from scratch, not pretrained BERT. "
    "Only 11.3% of training tweets contained emoji, which bounds the size of any emoji effect."
)
