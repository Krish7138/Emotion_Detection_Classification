# -*- coding: utf-8 -*-
"""Recover the tuned decision thresholds that the notebook never wrote to disk.

The training notebook computed, per condition, the threshold that maximises
development-set Micro-F1 (sweep 0.10 -> 0.55, step 0.05) but exported only
Micro-F1, delta and best-config. This script reruns that identical sweep
locally using the saved models and dev_cleaned.csv, and writes
app/results/thresholds.json.

Run once, after placing the four .keras files in app/models/ and
dev_cleaned.csv in app/data/:

    python recompute_thresholds.py
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
import inference


def best_threshold(y_true, y_prob):
    """Identical to the notebook's best_threshold()."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.6, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int),
                      average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return round(best_t, 2), best_f1


def main():
    missing = inference.missing_models()
    if missing:
        print("ERROR: missing model files in app/models/:")
        for m in missing:
            print("   -", m)
        return 1

    if not config.DEV_CSV.exists():
        print(f"ERROR: {config.DEV_CSV} not found.")
        print("Download cleaned_data/dev_cleaned.csv from the Kaggle output into app/data/.")
        return 1

    dev = pd.read_csv(config.DEV_CSV)
    needed = set(config.EMOTION_LABELS) | {"text_no_emoji", "text_with_emoji"}
    absent = needed - set(dev.columns)
    if absent:
        print("ERROR: dev_cleaned.csv is missing columns:", sorted(absent))
        return 1

    y_true = dev[config.EMOTION_LABELS].values.astype(int)
    print(f"Loaded {len(dev)} development rows.\n")

    thresholds, report = {}, []
    for model_key in config.MODELS:
        for track, text_col, _ in [(t[0], t[1], t[2]) for t in config.TRACKS]:
            name = f"{model_key}_{track}"
            print(f"  scoring {name} ...", flush=True)
            # dev_cleaned.csv already holds the pipeline output, so feed it
            # directly rather than re-running preprocessing.
            probs = inference.predict_pre_cleaned(
                dev[text_col].astype(str).values, model_key, track)
            thr, micro = best_threshold(y_true, probs)
            thresholds[name] = thr
            report.append((name, thr, micro * 100))
            print(f"      threshold = {thr:.2f}   dev Micro-F1 = {micro * 100:.2f}%")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.THRESHOLD_JSON, "w", encoding="utf-8") as fh:
        json.dump(thresholds, fh, indent=2)

    print(f"\nWrote {config.THRESHOLD_JSON}")
    print("\nCompare against the values reported in the dissertation:")
    print(f"  {'condition':<22}{'threshold':>10}{'dev Micro-F1':>15}{'report':>10}")
    expected = {"LSTM_no_emoji": 54.64, "LSTM_with_emoji": 53.52,
                "BERT_no_emoji": 50.63, "BERT_with_emoji": 52.26}
    for name, thr, micro in report:
        print(f"  {name:<22}{thr:>10.2f}{micro:>14.2f}%{expected[name]:>9.2f}%")
    print("\nSmall differences from the report are expected only if the models "
          "were retrained; identical models should reproduce these figures exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
