# -*- coding: utf-8 -*-
"""Emoji Impact on Emotion Recognition — interactive deliverable.

Primary flow: the user pastes a comment, sentence or single word and gets it
classified into the eleven emotions by the trained model.

Secondary flow (the research question): the same input is also run down the
other preprocessing track, so the user can see what the emoji changed.
"""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import inference
import preprocessing as pp

st.set_page_config(page_title="Emoji Impact on Emotion Recognition",
                   page_icon="🎭", layout="wide")

TRACK_COLOUR = {"no_emoji": "#3B6EA5", "with_emoji": "#B3512F"}


# --------------------------------------------------------------------------
# cached wrappers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _cached_model(model_key, track):
    return inference.load_model(model_key, track)


def _cached_thresholds():
    # deliberately uncached: the file appears only after the user runs
    # recompute_thresholds.py, and caching would pin a stale "missing" result
    return inference.load_thresholds()


def run_prediction(text, model_key, thresholds):
    for track in ("no_emoji", "with_emoji"):
        _cached_model(model_key, track)
    return inference.predict_one(text, model_key, thresholds)


def ranked_frame(result, track):
    """All eleven emotions ordered by probability, with pass/fail marked."""
    info = result[track]
    rows = [{"Emotion": lab,
             "Probability": float(p),
             "Detected": bool(p >= info["threshold"])}
            for lab, p in zip(config.EMOTION_LABELS, info["probs"])]
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


# --------------------------------------------------------------------------
# shared UI pieces
# --------------------------------------------------------------------------
def probability_figure(result, model_key):
    """Grouped bar of per-emotion probability, Track A vs Track B."""
    labels = config.EMOTION_LABELS
    fig = go.Figure()
    for track in ("no_emoji", "with_emoji"):
        fig.add_bar(
            name=config.TRACK_LABELS[track],
            x=labels,
            y=[round(float(p), 4) for p in result[track]["probs"]],
            marker_color=TRACK_COLOUR[track],
        )
    # annotate on opposite sides so the labels never collide when the two
    # tuned thresholds happen to be equal
    for track, side in (("no_emoji", "top left"), ("with_emoji", "bottom right")):
        thr = result[track]["threshold"]
        fig.add_hline(y=thr, line_dash="dot", line_width=1.5,
                      line_color=TRACK_COLOUR[track],
                      annotation_text=f"threshold {thr:.2f}",
                      annotation_position=side,
                      annotation_font_size=10,
                      annotation_font_color=TRACK_COLOUR[track])
    fig.update_layout(
        barmode="group", height=440,
        margin=dict(l=10, r=10, t=70, b=10),
        yaxis_title="Predicted probability", xaxis_title="",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", y=1.08, x=0),
        title=dict(
            text=f"{config.MODELS[model_key]['label']} — per-emotion probability by track",
            y=0.97),
    )
    return fig


def status_sidebar():
    st.sidebar.markdown("### Status")
    avail = inference.available_models()
    n_ok = sum(avail.values())
    if n_ok == 4:
        st.sidebar.success("4 / 4 models loaded")
    else:
        st.sidebar.error(f"{n_ok} / 4 models found")
    thr = _cached_thresholds()
    if thr:
        st.sidebar.success("Tuned thresholds loaded")
    else:
        st.sidebar.warning("Thresholds not recovered — using 0.5")
    return thr


def setup_help():
    st.title("Trained models not found")
    st.markdown(
        "This app serves the models produced by the training notebook. They are not "
        "on disk yet, so there is nothing to classify with. The **Model performance** "
        "page works without them."
    )

    st.subheader("Run the training notebook once")
    st.markdown(
        "`notebooks/03_model_train.ipynb` writes everything this app needs, straight "
        "into the folders it already reads — no copying and no extra scripts:"
    )
    st.code(
        "models/lstm_no_emoji.keras        <- the four trained models\n"
        "models/lstm_with_emoji.keras\n"
        "models/bert_no_emoji.keras\n"
        "models/bert_with_emoji.keras\n"
        "results/thresholds.json           <- tuned decision thresholds\n"
        "data/processed/dev_cleaned.csv    <- the processed splits",
        language=None)
    st.markdown(
        "Once it has finished, restart this app and it will pick everything up "
        "automatically:")
    st.code("streamlit run app/streamlit_app.py", language="bash")

    st.subheader("Currently missing")
    missing = inference.missing_models()
    st.code("\n".join(f"models/{m}" for m in missing) or "(models present)",
            language=None)
    if not config.THRESHOLD_JSON.exists():
        st.code("results/thresholds.json", language=None)

    st.info(
        "**Why the notebook order matters.** `01_data_load` → `02_eda` → "
        "`03_model_train` each read what the previous one wrote, so 03 will fail if "
        "01 has not been run. Only 03 is strictly required to make this app work; "
        "`04_evaluation` produces the result tables and figures for the report."
    )

    with st.expander("If the models were trained on Kaggle"):
        st.markdown(
            "Losing the interactive session does **not** delete the output. If the "
            "notebook was run with *Save Version* / *Save & Run All*, everything "
            "written to `/kaggle/working` is stored permanently:\n\n"
            "1. Open the notebook on Kaggle\n"
            "2. Go to the **Output** tab (or *Version history* → the run → **Output**)\n"
            "3. Download `models/` and `results/thresholds.json` into the matching "
            "folders here\n\n"
            "Re-training instead is fine, but it will not reproduce the exact figures "
            "in the dissertation: weight initialisation is not seeded, so Micro-F1 "
            "will land close to but not identical to the reported values."
        )


# --------------------------------------------------------------------------
# page 1 — classify
# --------------------------------------------------------------------------
def render_classification(result, track, model_key):
    """The headline answer: what emotions is this text?"""
    info = result[track]
    df = ranked_frame(result, track)
    detected = df[df["Detected"]]

    st.subheader("Detected emotions")

    if info["empty"]:
        st.error(
            "**No prediction possible for this track.** After preprocessing it "
            "received empty text — everything in the input was removed, so there is "
            "nothing for the model to classify."
        )
        other = "with_emoji" if track == "no_emoji" else "no_emoji"
        if not result[other]["empty"]:
            st.info(
                f"This is the research point in miniature: the emoji carried *all* "
                f"the content here. Switch **Emoji handling** to see "
                f"{config.TRACK_LABELS[other].split(' (')[0].lower()}, which read "
                f"`{result[other]['model_input']}`."
            )
        return

    if len(detected):
        chips = "  ".join(
            f"### `{r.Emotion}`" for r in detected.itertuples())
        st.markdown(" ".join(f"`{r.Emotion}`" for r in detected.itertuples()))
        cols = st.columns(min(len(detected), 5))
        for i, r in enumerate(detected.head(5).itertuples()):
            cols[i].metric(r.Emotion.capitalize(), f"{r.Probability * 100:.0f}%")
        if len(detected) > 5:
            st.caption("Also detected: " +
                       ", ".join(detected["Emotion"].iloc[5:]))
    else:
        st.warning(
            f"No emotion reached the decision threshold of {info['threshold']:.2f}, "
            "so the model detects no clear emotion here. The three strongest "
            "signals are shown below for reference."
        )
        cols = st.columns(3)
        for i, r in enumerate(df.head(3).itertuples()):
            cols[i].metric(r.Emotion.capitalize(), f"{r.Probability * 100:.0f}%",
                           delta="below threshold", delta_color="off")

    st.caption(
        f"{config.MODELS[model_key]['label']} · {config.TRACK_LABELS[track]} · "
        f"decision threshold {info['threshold']:.2f} · this is a multi-label model, "
        "so a comment can carry several emotions at once")

    with st.expander("All eleven emotions, ranked"):
        st.dataframe(
            df.assign(Probability=df["Probability"].round(4)),
            use_container_width=True, hide_index=True,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.3f"),
                "Detected": st.column_config.CheckboxColumn("Above threshold"),
            })
        st.caption(f"Model input after preprocessing: `{info['model_input'] or '(empty)'}`")


def page_predict(thresholds):
    st.title("Classify a comment")
    st.caption(
        "Paste any comment, sentence or single word. It is classified into eleven "
        "emotions by the model trained in this project. The same text is also run "
        "through the other preprocessing track so you can see what the emoji changed."
    )

    if "text" not in st.session_state:
        st.session_state.text = config.EXAMPLES[0]

    with st.expander("Try an example", expanded=False):
        cols = st.columns(3)
        for i, ex in enumerate(config.EXAMPLES):
            if cols[i % 3].button(ex[:38] + ("…" if len(ex) > 38 else ""),
                                  key=f"ex{i}", use_container_width=True):
                st.session_state.text = ex
                st.rerun()

    text = st.text_area(
        "Your comment", key="text", height=120,
        placeholder="Paste a comment, a sentence, or a single word…")

    c1, c2, c3 = st.columns([1, 1, 2])
    model_key = c1.selectbox(
        "Model", list(config.MODELS),
        format_func=lambda k: config.MODELS[k]["label"])
    track_choice = c2.selectbox(
        "Emoji handling", ["with_emoji", "no_emoji"],
        format_func=lambda t: {"with_emoji": "Keep emoji (as text)",
                               "no_emoji": "Remove emoji"}[t])
    c3.caption(config.MODELS[model_key]["long_label"])
    go_btn = st.button("Classify", type="primary")

    if not text.strip():
        st.info("Enter or paste some text to classify.")
        return

    found = pp.emoji_inventory(text)
    if found:
        st.markdown("**Emoji found:** " + "  ".join(
            f"{e} → *{d}*" for e, d in found))

    if not go_btn and "result" not in st.session_state:
        return

    if go_btn:
        with st.spinner("Classifying…"):
            st.session_state.result = run_prediction(text, model_key, thresholds)
            st.session_state.result_text = text
            st.session_state.result_model = model_key

    result = st.session_state.result
    model_key = st.session_state.result_model

    st.divider()
    render_classification(result, track_choice, model_key)

    # ---- the research angle, secondary to the classification itself -------
    st.divider()
    st.subheader("What the emoji changed")
    d = result["diff"]
    if not d["comparable"]:
        empty_track = ("no_emoji" if result["no_emoji"]["empty"] else "with_emoji")
        st.warning(
            f"The two tracks cannot be compared for this input, because "
            f"{config.TRACK_LABELS[empty_track].split(' (')[0].lower()} was left with "
            "no text at all after preprocessing.")
    elif not found:
        st.caption(
            "This text contains no emoji, so both tracks received identical input "
            "and the predictions agree by construction — a useful control case.")
    elif not d["changed"]:
        st.success(
            "The predicted emotion set is identical whether the emoji are removed "
            "or converted to text. Emoji handling made no difference here.")
    else:
        g, l = st.columns(2)
        with g:
            st.markdown("**Appeared only when emoji were kept**")
            st.markdown(" ".join(f"`{x}`" for x in d["gained"]) if d["gained"] else "—")
        with l:
            st.markdown("**Lost when emoji were kept**")
            st.markdown(" ".join(f"`{x}`" for x in d["lost"]) if d["lost"] else "—")
        if d["shared"]:
            st.caption("Predicted on both tracks: " + ", ".join(d["shared"]))

    st.plotly_chart(probability_figure(result, model_key), use_container_width=True)

    with st.expander("Preprocessing pipeline, stage by stage"):
        rows = pp.pipeline_stages(st.session_state.result_text)
        st.dataframe(
            pd.DataFrame(
                [{"Stage": n, "Track A — without emoji": x,
                  "Track B — with emoji": y} for n, x, y in rows]),
            use_container_width=True, hide_index=True)

    st.caption(config.DISCLAIMER)


# --------------------------------------------------------------------------
# page 2 — performance
# --------------------------------------------------------------------------
def page_performance():
    st.title("Model performance")
    st.caption(
        "Figures reproduced from the training notebook. All are development-split "
        "results; the test split was preserved unused.")

    s = config.CORPUS_STATS
    k = st.columns(4)
    k[0].metric("Training tweets", f"{s['train']:,}")
    k[1].metric("Development tweets", f"{s['dev']:,}")
    k[2].metric("Test tweets (unused)", f"{s['test']:,}")
    k[3].metric("Tweets with emoji", f"{s['emoji_pct']}%",
                help=f"{s['emoji_count']} of {s['train']:,} training tweets")

    st.subheader("Final performance of the four conditions")
    st.dataframe(pd.DataFrame(
        [{"Model": config.MODELS[m]["label"],
          "Track": config.TRACK_LABELS[t].split(" (")[0],
          "Config": c, "Micro-F1": mi, "Macro-F1": ma,
          "Label acc.": la, "Precision": pr, "Recall": rc}
         for m, t, c, mi, ma, la, pr, rc in config.FINAL_RESULTS]),
        use_container_width=True, hide_index=True)

    st.subheader("Emoji impact (ΔF1 = with emoji − without emoji)")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pd.DataFrame(
            [{"Model": config.MODELS[m]["label"],
              "ΔMicro-F1": f"{a:+.2f}", "ΔMacro-F1": f"{b:+.2f}",
              "Direction": "helps" if a > 0 else "hurts"}
             for m, (a, b) in config.EMOJI_IMPACT.items()]),
            use_container_width=True, hide_index=True)
        st.info(
            "The two architectures disagree in direction. Because both effects are "
            "within the range that run-to-run variance could produce, they are "
            "indicative rather than confirmed.")
    with c2:
        fig = go.Figure()
        ms = list(config.EMOJI_IMPACT)
        fig.add_bar(x=[config.MODELS[m]["label"] for m in ms],
                    y=[config.EMOJI_IMPACT[m][0] for m in ms],
                    marker_color=["#B3512F" if config.EMOJI_IMPACT[m][0] < 0
                                  else "#3B6EA5" for m in ms])
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(height=300, yaxis_title="ΔMicro-F1 (points)",
                          margin=dict(l=10, r=10, t=30, b=10),
                          title="Aggregate emoji effect")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-emotion effect")
    st.caption(
        "The aggregate figures above conceal much larger movements at the label "
        "level — this is the study's most robust finding.")
    rows = []
    for e, (ln, ly, bn, by) in config.PER_EMOTION_F1.items():
        dl, db = config.PER_EMOTION_DELTA[e]
        rows.append({"Emotion": e, "BiLSTM without": ln, "BiLSTM with": ly,
                     "Δ BiLSTM": f"{dl:+.2f}", "Transf. without": bn,
                     "Transf. with": by, "Δ Transformer": f"{db:+.2f}",
                     "_sort": (dl + db) / 2})
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    order = df["Emotion"].tolist()
    fig.add_bar(name="BiLSTM", x=order,
                y=[config.PER_EMOTION_DELTA[e][0] for e in order],
                marker_color="#3B6EA5")
    fig.add_bar(name="Transformer", x=order,
                y=[config.PER_EMOTION_DELTA[e][1] for e in order],
                marker_color="#B3512F")
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(barmode="group", height=400,
                      yaxis_title="ΔF1 (percentage points)",
                      margin=dict(l=10, r=10, t=60, b=10),
                      legend=dict(orientation="h", y=1.10, x=0),
                      title=dict(text="Per-emotion emoji effect", y=0.97))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Threshold-independent ranking quality (mean ROC-AUC)")
    st.dataframe(pd.DataFrame(
        [{"Condition": f"{config.MODELS[m]['label']} — "
                       f"{config.TRACK_LABELS[t].split(' (')[0]}", "Mean AUC": v}
         for (m, t), v in config.MEAN_AUC.items()]),
        use_container_width=True, hide_index=True)

    st.caption(config.DISCLAIMER)


# --------------------------------------------------------------------------
# page 3 — feedback
# --------------------------------------------------------------------------
def page_feedback():
    st.title("Evaluation feedback")
    st.caption(
        "This page collects the user-evaluation evidence required for the "
        "project's Evaluation chapter. Responses are appended to "
        "`app/feedback/user_feedback.csv`.")

    if "result" not in st.session_state:
        st.info("Classify a comment on the first page, then return here.")
        return

    res = st.session_state.result
    st.markdown("**Text evaluated**")
    st.code(st.session_state.result_text, language=None)
    c1, c2 = st.columns(2)
    c1.markdown("**Without emoji:** " +
                (", ".join(res["no_emoji"]["labels"]) or "none"))
    c2.markdown("**With emoji:** " +
                (", ".join(res["with_emoji"]["labels"]) or "none"))

    with st.form("feedback"):
        role = st.selectbox(
            "Your background",
            ["Student", "Researcher / academic", "Industry practitioner",
             "General social media user", "Prefer not to say"])
        acc = st.slider(
            "How well do the predicted emotions match your own reading of the text?",
            1, 5, 3, help="1 = not at all, 5 = matches closely")
        better = st.radio(
            "Which track produced the better prediction?",
            ["Without emoji", "With emoji", "About the same", "Both were poor"],
            horizontal=True)
        useful = st.slider(
            "How useful is seeing both tracks side by side?", 1, 5, 3)
        comment = st.text_area("Any comments (optional)", height=90)
        submitted = st.form_submit_button("Submit feedback", type="primary")

    if submitted:
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "text": st.session_state.result_text,
            "model": config.MODELS[st.session_state.result_model]["label"],
            "labels_no_emoji": "|".join(res["no_emoji"]["labels"]),
            "labels_with_emoji": "|".join(res["with_emoji"]["labels"]),
            "prediction_changed": res["diff"]["changed"],
            "accuracy_rating": acc,
            "better_track": better,
            "usefulness_rating": useful,
            "comment": comment.replace("\n", " ").strip(),
        }
        config.FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(
            config.FEEDBACK_CSV, mode="a", index=False,
            header=not config.FEEDBACK_CSV.exists(), encoding="utf-8")
        st.success("Thank you — your response has been recorded.")

    if config.FEEDBACK_CSV.exists():
        st.divider()
        fb = pd.read_csv(config.FEEDBACK_CSV)
        st.subheader(f"Responses collected: {len(fb)}")
        m = st.columns(3)
        m[0].metric("Mean accuracy rating", f"{fb['accuracy_rating'].mean():.2f} / 5")
        m[1].metric("Mean usefulness rating", f"{fb['usefulness_rating'].mean():.2f} / 5")
        m[2].metric("Predictions that changed",
                    f"{100 * fb['prediction_changed'].mean():.0f}%")
        st.dataframe(fb.tail(15), use_container_width=True, hide_index=True)
        st.download_button("Download all feedback (CSV)",
                           fb.to_csv(index=False).encode("utf-8"),
                           "user_feedback.csv", "text/csv")


# --------------------------------------------------------------------------
def main():
    st.sidebar.title("🎭 Emoji Impact")
    st.sidebar.caption(
        "Multi-label emotion classification of tweets\n\n"
        "MSc Computing Research Project")
    page = st.sidebar.radio(
        "Page", ["Classify", "Model performance", "Evaluation feedback"])
    thresholds = status_sidebar()
    st.sidebar.divider()
    st.sidebar.caption(
        "Track A removes emoji. Track B converts them to their Unicode "
        "descriptions. Everything else in the pipeline is identical.")

    if page == "Model performance":
        page_performance()
        return

    if not inference.all_models_present():
        setup_help()
        return

    if page == "Classify":
        page_predict(thresholds)
    else:
        page_feedback()


if __name__ == "__main__":
    main()
