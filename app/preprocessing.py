# -*- coding: utf-8 -*-
"""Faithful port of the training pipeline (krish.ipynb, cell 15).

Any drift between this file and the notebook silently invalidates every
prediction, because the models were fitted on the notebook's exact output.
Do not "improve" these functions.
"""
import re
import string

import emoji
import demoji
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

_READY = False


def ensure_nltk():
    """Download the corpora the pipeline needs. Safe to call repeatedly."""
    global _READY
    if _READY:
        return
    for pkg in ("punkt", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _READY = True


ensure_nltk()

tweet_tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
lemmatizer = WordNetLemmatizer()

slang_map = {
    "u": "you", "ur": "your", "r": "are", "lol": "laugh", "omg": "surprise",
    "im": "i am", "cant": "cannot", "dont": "do not", "gonna": "going to",
}


def clean_text(text):
    """Stage 1 — strip URLs, @mentions and hashtag symbols."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def handle_emoji_no(text):
    """Stage 2a — Track A: delete every emoji."""
    return emoji.replace_emoji(str(text), replace="")


def handle_emoji_yes(text):
    """Stage 2b — Track B: replace each emoji with its Unicode description."""
    return demoji.replace_with_desc(str(text), sep=" ")


def normalize_text(text):
    """Stage 3 — lowercase and strip non-alphanumerics."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_text(text):
    """Stage 4 — TweetTokenizer, dropping bare punctuation tokens."""
    tokens = tweet_tokenizer.tokenize(text)
    return [t for t in tokens if t not in string.punctuation]


def lemmatize_tokens(tokens):
    """Stage 5 — slang expansion then WordNet lemmatisation."""
    tokens = [slang_map.get(t, t) for t in tokens]
    return [lemmatizer.lemmatize(t) for t in tokens]


def build_text(text, emoji_mode="no"):
    """Full pipeline. emoji_mode='no' -> Track A, 'yes' -> Track B."""
    text = clean_text(text)
    text = handle_emoji_no(text) if emoji_mode == "no" else handle_emoji_yes(text)
    text = normalize_text(text)
    tokens = lemmatize_tokens(tokenize_text(text))
    return " ".join(tokens)


def build_both(text):
    """Return (track_a_text, track_b_text) for a single raw input."""
    return build_text(text, "no"), build_text(text, "yes")


def pipeline_stages(text):
    """Stage-by-stage trace for both tracks, used by the UI to show the fork."""
    raw = str(text)
    s1 = clean_text(raw)
    a2, b2 = handle_emoji_no(s1), handle_emoji_yes(s1)
    a3, b3 = normalize_text(a2), normalize_text(b2)
    a4, b4 = tokenize_text(a3), tokenize_text(b3)
    a5, b5 = " ".join(lemmatize_tokens(a4)), " ".join(lemmatize_tokens(b4))
    return [
        ("Input", raw, raw),
        ("1. Noise removal (URLs, @mentions, #)", s1, s1),
        ("2. Emoji handling  ← the fork", a2, b2),
        ("3. Normalisation", a3, b3),
        ("4. Tokenisation", " ".join(a4), " ".join(b4)),
        ("5. Slang + lemmatisation (model input)", a5, b5),
    ]


def has_emoji(text):
    return bool(emoji.emoji_count(str(text)))


def emoji_inventory(text):
    """List of (emoji, description) found in the text, for the UI."""
    found = demoji.findall(str(text))
    return [(k, v) for k, v in found.items()]
