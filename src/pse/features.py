import json
import re
import numpy as np
import pandas as pd
from .returns import parse_amount

MONTHS = ("january february march april may june july august september october november "
          "december jan feb mar apr jun jul aug sep sept oct nov dec").split()
_MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")


def _between(text, start, end):
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    return text[i + len(start): j if j > 0 else None]


def clean_dividend_text(raw, company=""):
    """Keep only the informative free text of a dividend filing.

    Uses the 'Subject of the Disclosure' block and the 'Source of Dividend Payment' to
    'Filed on behalf' block. Company-name words, numbers, dates and currency are removed
    so the model reads wording, not identity.
    """
    if not isinstance(raw, str):
        return ""
    subj = _between(raw, "Subject of the Disclosure", "Type of Securities")
    post = _between(raw, "Source of Dividend Payment", "Filed on behalf")
    s = (subj + " " + post).lower()
    s = re.sub(r"[^a-z\s]", " ", s)            # drops digits, currency symbols, punctuation
    s = _MONTH_RE.sub(" ", s)
    name_words = {w for w in re.sub(r"[^a-z\s]", " ", str(company).lower()).split() if len(w) > 3}
    if name_words:
        s = " ".join(w for w in s.split() if w not in name_words)
    return re.sub(r"\s+", " ", s).strip()


def load_dividend_texts(jsonl_path):
    with open(jsonl_path) as f:
        return pd.DataFrame([json.loads(line) for line in f])[["edge_no", "text"]]


def add_dividend_yield(events, prices, calendar):
    """Declared dividend divided by the close before t0. NaN if unparsed or no price."""
    out = []
    for _, e in events.iterrows():
        amt = parse_amount(e.get("amount_raw"))
        px = prices.get(e["cmpy_id"])
        if px is None or amt != amt:
            out.append(np.nan)
            continue
        i0 = calendar.get_loc(e["t0"])
        close = px["CLOSE"].reindex(calendar).ffill(limit=5)
        prev = close.iloc[i0 - 1] if i0 > 0 else np.nan
        out.append(amt / prev if prev == prev and prev > 0 else np.nan)
    return pd.Series(out, index=events.index)
