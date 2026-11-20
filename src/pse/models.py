import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .validation import walk_forward_splits

# Fixed in advance, no tuning on test data.
TFIDF = dict(min_df=5, max_df=0.9, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")


def _ohe():
    return OneHotEncoder(handle_unknown="ignore")


def make_model(name, cat_cols, num_cols=(), text_col="text_clean"):
    """B0 / S: no text. T: categorical + TF-IDF logistic. G: gradient boosting + SVD of TF-IDF."""
    lr = LogisticRegression(C=1.0, max_iter=2000)
    if name in ("B0", "S"):
        parts = [("cat", _ohe(), list(cat_cols))]
        if name == "S" and num_cols:
            from sklearn.impute import SimpleImputer
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True))]),
                          list(num_cols)))
        return Pipeline([("prep", ColumnTransformer(parts)), ("clf", lr)])
    if name == "T":
        prep = ColumnTransformer([("cat", _ohe(), list(cat_cols)),
                                  ("txt", TfidfVectorizer(**TFIDF), text_col)])
        return Pipeline([("prep", prep), ("clf", lr)])
    if name == "G":
        from sklearn.preprocessing import OrdinalEncoder
        prep = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(cat_cols)),
            ("num", "passthrough", list(num_cols)),
            ("txt", Pipeline([("tfidf", TfidfVectorizer(**TFIDF)),
                              ("svd", TruncatedSVD(20, random_state=0))]), text_col)])
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=100, learning_rate=0.05,
                                            random_state=0)
        return Pipeline([("prep", prep), ("clf", gb)])
    raise ValueError(name)


def run_walk_forward(df, specs, y_col="y", t0_col="t0", end_col="label_end", train_months=12):
    """specs: {model_name: (kind, cat_cols, num_cols)}. Returns pooled out-of-sample predictions."""
    df = df.reset_index(drop=True)
    preds = []
    for month, tr, te in walk_forward_splits(df[t0_col], df[end_col], train_months):
        ytr = df.loc[tr, y_col]
        if ytr.nunique() < 2:
            continue
        block = df.loc[te, ["edge_no", y_col]].copy()
        block["fold"] = str(month)
        block["majority"] = int(ytr.mean() >= 0.5)
        for name, (kind, cat, num) in specs.items():
            m = make_model(kind, cat, num)
            m.fit(df.loc[tr], ytr)
            block[f"p_{name}"] = m.predict_proba(df.loc[te])[:, 1]
        preds.append(block)
    return pd.concat(preds, ignore_index=True)


def score(preds, names, y_col="y"):
    y = preds[y_col]
    rows = [{"model": "majority", "auc": 0.5, "accuracy": accuracy_score(y, preds["majority"])}]
    for n in names:
        p = preds[f"p_{n}"]
        rows.append({"model": n, "auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p >= 0.5)})
    return pd.DataFrame(rows).assign(n_test=len(y), base_rate=y.mean())
