import os
import pandas as pd


def load_listing(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_psei(path):
    """PSEi close indexed by date. Defines the trading calendar."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.drop_duplicates("date").set_index("date")["Price"].astype(float).sort_index()
    return s[s > 0]


def load_directory(path):
    return pd.read_csv(path)


def load_dividends(path):
    df = pd.read_csv(path)
    for c in ["board_date", "record_date", "payment_date"]:
        df[c] = pd.to_datetime(df[c], format="%b %d, %Y", errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def load_price_file(path):
    """Daily OHLC and traded value for one company. Drops zero or missing closes."""
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty or "CLOSE" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").set_index("date").sort_index()
    df = df[df["CLOSE"] > 0]
    return df if len(df) else None


def load_prices(price_dir, cmpy_ids):
    out = {}
    for cid in cmpy_ids:
        p = os.path.join(price_dir, f"{cid}.csv")
        if os.path.exists(p):
            df = load_price_file(p)
            if df is not None:
                out[cid] = df
    return out
