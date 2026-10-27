import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = pd.read_csv(d["path"])
    df = df.rename(columns={d["query_column"]: "query", d["label_column"]: "label"})
    df = df[["query", "label"]].dropna()
    df["query"] = df["query"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["query"] != ""].drop_duplicates(subset="query")

    unknown = set(df["label"]) - set(cfg["labels"])
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}")
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, cfg: dict):
    d = cfg["data"]
    return train_test_split(
        df["query"], df["label"],
        test_size=d["test_size"], random_state=d["seed"], stratify=df["label"],
    )
