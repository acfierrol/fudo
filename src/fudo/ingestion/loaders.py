"""Data loaders for ingesting externally shared datasets."""

from pathlib import Path

import pandas as pd


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file with sensible defaults for external data."""
    return pd.read_csv(path, low_memory=False, **kwargs)


def load_excel(path: Path, sheet_name: str | int = 0, **kwargs) -> pd.DataFrame:
    """Load an Excel file."""
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def load_parquet(path: Path, **kwargs) -> pd.DataFrame:
    """Load a Parquet file."""
    return pd.read_parquet(path, **kwargs)


def load_from_config(path: Path, file_type: str | None = None, **kwargs) -> pd.DataFrame:
    """Auto-detect file type and load accordingly."""
    suffix = file_type or path.suffix.lower().lstrip(".")
    loaders = {
        "csv": load_csv,
        "tsv": lambda p, **kw: load_csv(p, sep="\t", **kw),
        "xlsx": load_excel,
        "xls": load_excel,
        "parquet": load_parquet,
    }
    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path, **kwargs)
