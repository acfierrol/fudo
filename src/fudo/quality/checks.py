"""Data quality checks for externally received datasets."""

import pandas as pd


def completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing values per column."""
    total = len(df)
    missing = df.isnull().sum()
    return pd.DataFrame({
        "missing_count": missing,
        "missing_pct": (missing / total * 100).round(2),
        "dtype": df.dtypes,
    }).sort_values("missing_pct", ascending=False)


def duplicate_report(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    """Report duplicate rows."""
    n_dupes = df.duplicated(subset=subset).sum()
    return {
        "total_rows": len(df),
        "duplicate_rows": int(n_dupes),
        "duplicate_pct": round(n_dupes / len(df) * 100, 2) if len(df) > 0 else 0.0,
        "subset_columns": subset,
    }


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report column types, unique counts, and sample values."""
    records = []
    for col in df.columns:
        records.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "n_missing": int(df[col].isnull().sum()),
            "sample_values": df[col].dropna().head(3).tolist(),
        })
    return pd.DataFrame(records)


def outlier_report(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Flag numeric columns with values beyond z_threshold standard deviations."""
    numeric = df.select_dtypes(include="number")
    records = []
    for col in numeric.columns:
        mean = numeric[col].mean()
        std = numeric[col].std()
        if std == 0:
            continue
        z_scores = ((numeric[col] - mean) / std).abs()
        n_outliers = int((z_scores > z_threshold).sum())
        records.append({
            "column": col,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "n_outliers": n_outliers,
            "outlier_pct": round(n_outliers / len(df) * 100, 2),
        })
    return pd.DataFrame(records)


def run_all_checks(df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    """Run the full quality suite and return all reports."""
    return {
        "completeness": completeness_report(df),
        "duplicates": duplicate_report(df),
        "schema": schema_report(df),
        "outliers": outlier_report(df),
    }
