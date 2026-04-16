"""EDA profiling utilities."""

import pandas as pd


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Extended describe for numeric columns."""
    numeric = df.select_dtypes(include="number")
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary["skew"] = numeric.skew()
    summary["kurtosis"] = numeric.kurtosis()
    summary["n_zeros"] = (numeric == 0).sum()
    return summary


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Value counts for categorical / object columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return {
        col: df[col].value_counts().head(top_n).to_frame("count")
        for col in cat_cols
    }


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix for numeric columns."""
    return df.select_dtypes(include="number").corr(method=method)


def temporal_summary(df: pd.DataFrame) -> dict[str, dict]:
    """Summary stats for datetime columns."""
    dt_cols = df.select_dtypes(include="datetime").columns
    result = {}
    for col in dt_cols:
        result[col] = {
            "min": str(df[col].min()),
            "max": str(df[col].max()),
            "n_missing": int(df[col].isnull().sum()),
            "n_unique": df[col].nunique(),
        }
    return result
