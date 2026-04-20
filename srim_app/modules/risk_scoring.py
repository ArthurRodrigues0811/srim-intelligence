from __future__ import annotations

import pandas as pd


def calculate_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted supplier risk score and business risk class.
    Business rule:
    - 40% environmental
    - 30% labor
    - 30% financial
    """
    scored_df = df.copy()
    scored_df["risk_score"] = (
        scored_df["risk_environmental"] * 0.4
        + scored_df["risk_labor"] * 0.3
        + scored_df["risk_financial"] * 0.3
    ).round(2)
    scored_df["risk_class"] = scored_df["risk_score"].apply(classify_risk)
    return scored_df


def classify_risk(score: float) -> str:
    """Classify risk level for executive communication."""
    if score <= 30:
        return "Baixo"
    if score <= 70:
        return "Médio"
    return "Alto"


def get_top_risky_suppliers(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the top N suppliers by highest risk score."""
    return df.sort_values("risk_score", ascending=False).head(n)
