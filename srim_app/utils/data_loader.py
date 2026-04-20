from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_UPLOAD_COLUMNS = {
    "supplier_name",
    "risk_environmental",
    "risk_labor",
    "risk_financial",
    "otif",
}


def load_suppliers_data(file_path: str | Path) -> pd.DataFrame:
    """Load supplier dataset keeping business identifiers as strings."""
    data = pd.read_csv(file_path, dtype={"cnpj": "string", "supplier_id": "string"})
    data["last_audit_date"] = pd.to_datetime(data["last_audit_date"], errors="coerce")
    data["document_status"] = data["document_status"].astype("string")
    return data


def load_suppliers_from_upload(uploaded_file) -> pd.DataFrame:
    """Load uploaded CSV with supplier identifiers as strings when present."""
    data = pd.read_csv(uploaded_file, dtype={"cnpj": "string", "supplier_id": "string"})
    if "last_audit_date" in data.columns:
        data["last_audit_date"] = pd.to_datetime(data["last_audit_date"], errors="coerce")
    if "document_status" in data.columns:
        data["document_status"] = data["document_status"].astype("string")
    return data


def validate_uploaded_columns(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate mandatory columns for uploaded CSV."""
    missing = sorted(list(REQUIRED_UPLOAD_COLUMNS - set(df.columns)))
    return (len(missing) == 0, missing)


def format_brl(value: float) -> str:
    """Format number to Brazilian Real currency pattern."""
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"
