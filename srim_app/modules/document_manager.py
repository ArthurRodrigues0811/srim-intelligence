from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

from PyPDF2 import PdfReader


def read_uploaded_file(file_name: str, file_bytes: bytes) -> dict[str, object]:
    """
    Simulate document extraction for PDF/image files.
    For PDF, returns basic metadata from page count when possible.
    """
    extension = file_name.lower().split(".")[-1]
    base_data: dict[str, object] = {
        "file_name": file_name,
        "file_type": extension,
        "pages": None,
        "notes": "Leitura simulada para MVP.",
    }

    if extension == "pdf":
        try:
            reader = PdfReader(BytesIO(file_bytes))
            base_data["pages"] = len(reader.pages)
            base_data["notes"] = f"PDF processado com {base_data['pages']} página(s)."
        except Exception:
            base_data["notes"] = "Falha na leitura do PDF. Mantendo fluxo mockado."

    return base_data


def classify_document_status(file_name: str) -> dict[str, object]:
    """
    Create deterministic status and fictitious expiration date.
    This avoids randomness and keeps executive demo repeatable.
    """
    checksum = sum(ord(char) for char in file_name)
    selector = checksum % 3
    today = date.today()

    if selector == 0:
        expiry = today + timedelta(days=120)
        status = "OK"
    elif selector == 1:
        expiry = today + timedelta(days=20)
        status = "A VENCER"
    else:
        expiry = today - timedelta(days=15)
        status = "VENCIDO"

    return {"status": status, "valid_until": expiry.strftime("%d/%m/%Y")}
