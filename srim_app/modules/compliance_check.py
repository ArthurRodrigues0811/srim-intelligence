from __future__ import annotations

import re


def _extract_last_digit(cnpj: str) -> int:
    digits = re.sub(r"\D", "", str(cnpj))
    if not digits:
        return 0
    return int(digits[-1])


def check_compliance(cnpj: str) -> dict[str, object]:
    """
    Mock compliance validation based on CNPJ last digit.
    This keeps behavior deterministic for demos and tests.
    """
    last_digit = _extract_last_digit(cnpj)

    receita_status = "ATIVA" if last_digit % 2 == 0 else "INATIVA"
    ibama_flag = last_digit in {1, 3, 5, 7, 9}
    trabalho_escravo_flag = last_digit in {0, 6, 8}

    return {
        "receita_status": receita_status,
        "ibama_flag": ibama_flag,
        "trabalho_escravo_flag": trabalho_escravo_flag,
    }
