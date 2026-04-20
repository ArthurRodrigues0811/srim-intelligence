from __future__ import annotations

import pandas as pd


def _format_brl(value: float) -> str:
    formatted = f"{value:,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def calculate_financial_impact(row: pd.Series) -> float:
    """
    Estimate financial loss when supplier is high-risk and underperforming.
    Rule:
    - risk_score > 70
    - otif < 85
    """
    if row["risk_score"] > 70 and row["otif"] < 85:
        return float((100 - row["otif"]) * 50000)
    return 0.0


def enrich_with_financial_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Apply financial impact logic and strategic priority segmentation."""
    enriched_df = df.copy()
    enriched_df["impacto_financeiro_estimado"] = enriched_df.apply(
        calculate_financial_impact, axis=1
    )
    enriched_df["prioridade_acao"] = enriched_df["impacto_financeiro_estimado"].apply(
        classify_action_priority
    )
    return enriched_df


def classify_action_priority(impact: float) -> str:
    if impact > 1_000_000:
        return "CRÍTICO"
    if 300_000 <= impact <= 1_000_000:
        return "ALTO"
    return "MÉDIO"


def create_performance_insight(row: pd.Series) -> str:
    """Generate tactical insight by combining risk and service level."""
    risk_score = row["risk_score"]
    impact = row["impacto_financeiro_estimado"]
    otif = row["otif"]

    if risk_score > 70 and impact > 1_000_000:
        return "Risco crítico imediato"
    if 31 <= risk_score <= 70 and impact >= 300_000:
        return "Monitoramento prioritário"
    if risk_score > 70 and otif >= 85:
        return "Alto risco com operação estável"
    if risk_score <= 30 and otif >= 90:
        return "Baixo risco e alta confiabilidade"
    return "Operação em monitoramento regular"


def get_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return table used by performance module with executive labels."""
    output = df.copy()
    output["insight"] = output.apply(create_performance_insight, axis=1)
    columns = [
        "supplier_id",
        "supplier_name",
        "risk_score",
        "risk_class",
        "otif",
        "impacto_financeiro_estimado",
        "prioridade_acao",
        "insight",
    ]
    return output[columns].sort_values("impacto_financeiro_estimado", ascending=False)


def simulate_scenarios(df: pd.DataFrame) -> dict[str, float]:
    """
    Simulate financial exposure for base, improvement, and critical scenarios.
    - Improvement: risk -20%, OTIF +10 (cap 100)
    - Critical: risk +15%, OTIF -10 (floor 0)
    """
    base_df = df.copy()

    improvement_df = df.copy()
    improvement_df["risk_score"] = (improvement_df["risk_score"] * 0.8).clip(lower=0, upper=100)
    improvement_df["otif"] = (improvement_df["otif"] + 10).clip(upper=100)

    critical_df = df.copy()
    critical_df["risk_score"] = (critical_df["risk_score"] * 1.15).clip(lower=0, upper=100)
    critical_df["otif"] = (critical_df["otif"] - 10).clip(lower=0)

    impact_base = base_df.apply(calculate_financial_impact, axis=1).sum()
    impact_melhoria = improvement_df.apply(calculate_financial_impact, axis=1).sum()
    impact_critico = critical_df.apply(calculate_financial_impact, axis=1).sum()

    return {
        "impacto_base": float(impact_base),
        "impacto_melhoria": float(impact_melhoria),
        "impacto_critico": float(impact_critico),
    }


def generate_executive_insights(df: pd.DataFrame) -> list[str]:
    """Generate dashboard-ready strategic insights for executive audience."""
    total_impact = float(df["impacto_financeiro_estimado"].sum())
    if total_impact == 0:
        return ["Não há exposição financeira relevante com as regras atuais."]

    ordered = df.sort_values("impacto_financeiro_estimado", ascending=False)
    top_20_count = max(1, int(len(df) * 0.2))
    top_20_impact = float(ordered.head(top_20_count)["impacto_financeiro_estimado"].sum())
    top_20_share = (top_20_impact / total_impact) * 100

    top_3_impact = float(ordered.head(3)["impacto_financeiro_estimado"].sum())

    scenario = simulate_scenarios(df)
    otif_upside = scenario["impacto_base"] - scenario["impacto_melhoria"]

    return [
        f"{top_20_share:.0f}% do risco financeiro está concentrado em 20% dos fornecedores.",
        f"Melhorar OTIF em 10 pontos reduz exposição em {_format_brl(otif_upside)}.",
        f"Top 3 fornecedores representam {_format_brl(top_3_impact)} de risco potencial.",
    ]


def generate_ai_strategic_analysis(df: pd.DataFrame) -> list[str]:
    """Create 3 executive paragraphs for AI strategic section."""
    if df.empty:
        return [
            "Não há dados suficientes para identificar gargalos de risco nesta visualização.",
            "Sem base carregada, não foi possível identificar oportunidades de otimização operacional.",
            "Recomendação de board: consolidar dados mínimos para habilitar decisões baseadas em evidências.",
        ]

    highest_impact_row = df.sort_values("impacto_financeiro_estimado", ascending=False).iloc[0]
    opportunity_df = df[
        (df["risk_score"] <= 30) & (df["otif"] >= 75) & (df["otif"] < 90)
    ].sort_values("otif", ascending=True)
    total_impact = float(df["impacto_financeiro_estimado"].sum())
    high_risk_share = float((df["risk_score"] > 70).mean() * 100)

    bottleneck = (
        f"Gargalo de Risco: {highest_impact_row['supplier_name']} concentra "
        f"{_format_brl(float(highest_impact_row['impacto_financeiro_estimado']))} de exposição, "
        "combinando risco elevado e vulnerabilidade operacional."
    )

    if opportunity_df.empty:
        optimization = (
            "Oportunidade de Otimização: os fornecedores de baixo risco já operam com OTIF robusto, "
            "indicando espaço maior para ganhos via prevenção em parceiros de maior criticidade."
        )
    else:
        candidate = opportunity_df.iloc[0]
        optimization = (
            f"Oportunidade de Otimização: {candidate['supplier_name']} apresenta risco baixo com OTIF em "
            f"{candidate['otif']:.1f}, perfil favorável para capturar eficiência operacional com baixo trade-off."
        )

    if total_impact > 5_000_000 or high_risk_share >= 35:
        board = (
            "Recomendação de Board: elevar o tema para comitê executivo e priorizar plano tático imediato "
            "nos fornecedores de maior impacto financeiro."
        )
    elif total_impact > 0:
        board = (
            "Recomendação de Board: manter governança ativa com foco em alavancas de OTIF para reduzir "
            "exposição sem comprometer continuidade de abastecimento."
        )
    else:
        board = (
            "Recomendação de Board: a base está saudável no cenário atual; preservar disciplina de compliance "
            "e monitoramento preventivo para sustentar o patamar."
        )

    return [bottleneck, optimization, board]
