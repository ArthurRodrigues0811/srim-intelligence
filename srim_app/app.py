from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.compliance_check import check_compliance
from modules.document_manager import classify_document_status, read_uploaded_file
from modules.performance_analysis import (
    enrich_with_financial_impact,
    generate_ai_strategic_analysis,
    generate_executive_insights,
    get_performance_table,
    simulate_scenarios,
)
from modules.risk_scoring import calculate_risk_score, get_top_risky_suppliers
from utils.data_loader import (
    format_brl,
    load_suppliers_data,
    load_suppliers_from_upload,
    validate_uploaded_columns,
)


st.set_page_config(
    page_title="SRIM | Supply Risk & Integrity Monitor",
    page_icon=":bar_chart:",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "suppliers_mock.csv"
STYLE_FILE = BASE_DIR / "assets" / "styles.css"


def load_css() -> None:
    if STYLE_FILE.exists():
        st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data
def get_base_dataframe() -> pd.DataFrame:
    df = load_suppliers_data(DATA_FILE)
    df = calculate_risk_score(df)
    df = enrich_with_financial_impact(df)
    return df


def render_risk_badge(risk_class: str) -> str:
    class_map = {"Baixo": "badge-baixo", "Médio": "badge-medio", "Alto": "badge-alto"}
    css_class = class_map.get(risk_class, "badge-medio")
    return f"<span class='srim-badge {css_class}'>{risk_class.upper()}</span>"


def render_kpi_card(title: str, value: str, icon: str) -> None:
    st.markdown(
        (
            "<div class='kpi-card'>"
            f"<div class='kpi-head'><span class='kpi-icon'>{icon}</span><span>{title}</span></div>"
            f"<div class='kpi-value'>{value}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def prepare_analysis_dataframe(base_df: pd.DataFrame) -> pd.DataFrame:
    prepared = calculate_risk_score(base_df)
    prepared = enrich_with_financial_impact(prepared)
    return prepared


def apply_sidebar_filters(df: pd.DataFrame, selected_sectors: list[str], selected_risk_classes: list[str]) -> pd.DataFrame:
    filtered = df.copy()
    if selected_sectors and "sector" in filtered.columns:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    if selected_risk_classes and "risk_class" in filtered.columns:
        filtered = filtered[filtered["risk_class"].isin(selected_risk_classes)]
    return filtered


def render_dashboard(df: pd.DataFrame) -> None:
    st.title("SRIM | Supply Risk & Integrity Monitor")
    st.markdown("Plataforma executiva para monitoramento de Governança, Compliance e Performance de fornecedores.")

    risk_mean = float(df["risk_score"].mean())
    total_financial_risk = float(df["impacto_financeiro_estimado"].sum())
    scenarios = simulate_scenarios(df)
    upside = scenarios["impacto_base"] - scenarios["impacto_melhoria"]
    downside = scenarios["impacto_critico"] - scenarios["impacto_base"]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_kpi_card(
            "Risk Score Médio",
            f"{risk_mean:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
            "🛡",
        )
    with kpi_col2:
        render_kpi_card("Risco Financeiro Total", format_brl(total_financial_risk), "$")
    with kpi_col3:
        render_kpi_card("Exposição Atual", format_brl(scenarios["impacto_base"]), "◉")
    with kpi_col4:
        render_kpi_card("Upside Potencial", format_brl(upside), "↘")

    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        render_kpi_card("Downside Risk", format_brl(downside), "↗")
    main_risk_class = df["risk_class"].value_counts().idxmax()
    sub_col2.markdown(
        f"<div class='srim-card'><strong>Classe de risco predominante</strong><br>{render_risk_badge(main_risk_class)}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Análise Estratégica da IA")
    for paragraph in generate_ai_strategic_analysis(df):
        st.markdown(f"<div class='srim-card'><p>{paragraph}</p></div>", unsafe_allow_html=True)

    st.markdown("### Exposição Executiva por Cenário")
    c1, c2, c3 = st.columns(3)
    c1.metric("Base", format_brl(scenarios["impacto_base"]))
    c2.metric("Melhoria", format_brl(scenarios["impacto_melhoria"]))
    c3.metric("Crítico", format_brl(scenarios["impacto_critico"]))

    waterfall = go.Figure(
        go.Waterfall(
            name="Cenários Financeiros",
            orientation="v",
            measure=["absolute", "relative", "relative"],
            x=["Base", "Downside (Crítico)", "Upside (Melhoria)"],
            y=[scenarios["impacto_base"], downside, -upside],
            connector={"line": {"color": "#2F2F2F"}},
            decreasing={"marker": {"color": "#C5A46D"}},
            increasing={"marker": {"color": "#0B1F3B"}},
            totals={"marker": {"color": "#2F2F2F"}},
        )
    )
    waterfall.update_layout(template="plotly_white", height=380, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(waterfall, use_container_width=True)

    st.markdown("### Top 5 Fornecedores Mais Arriscados")
    top_risky = get_top_risky_suppliers(df, 5).copy()
    top_risky["impacto_financeiro_estimado"] = top_risky["impacto_financeiro_estimado"].apply(format_brl)
    st.dataframe(
        top_risky[
            [
                "supplier_id",
                "supplier_name",
                "sector",
                "risk_score",
                "risk_class",
                "otif",
                "impacto_financeiro_estimado",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Risk Score vs OTIF")
    scatter = px.scatter(
        df,
        x="risk_score",
        y="otif",
        color="risk_class",
        size="impacto_financeiro_estimado",
        hover_data=["supplier_name", "sector"],
        color_discrete_map={"Baixo": "#426A76", "Médio": "#6A6A6A", "Alto": "#A63A3A"},
        template="plotly_white",
    )
    scatter.update_layout(height=420)
    st.plotly_chart(scatter, use_container_width=True)

    st.markdown("### Risk Score vs Impacto Financeiro")
    impact_scatter = px.scatter(
        df,
        x="risk_score",
        y="impacto_financeiro_estimado",
        color="prioridade_acao",
        size="impacto_financeiro_estimado",
        hover_data=["supplier_name", "otif"],
        color_discrete_map={"CRÍTICO": "#A63A3A", "ALTO": "#6A6A6A", "MÉDIO": "#426A76"},
        template="plotly_white",
    )
    impact_scatter.update_layout(height=420, yaxis_title="Impacto Financeiro Estimado")
    st.plotly_chart(impact_scatter, use_container_width=True)

    st.markdown("### Priorização por Impacto Financeiro")
    top_impact = df.sort_values("impacto_financeiro_estimado", ascending=False).head(5).copy()
    top_impact["impacto_formatado"] = top_impact["impacto_financeiro_estimado"].apply(format_brl)
    st.dataframe(
        top_impact[["supplier_id", "supplier_name", "risk_score", "otif", "impacto_formatado", "prioridade_acao"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Insights Executivos")
    for insight in generate_executive_insights(df):
        st.markdown(f"- {insight}")

    st.markdown(
        f"**Fornecedores críticos representam {format_brl(total_financial_risk)} em risco potencial na base atual.**"
    )


def render_compliance_check() -> None:
    st.title("Compliance Check")
    st.markdown("Simulação de consulta por CNPJ para triagem de conformidade regulatória.")
    cnpj = st.text_input("Digite o CNPJ do fornecedor", placeholder="00.000.000/0001-00")

    if cnpj:
        result = check_compliance(cnpj)
        col1, col2, col3 = st.columns(3)

        col1.metric("Receita Federal", result["receita_status"])
        col2.metric("Flag IBAMA", "Sim" if result["ibama_flag"] else "Não")
        col3.metric("Flag Trabalho Escravo", "Sim" if result["trabalho_escravo_flag"] else "Não")

        status_text = (
            "Fornecedor com sinais de atenção em compliance."
            if result["ibama_flag"] or result["trabalho_escravo_flag"] or result["receita_status"] == "INATIVA"
            else "Fornecedor sem alertas críticos no screening mockado."
        )
        st.markdown(f"<div class='srim-card'>{status_text}</div>", unsafe_allow_html=True)


def render_documents_module() -> None:
    st.title("Gestão de Documentos")
    st.markdown("Upload e classificação de documentos de fornecedores (simulação para MVP).")

    uploaded_file = st.file_uploader("Envie um PDF ou imagem", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        file_data = read_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
        status_data = classify_document_status(uploaded_file.name)

        st.markdown("### Resultado da Leitura")
        st.write(f"**Arquivo:** {file_data['file_name']}")
        st.write(f"**Tipo:** {file_data['file_type'].upper()}")
        st.write(f"**Observação:** {file_data['notes']}")
        if file_data.get("pages") is not None:
            st.write(f"**Páginas detectadas:** {file_data['pages']}")

        badge_html = render_risk_badge("Alto" if status_data["status"] == "VENCIDO" else "Médio" if status_data["status"] == "A VENCER" else "Baixo")
        st.markdown(
            (
                "<div class='srim-card'>"
                f"<strong>Status documental:</strong> {status_data['status']}<br>"
                f"<strong>Validade simulada:</strong> {status_data['valid_until']}<br>"
                f"{badge_html}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_performance_module(df: pd.DataFrame) -> None:
    st.title("Análise de Performance")
    st.markdown("Correlação entre performance logística (OTIF), risco e exposição financeira.")

    performance_table = get_performance_table(df).copy()
    performance_table["impacto_financeiro_estimado_fmt"] = performance_table["impacto_financeiro_estimado"].apply(format_brl)

    st.dataframe(
        performance_table[
            [
                "supplier_id",
                "supplier_name",
                "risk_score",
                "risk_class",
                "otif",
                "impacto_financeiro_estimado_fmt",
                "prioridade_acao",
                "insight",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    fig = px.scatter(
        performance_table,
        x="otif",
        y="risk_score",
        color="prioridade_acao",
        size="impacto_financeiro_estimado",
        hover_data=["supplier_name", "insight"],
        color_discrete_map={"CRÍTICO": "#A63A3A", "ALTO": "#6A6A6A", "MÉDIO": "#426A76"},
        template="plotly_white",
    )
    fig.update_layout(height=420, xaxis_title="OTIF", yaxis_title="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Insights Automáticos")
    for insight in performance_table["insight"].value_counts().index:
        st.markdown(f"- {insight}")


def main() -> None:
    load_css()
    st.sidebar.title("Navegação")
    uploaded_csv = st.sidebar.file_uploader("Upload de base de fornecedores (CSV)", type=["csv"])

    data_source = "mock"
    if uploaded_csv is not None:
        try:
            uploaded_df = load_suppliers_from_upload(uploaded_csv)
            is_valid, missing_columns = validate_uploaded_columns(uploaded_df)
            if is_valid:
                data_source = "upload"
                df = prepare_analysis_dataframe(uploaded_df)
                st.sidebar.success("CSV carregado com sucesso.")
            else:
                st.sidebar.warning(
                    "CSV com colunas ausentes. Campos obrigatórios: "
                    + ", ".join(missing_columns)
                    + ". Usando base mockada."
                )
                df = get_base_dataframe()
        except Exception:
            st.sidebar.warning("Não foi possível ler o CSV enviado. Usando base mockada.")
            df = get_base_dataframe()
    else:
        df = get_base_dataframe()

    st.sidebar.markdown("<div class='sidebar-spacer'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<p class='sidebar-filter-label'>Setor</p>", unsafe_allow_html=True)
    sectors = sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []
    risk_classes = ["Alto", "Médio", "Baixo"]

    selected_sectors = st.sidebar.multiselect(
        "Filtrar por Setor",
        options=sectors,
        default=[],
        placeholder="Selecionar...",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("<p class='sidebar-filter-label'>Classe de Risco</p>", unsafe_allow_html=True)
    selected_risk_classes = st.sidebar.multiselect(
        "Filtrar por Classe de Risco",
        options=risk_classes,
        default=[],
        placeholder="Selecionar...",
        label_visibility="collapsed",
    )

    effective_sectors = sectors if not selected_sectors else selected_sectors
    effective_risk_classes = risk_classes if not selected_risk_classes else selected_risk_classes

    st.sidebar.caption(
        "Todos os setores"
        if len(effective_sectors) == len(sectors)
        else f"{len(effective_sectors)} setor(es) selecionado(s)"
    )
    st.sidebar.caption(
        "Todas as classes"
        if len(effective_risk_classes) == len(risk_classes)
        else ", ".join(effective_risk_classes)
    )

    filtered_df = apply_sidebar_filters(df, effective_sectors, effective_risk_classes)
    if filtered_df.empty:
        st.warning("Nenhum fornecedor encontrado para os filtros aplicados. Ajuste os filtros na barra lateral.")
        return

    csv_export = filtered_df.copy()
    if "last_audit_date" in csv_export.columns:
        csv_export["last_audit_date"] = csv_export["last_audit_date"].astype("string")
    st.sidebar.download_button(
        "Baixar Relatório Consolidado (CSV)",
        data=csv_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"srim_relatorio_consolidado_{data_source}.csv",
        mime="text/csv",
    )

    page = st.sidebar.radio(
        "Selecione a visão",
        ["Dashboard", "Compliance Check", "Documentos", "Performance"],
    )

    if page == "Dashboard":
        render_dashboard(filtered_df)
    elif page == "Compliance Check":
        render_compliance_check()
    elif page == "Documentos":
        render_documents_module()
    else:
        render_performance_module(filtered_df)


if __name__ == "__main__":
    main()
