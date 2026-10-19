"""Página: API (FastAPI) e preparação dos dados para LLMs — entregas do TP3."""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from src.llm_prep import montar_corpus, resumo_corpus, salvar_corpus

st.title("🤖 API & preparação para IA")

aba_api, aba_llm = st.tabs(["🔌 API (FastAPI)", "🧠 Preparação para LLMs"])

# ---------------------------------------------------------------- API
with aba_api:
    st.markdown(
        "A API roda **separada** do app Streamlit (outro processo), reaproveitando o mesmo "
        "código de `src/` para calcular o IQAr e ler a tabela oficial. Para subir localmente:"
    )
    st.code("uvicorn api.main:app --reload", language="bash")
    st.caption("Documentação interativa (Swagger) gerada automaticamente em `/docs`.")

    st.markdown("##### Rotas disponíveis")
    st.dataframe(pd.DataFrame([
        {"Método": "GET", "Rota": "/cidades", "Função": "Lista as capitais suportadas (lat/lon)."},
        {"Método": "GET", "Rota": "/iqar/tabela", "Função": "Tabela oficial de faixas do IQAr (filtro opcional por poluente)."},
        {"Método": "GET", "Rota": "/iqar/calcular", "Função": "Calcula o IQAr de uma concentração pontual."},
        {"Método": "GET", "Rota": "/leituras", "Função": "Lista leituras já registradas via POST."},
        {"Método": "POST", "Rota": "/leituras", "Função": "Registra uma nova leitura (sensor, estação, relatório manual)."},
    ]), hide_index=True, width="stretch")

    st.markdown("##### Testar a API a partir desta página")
    url_base = st.text_input("Endereço da API", "http://127.0.0.1:8000")

    c1, c2, c3 = st.columns(3)
    pol = c1.selectbox("Poluente", ["MP2.5", "MP10", "O3", "NO2", "SO2", "CO"], key="api_pol")
    conc = c2.number_input("Concentração", min_value=0.0, value=30.0, key="api_conc")
    if c3.button("GET /iqar/calcular", width="stretch"):
        try:
            resp = requests.get(f"{url_base}/iqar/calcular",
                               params={"poluente": pol, "concentracao": conc}, timeout=5)
            st.json(resp.json())
        except requests.RequestException as erro:
            st.error(f"Não foi possível falar com a API em {url_base} ({erro}). Ela está rodando?")

    with st.form("form_post_leitura"):
        st.caption("POST /leituras — registra uma leitura e já devolve o IQAr calculado")
        f1, f2, f3 = st.columns(3)
        cidade_post = f1.text_input("Cidade", "Rio de Janeiro - RJ")
        pol_post = f2.selectbox("Poluente", ["MP2.5", "MP10", "O3", "NO2", "SO2", "CO"], key="post_pol")
        valor_post = f3.number_input("Valor", min_value=0.0, value=25.0, key="post_valor")
        enviado = st.form_submit_button("Enviar leitura", type="primary")
    if enviado:
        try:
            resp = requests.post(f"{url_base}/leituras",
                                 json={"cidade": cidade_post, "poluente": pol_post, "valor": valor_post},
                                 timeout=5)
            if resp.ok:
                st.success("Leitura registrada.")
                st.json(resp.json())
            else:
                st.error(resp.json())
        except requests.RequestException as erro:
            st.error(f"Não foi possível falar com a API em {url_base} ({erro}). Ela está rodando?")

# ---------------------------------------------------------------- LLM prep
with aba_llm:
    st.markdown(
        "Esta aba **não chama nenhum LLM** — ela organiza os dados já coletados (notícias, "
        "tabela oficial e texto de referência) em um corpus pronto para a próxima entrega, "
        "que vai aplicar técnicas de LLM de verdade sobre esse material."
    )
    st.dataframe(pd.DataFrame([
        {"Cenário de LLM (próxima entrega)": "Resumo automático de texto", "Documentos usados": "notícia"},
        {"Cenário de LLM (próxima entrega)": "Classificação de sentimento", "Documentos usados": "notícia"},
        {"Cenário de LLM (próxima entrega)": "Perguntas e respostas (Q&A)", "Documentos usados": "faixa_oficial, referência"},
        {"Cenário de LLM (próxima entrega)": "Geração de texto (boletim diário)", "Documentos usados": "os três tipos + leitura atual"},
    ]), hide_index=True, width="stretch")

    if st.button("Gerar / atualizar corpus (data/processed/corpus_llm.jsonl)"):
        total = salvar_corpus()
        st.success(f"{total} documentos salvos em `data/processed/corpus_llm.jsonl`.")

    docs = montar_corpus()
    if not docs:
        st.info("Ainda não há dados coletados. Rode os scrapers e volte a esta aba.")
    else:
        st.markdown("##### Documentos por tipo")
        resumo = resumo_corpus(docs)
        st.bar_chart(resumo.set_index("tipo"))

        st.markdown("##### Amostra do corpus")
        tipo_sel = st.selectbox("Tipo", sorted({d.tipo for d in docs}))
        amostra = [d for d in docs if d.tipo == tipo_sel][:5]
        for doc in amostra:
            with st.expander(f"{doc.id} · {doc.fonte}"):
                st.write(doc.texto)
                if doc.metadados:
                    st.caption(doc.metadados)
