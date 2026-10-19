"""Página: upload/download de medições próprias do usuário."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import state
from src.iqar import calcular_indice, categoria

state.iniciar_estado()

st.title("📂 Meus dados")

st.markdown("Envie medições próprias (de estações locais, sensores caseiros ou relatórios) "
            "para comparar com os limites oficiais. O arquivo precisa ter as colunas "
            "`data`, `cidade`, `poluente` e `valor`.")
st.download_button("Baixar modelo de CSV", dl.modelo_upload().to_csv(index=False).encode("utf-8-sig"),
                   "modelo_eter.csv", "text/csv")

arquivos = st.file_uploader("Enviar arquivos CSV", type="csv", accept_multiple_files=True)
for arq in arquivos or []:
    if arq.name in st.session_state.arquivos_importados:
        continue
    try:
        bruto = pd.read_csv(arq, sep=None, engine="python")
    except Exception as erro:
        st.error(f"{arq.name}: não foi possível ler o arquivo ({erro}).")
        continue
    limpo, avisos = dl.validar_upload(bruto)
    for aviso in avisos:
        st.warning(f"{arq.name}: {aviso}")
    if limpo is not None:
        st.session_state.dados_usuario = (
            pd.concat([st.session_state.dados_usuario, limpo], ignore_index=True)
            .drop_duplicates()
        )
        st.session_state.arquivos_importados.append(arq.name)
        st.success(f"{arq.name}: {len(limpo)} registro(s) adicionados.")

dados = st.session_state.dados_usuario
if dados.empty:
    st.info("Nenhum dado enviado ainda. Baixe o modelo acima, preencha e envie.")
else:
    dados = dados.copy()
    dados["data"] = pd.to_datetime(dados["data"])
    dados["valor"] = dados["valor"].astype(float)
    dados["iqar"] = [calcular_indice(p, v) for p, v in zip(dados["poluente"], dados["valor"])]
    dados["classificacao"] = dados["iqar"].map(categoria)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", len(dados))
    m2.metric("Cidades", dados["cidade"].nunique())
    m3.metric("IQAr médio", round(dados["iqar"].mean()))
    m4.metric("Dias acima de 'Boa'", int((dados["iqar"] > 40).groupby(dados["data"].dt.date).any().sum()))

    c1, c2 = st.columns(2)
    pol_sel = c1.selectbox("Poluente", sorted(dados["poluente"].unique()))
    cid_sel = c2.multiselect("Cidades", sorted(dados["cidade"].unique()),
                             default=sorted(dados["cidade"].unique()))
    filtro = dados[(dados["poluente"] == pol_sel) & dados["cidade"].isin(cid_sel)]
    if not filtro.empty:
        serie = filtro.pivot_table(index="data", columns="cidade", values="iqar", aggfunc="mean")
        st.line_chart(serie)
    st.markdown("##### Estatísticas por cidade e poluente")
    resumo = (dados.groupby(["cidade", "poluente"])["valor"]
              .agg(["count", "mean", "min", "max"]).round(2).reset_index()
              .rename(columns={"count": "n", "mean": "média", "min": "mínimo", "max": "máximo"}))
    st.dataframe(resumo, hide_index=True, width="stretch")

    with st.expander("Ver todos os registros"):
        st.dataframe(dados, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button("Baixar dados com IQAr calculado (CSV)",
                       dados.to_csv(index=False).encode("utf-8-sig"),
                       "eter_dados_classificados.csv", "text/csv", type="primary")
    if c2.button("Apagar dados enviados"):
        st.session_state.dados_usuario = pd.DataFrame(columns=dl.COLUNAS_UPLOAD)
        st.session_state.arquivos_importados = []
        st.rerun()
