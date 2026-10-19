"""Página: tabela oficial de referência do IQAr e calculadora rápida."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src import state
from src.iqar import RECOMENDACOES, calcular_indice, categoria

state.iniciar_estado()

st.title("📋 Tabela IQAr")

tabela = state.carregar_csv(str(dl.ARQ_IQAR), state.mtime(dl.ARQ_IQAR))
if tabela is None:
    st.info("Rode `python scripts/scraping/scrape_iqar.py` para gerar a tabela de referência.")
else:
    pol = st.segmented_control("Poluente", tabela["poluente"].unique(), default="MP2.5")
    vis = tabela[tabela["poluente"] == pol] if pol else tabela
    st.dataframe(vis.drop(columns=["nome"]), hide_index=True, width="stretch")

    st.markdown("##### Calculadora rápida")
    c1, c2 = st.columns(2)
    pol_calc = c1.selectbox("Poluente", tabela["poluente"].unique(), key="pol_calc")
    unidade = tabela.loc[tabela["poluente"] == pol_calc, "unidade"].iloc[0]
    valor = c2.number_input(f"Concentração ({unidade})", min_value=0.0, value=20.0, step=1.0)
    idx = calcular_indice(pol_calc, valor)
    st.markdown(f"IQAr **{idx}** → qualidade **{categoria(idx).lower()}**. {RECOMENDACOES[categoria(idx)]}")

    pasta = dl.DATA_RAW / "tabelas_web"
    brutas = sorted(pasta.glob("*.csv")) if pasta.exists() else []
    if brutas:
        with st.expander(f"Tabelas brutas extraídas da web ({len(brutas)})"):
            escolha = st.selectbox("Arquivo", [b.name for b in brutas])
            st.dataframe(pd.read_csv(pasta / escolha), width="stretch")
