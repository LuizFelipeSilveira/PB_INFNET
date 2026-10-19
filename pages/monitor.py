"""Página: consulta de qualidade do ar por cidade, comparação e histórico."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src import state
from src.api import CIDADES
from src.iqar import CORES, avaliar
from src.ui import cartao_indice, recomendacao_por_perfil

state.iniciar_estado()

st.title("📍 Monitor")
st.caption("Índices de poluição traduzidos em recomendações simples de saúde · ODS 3 e 11")

modo = st.session_state.modo
api_key = st.session_state.api_key
cenario = st.session_state.cenario
perfil = st.session_state.perfil

col_a, col_b = st.columns([3, 1], vertical_alignment="bottom")
cidade = col_a.selectbox("Cidade", list(CIDADES),
                         index=list(CIDADES).index(st.session_state.cidade), key="sel_cidade")
consultar = col_b.button("Consultar qualidade do ar", type="primary", width="stretch")

if consultar:
    st.session_state.cidade = cidade
    leitura = state.obter_leitura(cidade, modo, api_key, cenario)
    av = avaliar(leitura["concentracoes"])
    st.session_state.ultima_avaliacao = {**leitura, "avaliacao": av, "cidade": cidade}
    st.session_state.historico.append({
        "consultado_em": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "cidade": cidade, "fonte": leitura["fonte"],
        "iqar": av["indice"], "classificacao": av["categoria"],
        "poluente_critico": av["critico"],
    })

ultima = st.session_state.ultima_avaliacao
if ultima is None:
    st.info("Escolha uma cidade e clique em **Consultar qualidade do ar** para ver o índice e as recomendações.")
else:
    av = ultima["avaliacao"]
    cartao_indice(av, ultima["cidade"], ultima["fonte"], ultima["horario"])
    st.markdown(f"#### O que fazer hoje\n{recomendacao_por_perfil(av['categoria'], perfil)}")

    detalhes = pd.DataFrame(av["detalhes"])
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("##### Leitura por poluente")
        st.dataframe(detalhes.drop(columns="Sigla"), hide_index=True, width="stretch")
    with c2:
        st.markdown("##### Índice por poluente")
        st.bar_chart(detalhes.set_index("Sigla")["IQAr"], color=CORES[av["categoria"]])

st.divider()
st.markdown("##### Comparar cidades")
comparar = st.multiselect("Selecione até 6 cidades", list(CIDADES), max_selections=6,
                          default=["Rio de Janeiro - RJ", "São Paulo - SP", "Porto Velho - RO"])
if comparar:
    linhas = []
    for c in comparar:
        a = avaliar(state.obter_leitura(c, modo, api_key, cenario)["concentracoes"])
        linhas.append({"Cidade": c, "IQAr": a["indice"], "Classificação": a["categoria"],
                       "Poluente crítico": a["critico"]})
    comp = pd.DataFrame(linhas).sort_values("IQAr", ascending=False)
    st.bar_chart(comp.set_index("Cidade")["IQAr"], horizontal=True)
    st.dataframe(comp, hide_index=True, width="stretch")

if st.session_state.historico:
    with st.expander(f"Histórico desta sessão ({len(st.session_state.historico)} consultas)"):
        hist = pd.DataFrame(st.session_state.historico)
        st.dataframe(hist, hide_index=True, width="stretch")
        st.download_button("Baixar histórico (CSV)", hist.to_csv(index=False).encode("utf-8-sig"),
                           "historico_consultas.csv", "text/csv")
