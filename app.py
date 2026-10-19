"""Éter - Monitor Inteligente de Qualidade do Ar (Streamlit, multipage).

Este arquivo é só o ponto de entrada: monta o menu de navegação e a barra
lateral de configurações (compartilhada por todas as páginas via
st.session_state). O conteúdo de cada aba antiga agora vive em pages/.
"""
from __future__ import annotations

import streamlit as st

from src import state

st.set_page_config(page_title="Éter - Monitor", page_icon="🌬️", layout="wide")
state.iniciar_estado()

paginas = {
    "Qualidade do ar": [
        st.Page("pages/monitor.py", title="Monitor", icon="📍", default=True),
        st.Page("pages/noticias.py", title="Notícias e palavras", icon="📰"),
        st.Page("pages/referencia.py", title="Tabela IQAr", icon="📋"),
    ],
    "Seus dados": [
        st.Page("pages/dados.py", title="Meus dados", icon="📂"),
    ],
    "Projeto": [
        st.Page("pages/api_llm.py", title="API & IA", icon="🤖"),
        st.Page("pages/sobre.py", title="Sobre", icon="ℹ️"),
    ],
}
pagina_atual = st.navigation(paginas)

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configurações")
    chave_secrets = state.obter_api_key()
    st.radio("Origem dos dados", ["Simulação", "API OpenWeather"],
             index=1 if chave_secrets else 0,
             help="Sem chave de API, o app usa leituras simuladas.", key="modo")
    if st.session_state.modo == "API OpenWeather" and not chave_secrets:
        st.text_input("Chave da OpenWeather", type="password", key="api_key_input",
                      help="Crie uma chave gratuita em openweathermap.org/api")
    if st.session_state.modo == "Simulação":
        st.selectbox("Cenário simulado", ["Aleatório", "Ar limpo", "Poluído"], key="cenario")
    st.session_state.api_key = chave_secrets or st.session_state.get("api_key_input", "")

    st.radio("Seu perfil", ["População em geral", "Grupo sensível",
                            "Pratico esportes ao ar livre"], key="perfil")
    st.divider()
    st.caption(f"Consultas nesta sessão: **{len(st.session_state.historico)}**")
    st.caption(f"Registros enviados: **{len(st.session_state.dados_usuario)}**")
    if st.button("Limpar cache de dados"):
        st.cache_data.clear()
        st.toast("Cache limpo. Os dados serão recarregados.")

pagina_atual.run()
