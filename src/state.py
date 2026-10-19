"""Estado de sessão, cache e acesso a dados compartilhados entre as páginas do app.

Centralizado aqui porque, no modelo multipage do Streamlit, cada página é um
script independente: funções `@st.cache_data` e inicialização de
`st.session_state` precisam vir de um único lugar para serem reaproveitadas
(e cacheadas de verdade) em todas as páginas.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import data_loader as dl
from src.api import buscar_qualidade_ar, simular_qualidade_ar


def iniciar_estado() -> None:
    padroes = {
        "historico": [],            # consultas feitas pelo usuário
        "dados_usuario": pd.DataFrame(columns=dl.COLUNAS_UPLOAD),
        "arquivos_importados": [],  # nomes dos CSVs já incorporados
        "ultima_avaliacao": None,
        "cidade": "Rio de Janeiro - RJ",
        "modo": "Simulação",
        "cenario": "Aleatório",
        "perfil": "População em geral",
        # Preenchido de verdade pela barra lateral em app.py; o padrão aqui é só
        # para o caso de uma página ser executada/testada fora do fluxo normal.
        "api_key": "",
    }
    for chave, valor in padroes.items():
        st.session_state.setdefault(chave, valor)


# ---------------------------------------------------------------- cache
@st.cache_data(ttl=600, show_spinner="Consultando a OpenWeather...")
def consultar_api(lat: float, lon: float, api_key: str) -> dict:
    return buscar_qualidade_ar(lat, lon, api_key)


@st.cache_data(show_spinner=False)
def carregar_csv(caminho: str, _mtime: float) -> pd.DataFrame | None:
    return dl.ler_csv(dl.Path(caminho))


@st.cache_data(show_spinner=False)
def carregar_texto(caminho: str, _mtime: float) -> str:
    return dl.ler_texto(dl.Path(caminho))


@st.cache_data(show_spinner="Gerando nuvem de palavras...")
def gerar_nuvem(texto: str, max_palavras: int, colormap: str):
    from wordcloud import WordCloud
    nuvem = WordCloud(width=1000, height=450, background_color="white",
                      stopwords=dl.STOPWORDS_PT, max_words=max_palavras,
                      colormap=colormap, collocations=False)
    return nuvem.generate(texto).to_array()


def mtime(caminho) -> float:
    return caminho.stat().st_mtime if caminho.exists() else 0.0


def obter_api_key() -> str:
    try:
        return st.secrets.get("OPENWEATHER_API_KEY", "")
    except Exception:  # sem secrets.toml
        return ""


def obter_leitura(cidade: str, modo: str, api_key: str, cenario: str) -> dict:
    from src.api import CIDADES
    lat, lon = CIDADES[cidade]
    if modo == "API OpenWeather" and api_key:
        try:
            return consultar_api(lat, lon, api_key)
        except Exception as erro:  # requests.RequestException
            st.warning(f"Não foi possível consultar a API ({erro}). Mostrando dados simulados.")
    return simular_qualidade_ar(cidade, cenario)
