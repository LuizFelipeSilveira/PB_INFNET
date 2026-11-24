"""Componentes de interface reaproveitados entre páginas."""
from __future__ import annotations

import streamlit as st

from src.iqar import CATEGORIAS, CORES, RECOMENDACOES


def cartao_indice(av: dict, cidade: str, fonte: str, horario: str) -> None:
    cor = CORES[av["categoria"]]
    st.markdown(
        f"""
        <div style="border-left:10px solid {cor};padding:1rem 1.4rem;border-radius:6px;
                    background:var(--secondary-background-color);">
          <div style="font-size:0.95rem;opacity:.75">{cidade} · {horario} · {fonte}</div>
          <div style="display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap">
            <span style="font-size:3.6rem;font-weight:700;color:{cor};line-height:1">{av['indice']}</span>
            <span style="font-size:1.6rem;font-weight:600">Qualidade {av['categoria'].lower()}</span>
          </div>
          <div style="margin-top:.3rem">Poluente crítico: <b>{av['critico']}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recomendacao_por_perfil(cat: str, perfil: str) -> str:
    base = RECOMENDACOES[cat]
    nivel = CATEGORIAS.index(cat)
    if perfil == "Grupo sensível" and nivel >= 1:
        return base + " Como você faz parte de um grupo sensível, prefira atividades leves em ambientes fechados."
    if perfil == "Pratico esportes ao ar livre":
        if nivel == 0:
            return base + " Ótimo dia para treinar."
        if nivel == 1:
            return base + " Se for treinar, reduza a intensidade e prefira o início da manhã."
        return base + " Troque o treino externo por um treino indoor hoje."
    return base


# ---------------------------------------------------------------- erros da API
TIMEOUT_APP_S = 600  # maior que o limite da API (ETER_LLM_TIMEOUT_S) + a carga do modelo
TITULOS_ERRO = {  # títulos amigáveis para os códigos de erro da API (api/erros.py)
    "entrada_invalida": "O texto enviado não é válido",
    "llm_dependencias_ausentes": "O modelo de linguagem não está instalado",
    "llm_modelo_nao_encontrado": "O modelo de linguagem não foi baixado",
    "llm_memoria_insuficiente": "Faltou memória na placa de vídeo",
    "llm_falha_carga": "O modelo de linguagem não pôde ser carregado",
    "llm_ocupado": "O modelo está ocupado com outro pedido",
    "llm_tempo_esgotado": "O processamento demorou demais",
    "resposta_invalida_do_modelo": "O modelo respondeu fora do formato esperado",
    "sem_noticias": "Nenhuma notícia neste recorte",
    "dados_indisponiveis": "Os dados ainda não foram coletados",
}


def mostrar_erro(resp, dados: dict) -> None:
    """Mostra um erro da API (formato ErroAPI) de forma clara: o quê, por quê e o que fazer."""
    codigo = dados.get("erro", f"http_{resp.status_code}")
    titulo = TITULOS_ERRO.get(codigo, f"A API devolveu o erro {resp.status_code}")
    st.error(f"**{titulo}.** {dados.get('mensagem', '')}")
    for campo in dados.get("campos", []):
        st.caption(f"• `{campo['campo']}`: {campo['mensagem']}")
    if dados.get("sugestao"):
        st.info(f"💡 {dados['sugestao']}")
    if resp.headers.get("Retry-After"):
        st.caption(f"Tente de novo em cerca de {resp.headers['Retry-After']} s.")
    st.caption(f"Código `{codigo}` · status {resp.status_code} · "
               f"id da requisição `{dados.get('id_requisicao', '-')}` (aparece no log da API)")
