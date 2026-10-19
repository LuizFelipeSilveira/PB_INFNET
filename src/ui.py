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
