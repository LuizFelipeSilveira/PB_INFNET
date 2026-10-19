"""Página: notícias coletadas, nuvem de palavras e estatísticas de texto."""
from __future__ import annotations

import streamlit as st

from src import data_loader as dl
from src import state

state.iniciar_estado()

st.title("📰 Notícias e palavras")

noticias = state.carregar_csv(str(dl.ARQ_NOTICIAS), state.mtime(dl.ARQ_NOTICIAS))
texto_noticias = state.carregar_texto(str(dl.ARQ_TEXTO), state.mtime(dl.ARQ_TEXTO))
arq_ref = dl.DATA_RAW / "referencia_iqar_texto.txt"
texto_ref = state.carregar_texto(str(arq_ref), state.mtime(arq_ref))

if noticias is None and not texto_ref:
    st.info("Ainda não há dados coletados. Rode os scripts de extração na raiz do projeto:\n\n"
            "```bash\npython scripts/scraping/scrape_noticias.py\npython scripts/scraping/scrape_iqar.py\n```")
else:
    fontes_texto = {"Manchetes de notícias": texto_noticias, "Páginas de referência (IQAr)": texto_ref}
    fontes_texto = {k: v for k, v in fontes_texto.items() if v.strip()}

    st.markdown("##### Nuvem de palavras")
    c1, c2, c3 = st.columns(3)
    origem = c1.selectbox("Texto analisado", list(fontes_texto))
    max_pal = c2.slider("Número máximo de palavras", 20, 200, 80, step=10)
    paleta = c3.selectbox("Paleta", ["viridis", "ocean", "plasma", "cividis"])
    texto = fontes_texto[origem]
    st.image(state.gerar_nuvem(texto, max_pal, paleta), width="stretch")

    freq = dl.frequencia_palavras(texto, top=15)
    c1, c2 = st.columns(2)
    c1.markdown("##### Palavras mais frequentes")
    c1.bar_chart(freq.set_index("Palavra"), horizontal=True)
    c2.markdown("##### Estatísticas do texto")
    palavras = texto.split()
    c2.metric("Total de palavras", f"{len(palavras):,}".replace(",", "."))
    c2.metric("Palavras distintas (sem stopwords)", len(dl.frequencia_palavras(texto, top=100000)))

    if noticias is not None:
        st.divider()
        st.markdown("##### Notícias coletadas")
        m1, m2, m3 = st.columns(3)
        m1.metric("Notícias", len(noticias))
        m2.metric("Veículos", noticias["fonte"].nunique())
        m3.metric("Coletadas em", str(noticias["coletado_em"].iloc[0])[:10])

        f1, f2 = st.columns([2, 1])
        busca = f1.text_input("Filtrar por palavra no título", placeholder="ex.: queimadas")
        termos = f2.multiselect("Termo de busca", sorted(noticias["termo_busca"].unique()))
        filtradas = noticias
        if busca:
            filtradas = filtradas[filtradas["titulo"].str.contains(busca, case=False, na=False)]
        if termos:
            filtradas = filtradas[filtradas["termo_busca"].isin(termos)]

        top_fontes = filtradas["fonte"].value_counts().head(10)
        if not top_fontes.empty:
            st.bar_chart(top_fontes, horizontal=True)
        st.dataframe(
            filtradas[["data_publicacao", "titulo", "fonte", "link"]],
            hide_index=True, width="stretch",
            column_config={"link": st.column_config.LinkColumn("Link", display_text="abrir")},
        )
