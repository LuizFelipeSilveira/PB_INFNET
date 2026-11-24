"""Dashboard final (TP5, Tarefa 3): do dado coletado ao insight gerado pela IA.

Mostra, numa página só, o ciclo de ciência de dados do Éter:
  coleta (notícias, tabela oficial, OpenWeather) -> limpeza -> IA local (sentimento e
  sumarização) -> insight (boletim com orientação oficial e fontes).

Os gráficos e boletins vêm de data/processed/insights/ (gerados por
scripts/llm/sumarizar_noticias.py), então funcionam mesmo onde o LLM não está
instalado. A aba "Gerar agora" chama a API (POST /sumarizar_noticias) para criar um
boletim novo em tempo real.
"""
from __future__ import annotations

import json

import altair as alt
import pandas as pd
import requests
import streamlit as st

from src import sumarizacao as sz
from src import state
from src.api import CIDADES
from src.iqar import CATEGORIAS, CORES, FONTE_LIMITES, avaliar
from src.ui import TIMEOUT_APP_S, cartao_indice, mostrar_erro

CORES_SENTIMENTO = {"negativo": "#C0392B", "neutro": "#95A5A6", "positivo": "#2E9E5B"}


# ---------------------------------------------------------------- dados
@st.cache_data(show_spinner=False)
def carregar_insights(_mtime_resumos: float, _mtime_noticias: float):
    resumos = (json.loads(sz.ARQ_RESUMOS.read_text(encoding="utf-8"))
               if sz.ARQ_RESUMOS.exists() else None)
    noticias = sz.carregar_noticias()
    noticias["categoria_citada"] = noticias["titulo"].map(sz.categoria_da_manchete)
    if "sentimento" not in noticias.columns:
        noticias["sentimento"] = None
    noticias["sentimento"] = noticias["sentimento"].fillna("não classificado")
    return resumos, noticias


resumos, df = carregar_insights(state.mtime(sz.ARQ_RESUMOS), state.mtime(sz.ARQ_ENRIQUECIDAS))
brutas = len(pd.read_csv(sz.ARQ_NOTICIAS))
classificadas = int((df["sentimento"] != "não classificado").sum())


def pct(parte: int, todo: int) -> str:
    return f"{parte / todo:.0%}" if todo else "—"


# ---------------------------------------------------------------- cabeçalho
st.title("🧭 Do dado ao insight")
st.caption("Notícias reais sobre qualidade do ar, processadas por um modelo de linguagem que roda "
           "localmente e transformadas em boletins com a orientação oficial de saúde.")

c1, c2, c3, c4, c5 = st.columns(5)
# Valores de apoio vão em legendas, não em "delta": não são variações, e a seta confundiria.
c1.metric("Notícias coletadas", brutas, help="Web scraping do Google Notícias (RSS).")
c1.caption("scraping do Google Notícias")
c2.metric("Notícias válidas", len(df), help="Sem páginas fixas do IQAir e sem duplicatas.")
c2.caption(f"{brutas - len(df)} páginas de índice removidas")
c3.metric("Classificadas pela IA", classificadas,
          help="Sentimento de cada manchete pelo LLM local (92% de acerto na avaliação).")
c3.caption(f"{pct(int((df['sentimento'] == 'negativo').sum()), classificadas)} negativas")
c4.metric("Boletins gerados", len(resumos["recortes"]) if resumos else 0,
          help="Um por recorte: geral, por local, por mês e por tema.")
c4.caption("resumos do LLM local")
c5.metric("Norma do IQAr", "506/2024", help=FONTE_LIMITES)
c5.caption("Resolução CONAMA (PDF do MMA)")

st.markdown(
    f"**Coleta** ({brutas} notícias, tabela oficial do PDF do MMA, OpenWeather) → "
    f"**Limpeza** ({brutas - len(df)} páginas de índice removidas, local e mês detectados) → "
    f"**IA local** (sentimento de {classificadas} manchetes, resumos em cadeia de prompts) → "
    "**Insight** (boletim com fatos, orientação oficial e fontes conferíveis)"
)

aba_panorama, aba_boletins, aba_gerar, aba_dados = st.tabs(
    ["📊 Panorama", "📰 Boletins da IA", "⚡ Gerar agora", "🔎 Dados"])


# ---------------------------------------------------------------- panorama
with aba_panorama:
    neg = int((df["sentimento"] == "negativo").sum())
    top_local = df.loc[df["local"] != sz.OUTROS_LOCAIS, "local"].value_counts()
    por_dia = df.groupby(df["data"].dt.date).size()
    insights = [f"**{pct(neg, classificadas)}** das notícias classificadas pela IA relatam piora "
                f"do ar ou danos à saúde."]
    if not top_local.empty:
        insights.append(f"**{top_local.index[0]}** aparece em **{pct(top_local.iloc[0], len(df))}** "
                        "das notícias válidas: é o centro do noticiário no período.")
    if not por_dia.empty:
        insights.append(f"O pico foi em **{por_dia.idxmax():%d/%m/%Y}**, com **{por_dia.max()}** "
                        "notícias num só dia.")
    citadas = df["categoria_citada"].value_counts()
    if not citadas.empty:
        insights.append(f"Quando as manchetes citam uma categoria do IQAr, a mais citada é "
                        f"**{citadas.index[0]}** ({citadas.iloc[0]} manchetes).")
    st.markdown("\n".join(f"- {i}" for i in insights))
    st.caption("Esses números são calculados por código a partir das notícias e do sentimento "
               "gerado pela IA.")

    meses = sorted(df["mes"].unique())
    # começa nos dois últimos meses, onde está quase todo o noticiário
    periodo = st.select_slider("Período", options=meses, value=(meses[-2], meses[-1]),
                               format_func=lambda m: f"{m[5:]}/{m[:4]}") if len(meses) > 1 \
        else (meses[0], meses[0])
    vis = df[(df["mes"] >= periodo[0]) & (df["mes"] <= periodo[1])]
    escala_sent = alt.Scale(domain=list(CORES_SENTIMENTO) + ["não classificado"],
                            range=list(CORES_SENTIMENTO.values()) + ["#D5D8DC"])

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.subheader("Notícias por dia e sentimento")
        st.altair_chart(
            alt.Chart(vis.assign(dia=vis["data"].dt.date.astype(str))).mark_bar().encode(
                x=alt.X("dia:T", title=None, axis=alt.Axis(format="%d/%m")),
                y=alt.Y("count():Q", title="notícias"),
                color=alt.Color("sentimento:N", scale=escala_sent, title="Sentimento"),
                tooltip=["dia:T", "sentimento:N", "count():Q"],
            ).properties(height=280), use_container_width=True)
    with col_b:
        st.subheader("Onde se fala do problema")
        st.altair_chart(
            alt.Chart(vis).mark_bar().encode(
                y=alt.Y("local:N", sort="-x", title=None),
                x=alt.X("count():Q", title="notícias"),
                color=alt.Color("sentimento:N", scale=escala_sent, legend=None),
                tooltip=["local:N", "sentimento:N", "count():Q"],
            ).properties(height=280), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Gravidade relatada nas manchetes")
        sev = vis.dropna(subset=["categoria_citada"])
        if sev.empty:
            st.info("Nenhuma manchete do período cita uma categoria do IQAr.")
        else:
            st.altair_chart(
                alt.Chart(sev).mark_bar().encode(
                    x=alt.X("categoria_citada:N", sort=CATEGORIAS, title="categoria do IQAr citada",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("count():Q", title="manchetes"),
                    color=alt.Color("categoria_citada:N", legend=None, scale=alt.Scale(
                        domain=CATEGORIAS, range=[CORES[c] for c in CATEGORIAS])),
                    tooltip=["categoria_citada:N", "count():Q"],
                ).properties(height=250), use_container_width=True)
            st.caption("Manchetes que usam termos das categorias oficiais (ex.: \"péssima\", "
                       "\"perigosa\", \"muito insalubre\"). É daqui que vem a orientação dos boletins.")
    with col_d:
        st.subheader("Quem mais noticia")
        st.altair_chart(
            alt.Chart(vis[vis["fonte"].isin(vis["fonte"].value_counts().head(8).index)])
            .mark_bar(color="#1F6F8B").encode(
                y=alt.Y("fonte:N", sort="-x", title=None),
                x=alt.X("count():Q", title="notícias"),
                tooltip=["fonte:N", "count():Q"],
            ).properties(height=250), use_container_width=True)


# ---------------------------------------------------------------- boletim (componente)
def cidade_monitorada(local: str | None) -> str | None:
    if not local:
        return None
    return next((c for c in CIDADES if c.split(" - ")[0] == local), None)


def mostrar_boletim(item: dict) -> None:
    """Exibe um boletim (do arquivo de insights ou da API): texto da IA, orientação e fontes."""
    b, est, fid, proc = item["boletim"], item["estatisticas"], item["fidelidade"], item["processamento"]
    col_txt, col_num = st.columns([2, 1])
    with col_txt:
        st.markdown(f"#### {b['resumo']}")
        st.markdown("**Pontos principais**\n" + "\n".join(f"- {p}" for p in b["pontos"]))
        if b.get("categoria_recomendacao"):
            cat = b["categoria_recomendacao"]
            st.markdown(
                f"""<div style="border-left:8px solid {CORES[cat]};padding:.7rem 1rem;
                border-radius:6px;background:var(--secondary-background-color);margin-top:.6rem">
                <b>Orientação oficial: qualidade {cat.lower()}</b><br>{b['recomendacao']}<br>
                <span style="opacity:.7;font-size:.85rem">Categoria citada em
                {b.get('manchetes_que_citam_categoria', 0)} manchete(s) deste recorte.</span></div>""",
                unsafe_allow_html=True)
        else:
            st.info("As manchetes deste recorte não citam uma categoria do IQAr; sem orientação.")
        if not b.get("formato_ok", True):
            st.warning("O modelo não seguiu totalmente o formato pedido neste boletim.")
    with col_num:
        st.metric("Notícias resumidas", est["n_noticias"])
        st.caption(f"de {est['periodo_inicio'][8:10]}/{est['periodo_inicio'][5:7]}/"
                   f"{est['periodo_inicio'][:4]} a {est['periodo_fim'][8:10]}/"
                   f"{est['periodo_fim'][5:7]}/{est['periodo_fim'][:4]}")
        st.metric("Leitura reduzida a", f"{proc['palavras_saida'] / proc['palavras_entrada']:.0%}")
        st.caption(f"{proc['palavras_entrada']} palavras de manchetes → "
                   f"{proc['palavras_saida']} no boletim")
        st.metric("Fidelidade às fontes", f"{fid['taxa']:.0%}")
        st.caption(("Conferir: " + ", ".join(fid["nao_encontrados"])) if fid["nao_encontrados"]
                   else f"{fid['termos_verificados']} números e nomes conferidos nas manchetes")
        if est.get("sentimento"):
            total = sum(est["sentimento"].values())
            partes = [f"<span style='color:{CORES_SENTIMENTO.get(k, '#95A5A6')}'>■</span> "
                      f"{k} {v / total:.0%}" for k, v in
                      sorted(est["sentimento"].items(), key=lambda kv: -kv[1])]
            st.markdown("**Sentimento (IA):** " + " · ".join(partes), unsafe_allow_html=True)

    cidade = cidade_monitorada((item.get("filtros") or {}).get("local"))
    if cidade:
        with st.expander(f"Agora em {cidade}: medição atual de poluentes", expanded=True):
            leitura = state.obter_leitura(cidade, st.session_state.modo,
                                          st.session_state.api_key, st.session_state.cenario)
            fonte = "OpenWeather" if st.session_state.modo == "API OpenWeather" else "simulação"
            cartao_indice(avaliar(leitura["concentracoes"]), cidade, fonte, leitura["horario"])
            st.caption("O boletim resume o que foi noticiado; o cartão mostra o ar medido agora.")

    with st.expander(f"Manchetes de origem ({len(item['fontes'])}) para conferir o boletim"):
        st.dataframe(pd.DataFrame(item["fontes"]), hide_index=True, width="stretch",
                     column_config={"link": st.column_config.LinkColumn("link", display_text="abrir")})
    with st.expander("Como este boletim foi gerado"):
        st.markdown(
            f"- **{est['n_noticias']} manchetes** divididas em **{proc['n_lotes']} lote(s)** de até "
            f"{sz.TAMANHO_LOTE}; cada lote virou 3 fatos.\n"
            f"- Fatos consolidados até os 3 pontos principais; um último prompt escreveu o resumo "
            f"({proc['n_chamadas']} chamadas ao modelo, {proc['tokens_gerados']} tokens, "
            f"{proc['tempo_s']} s).\n"
            "- A orientação é o texto oficial da categoria mais grave citada nas manchetes "
            "(escolhida por código, não pelo modelo).\n"
            "- Fidelidade: números, nomes e locais do texto gerado conferidos contra as manchetes.\n"
            "- Detalhes e avaliação: `docs/model/Sumarizacao.md`.")


# ---------------------------------------------------------------- boletins prontos
with aba_boletins:
    if not resumos:
        st.info("Os boletins ainda não foram gerados. Rode `python scripts/llm/sumarizar_noticias.py`.")
    else:
        opcoes = {r["titulo"]: r for r in resumos["recortes"]}
        escolha = st.selectbox("Recorte", list(opcoes))
        mostrar_boletim(opcoes[escolha])
        st.caption(f"Gerado em {resumos['gerado_em'].replace('T', ' ')} com {resumos['modelo']} "
                   f"({resumos['formato']}), rodando localmente.")


# ---------------------------------------------------------------- gerar agora
with aba_gerar:
    st.markdown("Escolha um recorte e peça um boletim novo ao modelo de linguagem local, pela "
                "API (`POST /sumarizar_noticias`). A API precisa estar rodando "
                "(`uvicorn api.main:app`) numa máquina com o modelo instalado.")
    todos = "Todos"
    with st.form("form_sumarizar"):
        f1, f2, f3 = st.columns(3)
        local = f1.selectbox("Local", [todos] + df["local"].value_counts().index.tolist())
        mes = f2.selectbox("Mês", [todos] + sorted(df["mes"].unique(), reverse=True),
                           format_func=lambda m: m if m == todos else f"{m[5:]}/{m[:4]}")
        tema = f3.selectbox("Tema (termo de busca)", [todos] + df["termo_busca"].value_counts().index.tolist())
        n = st.slider("Notícias mais recentes a resumir", 5, 60, 20, step=5,
                      help="Mais notícias = mais contexto e mais tempo (cerca de 10 s a cada 20).")
        enviar = st.form_submit_button("Gerar boletim agora", type="primary")
    if enviar:
        corpo = {"max_noticias": n, **{k: v for k, v in
                                        {"local": local, "mes": mes, "tema": tema}.items()
                                        if v != todos}}
        url = st.session_state.get("url_base", "http://127.0.0.1:8000")
        resp = None
        with st.spinner("O modelo está lendo as manchetes e escrevendo o boletim..."):
            try:
                resp = requests.post(f"{url}/sumarizar_noticias", json=corpo, timeout=TIMEOUT_APP_S)
            except requests.ConnectionError:
                st.error(f"A API não está respondendo em {url}.")
                st.info("Inicie a API em outro terminal com `uvicorn api.main:app`. Enquanto isso, "
                        "a aba **Boletins da IA** mostra os boletins já gerados.")
            except requests.Timeout:
                st.error(f"A API não respondeu em {TIMEOUT_APP_S // 60} minutos.")
            except requests.RequestException as erro:
                st.error(f"Falha de comunicação com a API ({type(erro).__name__}).")
        if resp is not None:
            dados = resp.json() if resp.headers.get("content-type", "").startswith(
                "application/json") else {"mensagem": resp.text[:300]}
            if resp.ok:
                st.success(f"Boletim gerado: {dados['descricao']}.")
                mostrar_boletim(dados)
            else:
                mostrar_erro(resp, dados)


# ---------------------------------------------------------------- dados
with aba_dados:
    st.markdown("Notícias **depois** da limpeza e do processamento pela IA: local e mês "
                "detectados por código, sentimento classificado pelo LLM e categoria do IQAr "
                "citada na manchete.")
    colunas = ["data", "titulo", "fonte", "local", "mes", "termo_busca", "sentimento",
               "categoria_citada", "link"]
    tabela = df[[c for c in colunas if c in df.columns]]
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={"link": st.column_config.LinkColumn("link", display_text="abrir"),
                                "data": st.column_config.DatetimeColumn("data", format="DD/MM/YYYY")})
    st.download_button("Baixar CSV", tabela.to_csv(index=False).encode("utf-8"),
                       "noticias_processadas.csv", "text/csv")
    st.caption("Outras fontes do projeto: a tabela oficial do IQAr está na página **Tabela IQAr**, "
               "as medições em tempo real no **Monitor**, e a avaliação dos modelos em "
               "`docs/model/`.")
