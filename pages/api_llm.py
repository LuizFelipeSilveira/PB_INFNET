"""Página: API (FastAPI), LLM local (TP4) e preparação dos dados para LLMs (TP3)."""
from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from src import data_loader as dl
from src import state
from src.llm_prep import montar_corpus, resumo_corpus, salvar_corpus

st.title("🤖 API & preparação para IA")

aba_api, aba_local, aba_llm = st.tabs(
    ["🔌 API (FastAPI)", "💬 LLM local", "🧠 Preparação para LLMs"])

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
        {"Método": "POST", "Rota": "/processar_texto", "Função": "Resumo, sentimento ou pergunta com o LLM local (TP4)."},
        {"Método": "GET", "Rota": "/llm/status", "Função": "Modelo configurado e se já está carregado."},
    ]), hide_index=True, width="stretch")

    st.markdown("##### Testar a API a partir desta página")
    url_base = st.text_input("Endereço da API", "http://127.0.0.1:8000", key="url_base")

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

# ---------------------------------------------------------------- LLM local (TP4)
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


with aba_local:
    st.markdown(
        "O texto é enviado para a rota `POST /processar_texto` da API, que o processa com um "
        "modelo de linguagem do Hugging Face rodando **nesta máquina** (biblioteca Transformers, "
        "sem nenhum serviço na nuvem). Veja a escolha do modelo em `docs/model/ModelSelection.md`."
    )
    url_llm = st.session_state.get("url_base", "http://127.0.0.1:8000")
    tarefas = {"Resumo": "resumo", "Sentimento": "sentimento",
               "Pergunta sobre a tabela IQAr": "pergunta"}
    rotulo = st.radio("Tarefa", list(tarefas), horizontal=True)
    tarefa = tarefas[rotulo]

    sugestao = ""
    noticias = state.carregar_csv(str(dl.ARQ_NOTICIAS), state.mtime(dl.ARQ_NOTICIAS))
    if tarefa == "pergunta":
        sugestao = "Qual é a faixa de concentração de MP2.5 para a qualidade do ar ruim?"
    elif noticias is not None and not noticias.empty:
        escolha = st.selectbox("Usar uma notícia coletada (opcional)",
                               ["(digitar meu próprio texto)"] + noticias["titulo"].head(30).tolist())
        if escolha != "(digitar meu próprio texto)":
            linha = noticias[noticias["titulo"] == escolha].iloc[0]
            sugestao = escolha if tarefa == "sentimento" else str(linha["resumo"])

    with st.form("form_llm"):
        texto = st.text_area("Texto", sugestao, height=150, max_chars=4000)
        enviar = st.form_submit_button("Processar com o LLM local", type="primary")
    if enviar:
        with st.spinner("Processando... o primeiro pedido também carrega o modelo e pode "
                        "levar alguns minutos."):
            resp = None
            try:
                resp = requests.post(f"{url_llm}/processar_texto",
                                     json={"texto": texto, "tarefa": tarefa}, timeout=TIMEOUT_APP_S)
            except requests.ConnectionError:
                st.error(f"A API não está respondendo em {url_llm}.")
                st.info("Inicie a API em outro terminal com `uvicorn api.main:app` e tente de novo.")
            except requests.Timeout:
                st.error(f"A API não respondeu em {TIMEOUT_APP_S // 60} minutos.")
                st.info("Na primeira chamada o modelo ainda está sendo carregado. Confira o "
                        "terminal da API e tente de novo em instantes.")
            except requests.RequestException as erro:
                st.error(f"Falha de comunicação com a API ({type(erro).__name__}).")
            if resp is not None:
                try:
                    dados = resp.json()
                except ValueError:  # resposta que não é JSON (ex.: proxy no meio do caminho)
                    dados = {"mensagem": resp.text[:300]}
                if resp.ok:
                    analise, meta = dados["analise"], dados["metadados"]
                    if analise["tipo"] == "sentimento":
                        icones = {"positivo": "🟢", "negativo": "🔴", "neutro": "⚪"}
                        if analise["rotulo_valido"]:
                            st.metric("Sentimento", f"{icones[analise['rotulo']]} {analise['rotulo']}")
                        else:
                            st.warning("O modelo não devolveu um rótulo válido: "
                                       f"“{analise['resposta_modelo']}”")
                    elif analise["tipo"] == "resumo":
                        st.success(analise["resumo"])
                        st.caption(f"{analise['palavras_entrada']} → {analise['palavras_resumo']} "
                                   f"palavras (taxa de compressão {analise['taxa_compressao']:.0%})")
                    else:
                        st.success(analise["resposta"])
                        st.caption("Contexto consultado: tabela IQAr de "
                                   + ", ".join(analise["poluentes_consultados"]))
                    st.caption(f"{meta['modelo']} · {meta['formato']} · "
                               f"{meta['tokens_gerados']} tokens em {meta['tempo_s']} s")
                    with st.expander("Ver o JSON completo devolvido pela API"):
                        st.json(dados)
                else:
                    mostrar_erro(resp, dados)

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
