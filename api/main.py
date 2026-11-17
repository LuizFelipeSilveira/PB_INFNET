"""API do Éter (FastAPI).

Expõe, para outros sistemas (ou para o próprio app Streamlit), os mesmos dados e
regras de negócio usados na interface: a tabela oficial do IQAr, o cálculo do
índice e um jeito de registrar novas leituras (de sensores, estações locais etc.)
sem precisar passar pela interface web.

Para rodar (a partir da raiz do projeto):
    uvicorn api.main:app --reload

Depois, a documentação interativa fica disponível em http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))

from src import data_loader as dl  # noqa: E402
from src.api import CIDADES  # noqa: E402
from src.iqar import RECOMENDACOES, calcular_indice, categoria  # noqa: E402

from src.llm import LLMErro, LLMLocal, contexto_iqar, rotulo_sentimento  # noqa: E402

from api import erros as tratamento  # noqa: E402
from api.erros import ErroHTTP, erro_do_llm  # noqa: E402
from api.schemas import (  # noqa: E402
    AnalisePergunta,
    AnaliseResumo,
    AnaliseSentimento,
    Cidade,
    ErroAPI,
    FaixaReferencia,
    LeituraRegistrada,
    MetadadosLLM,
    NovaLeitura,
    PedidoTexto,
    RespostaProcessamento,
    ResultadoIQAr,
    StatusAPI,
    StatusLLM,
)

ARQ_LEITURAS_API = dl.DATA_RAW.parent / "processed" / "leituras_api.csv"
COLUNAS_LEITURAS_API = ["cidade", "poluente", "valor", "data", "origem",
                        "recebido_em", "iqar", "categoria"]

# Um único modelo para a API inteira, carregado no primeiro pedido (ver src/llm.py).
llm = LLMLocal()

app = FastAPI(
    title="Éter API",
    description="Dados e cálculo de qualidade do ar (IQAr) do projeto Éter.",
    version="1.2.0",
)

# Em produção, restrinja allow_origins ao domínio real do front-end.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------- erros padronizados
# Middleware de id de requisição + tratadores de exceção (formato ErroAPI). Ver api/erros.py.
tratamento.registrar(app)


def erros(*status: int) -> dict:
    """Documenta em /docs que a rota pode devolver esses status, todos no formato ErroAPI."""
    return {s: {"model": ErroAPI} for s in status}


# ---------------------------------------------------------------- rotas
@app.get("/", response_model=StatusAPI, tags=["status"])
def raiz() -> StatusAPI:
    """Health check simples: confirma que a API está no ar."""
    return StatusAPI(status="ok", app="Éter API", versao=app.version, docs="/docs")


@app.get("/cidades", response_model=list[Cidade], tags=["consulta"])
def listar_cidades() -> list[Cidade]:
    """GET — lista as capitais suportadas, com latitude/longitude (usadas na API OpenWeather)."""
    return [Cidade(nome=nome, latitude=lat, longitude=lon) for nome, (lat, lon) in CIDADES.items()]


@app.get("/iqar/tabela", response_model=list[FaixaReferencia], tags=["consulta"],
         responses=erros(404, 422, 500, 503))
def tabela_iqar(
    poluente: str | None = Query(None, description="Filtra por sigla (ex.: MP2.5). Omita para trazer tudo."),
) -> list[FaixaReferencia]:
    """GET — devolve a tabela oficial de faixas do IQAr (gerada por `scripts/scraping/scrape_iqar.py`).

    Útil para qualquer cliente (o app, um notebook, outro serviço) que precise
    das faixas oficiais sem reimplementar a metodologia CETESB/CONAMA 491/2018.
    """
    if not dl.ARQ_IQAR.exists():
        raise ErroHTTP(503, "dados_indisponiveis", "A tabela de referência do IQAr ainda não foi gerada.",
                       "Rode `python scripts/scraping/scrape_iqar.py` na raiz do projeto.")
    tabela = _ler_csv_seguro(dl.ARQ_IQAR)
    if poluente:
        sigla = dl.normalizar_poluente(poluente)
        tabela = tabela[tabela["poluente"] == sigla]
        if tabela.empty:
            raise ErroHTTP(404, "poluente_nao_encontrado", f"Poluente '{poluente}' não encontrado.",
                           f"Use um de {sorted(dl.POLUENTES_VALIDOS)} (PM2.5 também é aceito).")
    # astype(object) antes do where: numa coluna float, o None voltaria a ser NaN,
    # que não é um número válido (a regra conc_max >= 0 do modelo de resposta recusa).
    tabela = tabela.astype(object).where(pd.notnull(tabela), None)
    return [FaixaReferencia(**linha) for linha in tabela.to_dict(orient="records")]


@app.get("/iqar/calcular", response_model=ResultadoIQAr, tags=["consulta"],
         responses=erros(400, 422))
def calcular(
    poluente: str = Query(..., description="Sigla do poluente (ex.: MP2.5, O3, CO)."),
    concentracao: float = Query(..., ge=0, description="Concentração na unidade oficial do poluente."),
) -> ResultadoIQAr:
    """GET — calcula o IQAr de uma concentração pontual, sem precisar registrar nada.

    Equivalente à calculadora da aba *Tabela IQAr* do app, mas acessível por
    qualquer cliente HTTP (curl, Postman, outro serviço).
    """
    sigla = dl.normalizar_poluente(poluente)
    try:
        idx = calcular_indice(sigla, concentracao)
    except ValueError as erro:
        raise ErroHTTP(400, "poluente_invalido", f"Poluente desconhecido: '{poluente}'.",
                       f"Use um de {sorted(dl.POLUENTES_VALIDOS)} (PM2.5 também é aceito).") from erro
    cat = categoria(idx)
    return ResultadoIQAr(poluente=sigla, concentracao=concentracao,
                         iqar=idx, categoria=cat, recomendacao=RECOMENDACOES[cat])


def _ler_csv_seguro(caminho: Path) -> pd.DataFrame:
    """Lê um CSV de dados do projeto; arquivo ilegível vira erro 500 claro, não um traceback."""
    try:
        return pd.read_csv(caminho)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as erro:
        raise ErroHTTP(500, "dados_corrompidos",
                       f"Não foi possível ler {caminho.relative_to(RAIZ)} ({type(erro).__name__}).",
                       "Confira o arquivo ou gere-o de novo. Se for o de leituras, mova-o para "
                       "outro lugar: a API cria um novo no próximo POST /leituras.") from erro


def _carregar_leituras() -> pd.DataFrame:
    if ARQ_LEITURAS_API.exists():
        df = _ler_csv_seguro(ARQ_LEITURAS_API)
        if faltando := set(COLUNAS_LEITURAS_API) - set(df.columns):
            raise ErroHTTP(500, "dados_corrompidos",
                           f"O arquivo de leituras não tem as colunas {sorted(faltando)}.",
                           "Mova data/processed/leituras_api.csv para outro lugar; a API cria "
                           "um novo no próximo POST /leituras.")
        return df
    return pd.DataFrame(columns=COLUNAS_LEITURAS_API)


@app.get("/leituras", response_model=list[LeituraRegistrada], tags=["leituras"],
         responses=erros(422, 500))
def listar_leituras(
    cidade: str | None = None,
    limite: int = Query(100, ge=1, le=1000),
) -> list[LeituraRegistrada]:
    """GET — lista as leituras já registradas via `POST /leituras` (as mais recentes primeiro)."""
    df = _carregar_leituras()
    if cidade:
        df = df[df["cidade"].str.casefold() == cidade.casefold()]
    df = df.sort_values("recebido_em", ascending=False).head(limite)
    try:
        return [LeituraRegistrada(**linha) for linha in df.to_dict(orient="records")]
    except ValidationError as erro:  # linha editada à mão com valor fora do contrato
        raise ErroHTTP(500, "dados_corrompidos",
                       f"Há leituras gravadas fora do formato esperado ({erro.errors()[0]['msg']}).",
                       "Corrija ou remova as linhas inválidas de data/processed/leituras_api.csv.") from erro


@app.post("/leituras", response_model=LeituraRegistrada, status_code=201, tags=["leituras"],
          responses=erros(400, 422, 500, 503))
def registrar_leitura(leitura: NovaLeitura) -> LeituraRegistrada:
    """POST — registra uma nova leitura (de um sensor, estação local ou relatório manual).

    Valida o poluente, calcula o IQAr correspondente e acrescenta o registro em
    `data/processed/leituras_api.csv`, que passa a alimentar as demais rotas de
    consulta. É o mesmo tipo de dado que a aba *Meus dados* aceita por upload de
    CSV, só que por integração direta (POST) em vez de arquivo.
    """
    sigla = dl.normalizar_poluente(leitura.poluente)
    if sigla not in dl.POLUENTES_VALIDOS:
        raise ErroHTTP(400, "poluente_invalido", f"Poluente inválido: '{leitura.poluente}'.",
                       f"Use um de {sorted(dl.POLUENTES_VALIDOS)} (PM2.5 também é aceito).")

    idx = calcular_indice(sigla, leitura.valor)
    registro = {
        "cidade": leitura.cidade.strip(),
        "poluente": sigla,
        "valor": leitura.valor,
        "data": (leitura.data or date.today()).isoformat(),
        "origem": leitura.origem,
        "recebido_em": datetime.now().isoformat(timespec="seconds"),
        "iqar": idx,
        "categoria": categoria(idx),
    }

    df = _carregar_leituras()
    novo = pd.DataFrame([registro])
    # Na primeira leitura não há o que juntar; concatenar com uma tabela vazia gera
    # FutureWarning no pandas (o comportamento com tabelas vazias vai mudar).
    df = novo if df.empty else pd.concat([df, novo], ignore_index=True)
    try:
        ARQ_LEITURAS_API.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARQ_LEITURAS_API, index=False)
    except OSError as erro:  # sem permissão, disco cheio, arquivo aberto no Excel...
        raise ErroHTTP(503, "armazenamento_indisponivel",
                       f"Não foi possível gravar a leitura ({type(erro).__name__}).",
                       "Feche o arquivo data/processed/leituras_api.csv se estiver aberto em "
                       "outro programa e tente de novo.") from erro

    return LeituraRegistrada(
        cidade=registro["cidade"], poluente=sigla, valor=leitura.valor,
        data=leitura.data or date.today(), origem=leitura.origem,
        recebido_em=datetime.fromisoformat(registro["recebido_em"]),
        iqar=idx, categoria=registro["categoria"],
    )


# ---------------------------------------------------------------- LLM local (TP4)
@app.get("/llm/status", response_model=StatusLLM, tags=["llm"])
def status_llm() -> StatusLLM:
    """GET — modelo configurado, se está carregado/ocupado e a última falha (diagnóstico)."""
    return StatusLLM(**llm.status())


def _montar_analise(tarefa: str, texto: str, resposta: str):
    """Transforma o texto cru do modelo no formato estruturado de cada tarefa.

    Os modelos Pydantic validam o resultado (ex.: resumo vazio é recusado). Se a
    validação falhar, quem chama devolve 502: o problema é a saída do modelo, não o
    pedido do cliente.
    """
    if tarefa == "resumo":
        n_entrada, n_resumo = len(texto.split()), len(resposta.split())
        return AnaliseResumo(resumo=resposta, palavras_entrada=n_entrada,
                             palavras_resumo=n_resumo,
                             taxa_compressao=round(n_resumo / max(n_entrada, 1), 2))
    if tarefa == "sentimento":
        rotulo = rotulo_sentimento(resposta)
        return AnaliseSentimento(rotulo=rotulo, rotulo_valido=rotulo is not None,
                                 resposta_modelo=resposta)
    return AnalisePergunta(pergunta=texto, resposta=resposta,
                           poluentes_consultados=contexto_iqar(texto)[1])


@app.post("/processar_texto", response_model=RespostaProcessamento, tags=["llm"],
          responses=erros(422, 500, 502, 503, 504))
def processar_texto(pedido: PedidoTexto) -> RespostaProcessamento:
    """POST — processa um texto com o LLM que roda **localmente** (Transformers, sem nuvem).

    Tarefas:
    * `resumo`: resumo em até duas frases, com contagem de palavras e taxa de compressão;
    * `sentimento`: rótulo positivo/negativo/neutro, mais a resposta exata do modelo;
    * `pergunta`: resposta baseada na tabela oficial do IQAr, com os poluentes consultados.

    Erros possíveis (todos no formato ErroAPI, com `sugestao` e `id_requisicao`):
    * 422 `entrada_invalida`: texto vazio, curto/longo demais ou tarefa inexistente;
    * 503 `llm_dependencias_ausentes`, `llm_modelo_nao_encontrado`, `llm_memoria_insuficiente`,
      `llm_falha_carga`: o modelo não pôde ser carregado;
    * 503 `llm_ocupado` (com cabeçalho Retry-After): outro pedido está em processamento;
    * 504 `llm_tempo_esgotado`: a geração passou de ETER_LLM_TIMEOUT_S;
    * 502 `resposta_invalida_do_modelo`: o modelo respondeu fora do formato (ex.: vazio).

    O campo `analise.tipo` indica qual dos três formatos veio. O primeiro pedido carrega
    o modelo e pode levar de segundos a alguns minutos; os seguintes reaproveitam o
    modelo já carregado. A rota é síncrona (`def`, não `async def`): o FastAPI a executa
    numa thread separada, então a geração, que é pesada, não trava as outras rotas.
    """
    texto = pedido.texto.strip()
    try:
        saida = llm.processar(pedido.tarefa, texto)
    except LLMErro as erro:  # cada falha do modelo vira um status e uma orientação próprios
        raise erro_do_llm(erro) from erro
    try:
        analise = _montar_analise(pedido.tarefa, texto, saida["resultado"])
    except ValidationError as erro:
        raise ErroHTTP(502, "resposta_invalida_do_modelo",
                       f"O modelo gerou uma resposta fora do formato esperado "
                       f"({erro.errors()[0]['msg']}).",
                       "Tente novamente ou reformule o texto (ex.: deixe-o mais longo e claro).") from erro
    return RespostaProcessamento(
        tarefa=pedido.tarefa, texto_entrada=texto, analise=analise,
        metadados=MetadadosLLM(modelo=llm.repo, formato=llm.formato,
                               tokens_gerados=saida["tokens_gerados"], tempo_s=saida["tempo_s"],
                               processado_em=datetime.now()),
    )
