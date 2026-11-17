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
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))

from src import data_loader as dl  # noqa: E402
from src.api import CIDADES  # noqa: E402
from src.iqar import RECOMENDACOES, calcular_indice, categoria  # noqa: E402

from src.llm import LLMLocal, contexto_iqar, rotulo_sentimento  # noqa: E402

from api.schemas import (  # noqa: E402
    AnalisePergunta,
    AnaliseResumo,
    AnaliseSentimento,
    CampoInvalido,
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
    version="1.1.0",
)

# Em produção, restrinja allow_origins ao domínio real do front-end.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------- erros padronizados
# Todo erro sai no formato `ErroAPI` (ver api/schemas.py): {status, erro, mensagem, campos}.
CODIGOS_ERRO = {400: "requisicao_invalida", 404: "nao_encontrado", 422: "entrada_invalida",
                502: "resposta_invalida_do_modelo", 503: "llm_indisponivel"}


def _erro(status: int, mensagem: str, campos: list[CampoInvalido] | None = None) -> JSONResponse:
    corpo = ErroAPI(status=status, erro=CODIGOS_ERRO.get(status, "erro"), mensagem=mensagem,
                    campos=campos or [])
    return JSONResponse(status_code=status, content=corpo.model_dump())


def _traduzir(e: dict) -> str:
    """Mensagens do Pydantic em português, para os casos mais comuns."""
    ctx = e.get("ctx", {})
    return {
        "missing": "campo obrigatório",
        "string_too_short": f"deve ter pelo menos {ctx.get('min_length')} caracteres",
        "string_too_long": f"deve ter no máximo {ctx.get('max_length')} caracteres",
        "literal_error": f"valor inválido; use um de: {ctx.get('expected')}",
        "greater_than_equal": f"deve ser maior ou igual a {ctx.get('ge')}",
        "less_than_equal": f"deve ser menor ou igual a {ctx.get('le')}",
        "float_parsing": "deve ser um número",
        "int_parsing": "deve ser um número inteiro",
        "date_from_datetime_parsing": "deve ser uma data no formato AAAA-MM-DD",
        "extra_forbidden": "campo não reconhecido por esta rota",
        "json_invalid": "o corpo não é um JSON válido",
        "model_attributes_type": "o corpo deve ser um objeto JSON",
    }.get(e["type"], e["msg"])


@app.exception_handler(RequestValidationError)
def _erro_validacao(_: Request, exc: RequestValidationError) -> JSONResponse:
    campos = [CampoInvalido(campo=".".join(str(p) for p in e["loc"][1:]) or e["loc"][0],
                            mensagem=_traduzir(e)) for e in exc.errors()]
    return _erro(422, "A requisição tem campos inválidos. Veja a lista em 'campos'.", campos)


@app.exception_handler(StarletteHTTPException)
def _erro_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _erro(exc.status_code, str(exc.detail))


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
         responses=erros(404, 422))
def tabela_iqar(
    poluente: str | None = Query(None, description="Filtra por sigla (ex.: MP2.5). Omita para trazer tudo."),
) -> list[FaixaReferencia]:
    """GET — devolve a tabela oficial de faixas do IQAr (gerada por `scripts/scraping/scrape_iqar.py`).

    Útil para qualquer cliente (o app, um notebook, outro serviço) que precise
    das faixas oficiais sem reimplementar a metodologia CETESB/CONAMA 491/2018.
    """
    if not dl.ARQ_IQAR.exists():
        raise HTTPException(404, "Tabela de referência ainda não gerada. "
                                  "Rode scripts/scraping/scrape_iqar.py.")
    tabela = pd.read_csv(dl.ARQ_IQAR)
    if poluente:
        tabela = tabela[tabela["poluente"].str.upper() == poluente.upper()]
        if tabela.empty:
            raise HTTPException(404, f"Poluente '{poluente}' não encontrado.")
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
        raise HTTPException(400, str(erro)) from erro
    cat = categoria(idx)
    return ResultadoIQAr(poluente=sigla, concentracao=concentracao,
                         iqar=idx, categoria=cat, recomendacao=RECOMENDACOES[cat])


def _carregar_leituras() -> pd.DataFrame:
    if ARQ_LEITURAS_API.exists():
        return pd.read_csv(ARQ_LEITURAS_API)
    return pd.DataFrame(columns=COLUNAS_LEITURAS_API)


@app.get("/leituras", response_model=list[LeituraRegistrada], tags=["leituras"],
         responses=erros(422))
def listar_leituras(
    cidade: str | None = None,
    limite: int = Query(100, ge=1, le=1000),
) -> list[LeituraRegistrada]:
    """GET — lista as leituras já registradas via `POST /leituras` (as mais recentes primeiro)."""
    df = _carregar_leituras()
    if cidade:
        df = df[df["cidade"].str.casefold() == cidade.casefold()]
    df = df.sort_values("recebido_em", ascending=False).head(limite)
    return [LeituraRegistrada(**linha) for linha in df.to_dict(orient="records")]


@app.post("/leituras", response_model=LeituraRegistrada, status_code=201, tags=["leituras"],
          responses=erros(400, 422))
def registrar_leitura(leitura: NovaLeitura) -> LeituraRegistrada:
    """POST — registra uma nova leitura (de um sensor, estação local ou relatório manual).

    Valida o poluente, calcula o IQAr correspondente e acrescenta o registro em
    `data/processed/leituras_api.csv`, que passa a alimentar as demais rotas de
    consulta. É o mesmo tipo de dado que a aba *Meus dados* aceita por upload de
    CSV, só que por integração direta (POST) em vez de arquivo.
    """
    sigla = dl.normalizar_poluente(leitura.poluente)
    if sigla not in dl.POLUENTES_VALIDOS:
        raise HTTPException(400, f"Poluente inválido: {leitura.poluente}. "
                                  f"Use um de {sorted(dl.POLUENTES_VALIDOS)}.")

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

    ARQ_LEITURAS_API.parent.mkdir(parents=True, exist_ok=True)
    df = _carregar_leituras()
    df = pd.concat([df, pd.DataFrame([registro])], ignore_index=True)
    df.to_csv(ARQ_LEITURAS_API, index=False)

    return LeituraRegistrada(
        cidade=registro["cidade"], poluente=sigla, valor=leitura.valor,
        data=leitura.data or date.today(), origem=leitura.origem,
        recebido_em=datetime.fromisoformat(registro["recebido_em"]),
        iqar=idx, categoria=registro["categoria"],
    )


# ---------------------------------------------------------------- LLM local (TP4)
@app.get("/llm/status", response_model=StatusLLM, tags=["llm"])
def status_llm() -> StatusLLM:
    """GET — informa qual modelo está configurado e se já foi carregado na memória."""
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
          responses=erros(422, 502, 503))
def processar_texto(pedido: PedidoTexto) -> RespostaProcessamento:
    """POST — processa um texto com o LLM que roda **localmente** (Transformers, sem nuvem).

    Tarefas:
    * `resumo`: resumo em até duas frases, com contagem de palavras e taxa de compressão;
    * `sentimento`: rótulo positivo/negativo/neutro, mais a resposta exata do modelo;
    * `pergunta`: resposta baseada na tabela oficial do IQAr, com os poluentes consultados.

    O campo `analise.tipo` indica qual dos três formatos veio. O primeiro pedido carrega
    o modelo e pode levar de segundos a alguns minutos; os seguintes reaproveitam o
    modelo já carregado. A rota é síncrona (`def`, não `async def`): o FastAPI a executa
    numa thread separada, então a geração, que é pesada, não trava as outras rotas.
    """
    texto = pedido.texto.strip()
    try:
        saida = llm.processar(pedido.tarefa, texto)
    except (ImportError, OSError, RuntimeError) as erro:
        raise HTTPException(503, f"O modelo de linguagem local não pôde ser carregado ({erro}). "
                                 "Instale as dependências de requirements-llm.txt e baixe o "
                                 "modelo com scripts/llm/baixar_modelo.py.") from erro
    try:
        analise = _montar_analise(pedido.tarefa, texto, saida["resultado"])
    except ValidationError as erro:
        raise HTTPException(502, "O modelo gerou uma resposta fora do formato esperado "
                                 f"({erro.errors()[0]['msg']}). Tente novamente ou reformule "
                                 "o texto.") from erro
    return RespostaProcessamento(
        tarefa=pedido.tarefa, texto_entrada=texto, analise=analise,
        metadados=MetadadosLLM(modelo=llm.repo, formato=llm.formato,
                               tokens_gerados=saida["tokens_gerados"], tempo_s=saida["tempo_s"],
                               processado_em=datetime.now()),
    )
