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
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.append(str(RAIZ))

from src import data_loader as dl  # noqa: E402
from src.api import CIDADES  # noqa: E402
from src.iqar import RECOMENDACOES, calcular_indice, categoria  # noqa: E402

from api.schemas import (  # noqa: E402
    Cidade,
    FaixaReferencia,
    LeituraRegistrada,
    NovaLeitura,
    ResultadoIQAr,
)

ARQ_LEITURAS_API = dl.DATA_RAW.parent / "processed" / "leituras_api.csv"
COLUNAS_LEITURAS_API = ["cidade", "poluente", "valor", "data", "origem",
                        "recebido_em", "iqar", "categoria"]

app = FastAPI(
    title="Éter API",
    description="Dados e cálculo de qualidade do ar (IQAr) do projeto Éter.",
    version="1.0.0",
)

# Em produção, restrinja allow_origins ao domínio real do front-end.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/", tags=["status"])
def raiz() -> dict:
    """Health check simples: confirma que a API está no ar."""
    return {"status": "ok", "app": "Éter API", "docs": "/docs"}


@app.get("/cidades", response_model=list[Cidade], tags=["consulta"])
def listar_cidades() -> list[Cidade]:
    """GET — lista as capitais suportadas, com latitude/longitude (usadas na API OpenWeather)."""
    return [Cidade(nome=nome, latitude=lat, longitude=lon) for nome, (lat, lon) in CIDADES.items()]


@app.get("/iqar/tabela", response_model=list[FaixaReferencia], tags=["consulta"])
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
    tabela = tabela.where(pd.notnull(tabela), None)
    return [FaixaReferencia(**linha) for linha in tabela.to_dict(orient="records")]


@app.get("/iqar/calcular", response_model=ResultadoIQAr, tags=["consulta"])
def calcular(
    poluente: str = Query(..., description="Sigla do poluente (ex.: MP2.5, O3, CO)."),
    concentracao: float = Query(..., ge=0, description="Concentração na unidade oficial do poluente."),
) -> ResultadoIQAr:
    """GET — calcula o IQAr de uma concentração pontual, sem precisar registrar nada.

    Equivalente à calculadora da aba *Tabela IQAr* do app, mas acessível por
    qualquer cliente HTTP (curl, Postman, outro serviço).
    """
    try:
        idx = calcular_indice(poluente.upper(), concentracao)
    except ValueError as erro:
        raise HTTPException(400, str(erro)) from erro
    cat = categoria(idx)
    return ResultadoIQAr(poluente=poluente.upper(), concentracao=concentracao,
                         iqar=idx, categoria=cat, recomendacao=RECOMENDACOES[cat])


def _carregar_leituras() -> pd.DataFrame:
    if ARQ_LEITURAS_API.exists():
        return pd.read_csv(ARQ_LEITURAS_API)
    return pd.DataFrame(columns=COLUNAS_LEITURAS_API)


@app.get("/leituras", response_model=list[LeituraRegistrada], tags=["leituras"])
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


@app.post("/leituras", response_model=LeituraRegistrada, status_code=201, tags=["leituras"])
def registrar_leitura(leitura: NovaLeitura) -> LeituraRegistrada:
    """POST — registra uma nova leitura (de um sensor, estação local ou relatório manual).

    Valida o poluente, calcula o IQAr correspondente e acrescenta o registro em
    `data/processed/leituras_api.csv`, que passa a alimentar as demais rotas de
    consulta. É o mesmo tipo de dado que a aba *Meus dados* aceita por upload de
    CSV, só que por integração direta (POST) em vez de arquivo.
    """
    sigla = leitura.poluente.strip().upper().replace("PM", "MP")
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
