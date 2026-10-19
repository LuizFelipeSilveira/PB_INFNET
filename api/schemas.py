"""Modelos Pydantic da API (entrada e saída das rotas)."""
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from src.data_loader import POLUENTES_VALIDOS


class Cidade(BaseModel):
    nome: str
    latitude: float
    longitude: float


class FaixaReferencia(BaseModel):
    poluente: str
    nome: str
    unidade: str
    categoria: str
    indice_min: int
    indice_max: int
    conc_min: float
    conc_max: float | None = None
    recomendacao: str


class ResultadoIQAr(BaseModel):
    poluente: str
    concentracao: float
    iqar: int
    categoria: str
    recomendacao: str


class NovaLeitura(BaseModel):
    """Corpo esperado por `POST /leituras`."""

    cidade: str = Field(..., examples=["Rio de Janeiro - RJ"])
    poluente: str = Field(..., description=f"Um de: {sorted(POLUENTES_VALIDOS)}")
    valor: float = Field(..., gt=0, description="Concentração na unidade oficial do poluente.")
    data: date | None = Field(None, description="Data da medição. Se omitida, usa a data atual.")
    origem: str = Field("api", description="De onde veio a medição (sensor, estação, etc.).")


class LeituraRegistrada(NovaLeitura):
    """O que a API devolve depois de registrar uma leitura (inclui o IQAr calculado)."""

    recebido_em: datetime
    iqar: int
    categoria: str
