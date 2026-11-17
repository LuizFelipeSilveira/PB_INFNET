"""Modelos Pydantic da API (entrada e saída das rotas)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

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
    valor: float = Field(..., ge=0, description="Concentração na unidade oficial do poluente.")
    data: date | None = Field(None, description="Data da medição. Se omitida, usa a data atual.")
    origem: str = Field("api", description="De onde veio a medição (sensor, estação, etc.).")


class LeituraRegistrada(NovaLeitura):
    """O que a API devolve depois de registrar uma leitura (inclui o IQAr calculado)."""

    recebido_em: datetime
    iqar: int
    categoria: str


class PedidoTexto(BaseModel):
    """Corpo esperado por `POST /processar_texto`."""

    texto: str = Field(..., min_length=3, max_length=4000,
                       examples=["Fumaça das queimadas deixa o ar de Manaus em nível péssimo e "
                                 "escolas suspendem as aulas nesta segunda-feira."])
    tarefa: Literal["resumo", "sentimento", "pergunta"] = Field(
        "resumo",
        description="resumo: resume o texto em até duas frases; sentimento: positivo, negativo "
                    "ou neutro; pergunta: responde usando a tabela oficial do IQAr.")


class ResultadoTexto(BaseModel):
    """Resposta de `POST /processar_texto`."""

    tarefa: str
    resultado: str = Field(..., description="Texto gerado pelo modelo.")
    sentimento: Literal["positivo", "negativo", "neutro"] | None = Field(
        None, description="Rótulo padronizado (só na tarefa 'sentimento'; vazio se o modelo "
                          "não respondeu um rótulo válido).")
    modelo: str
    formato: str = Field(..., description="Quantização e dispositivo, ex.: 4bit-gpu ou fp32-cpu.")
    tokens_gerados: int
    tempo_s: float


class StatusLLM(BaseModel):
    modelo: str
    origem: str = Field(..., description="Pasta local (modelos/...) ou nome no Hugging Face.")
    carregado: bool
    formato: str | None = None
    tempo_carga_s: float | None = None
