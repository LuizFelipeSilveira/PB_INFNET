"""Extração e validação da tabela oficial do IQAr (Quadro 1 do PDF do MMA).

O Ministério do Meio Ambiente publica as faixas do IQAr da Resolução CONAMA
506/2024 na "Orientação técnica: índice de qualidade do ar" (jan/2025), em PDF.
O Quadro 1 desse PDF tem, para cada categoria (N1 Boa ... N5 Péssima), a faixa
do índice e a faixa de concentração de cada poluente.

Este módulo:
  * `extrair_quadro1(pdf)`: lê o PDF com pdfplumber e devolve as faixas;
  * `validar(limites, indices)`: confere se a tabela faz sentido (6 poluentes,
    5 faixas cada, faixas contínuas e crescentes) antes de o app confiar nela.

Usado por scripts/scraping/scrape_iqar.py (coleta) e por src/iqar.py (que só
aceita a tabela gravada em data/raw/tabela_iqar.csv se ela passar na validação).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

POLUENTES = ["MP10", "MP2.5", "O3", "CO", "NO2", "SO2"]
FAIXAS_INDICE = [(0, 40), (41, 80), (81, 120), (121, 200), (201, 400)]

# "0 – 45", ">45 - 100", ">9 – 11", "0,5 – 1,0" (traço comum ou meia-risca)
_FAIXA = re.compile(r"^>?\s*(\d+(?:,\d+)?)\s*[–-]\s*(\d+(?:,\d+)?)$")
_SIGLA_PDF = {"MP10": "MP10", "MP2,5": "MP2.5", "MP2.5": "MP2.5", "O3": "O3", "CO": "CO",
              "NO2": "NO2", "SO2": "SO2"}


class TabelaInvalida(ValueError):
    """O PDF não tem o Quadro 1 no formato esperado, ou os números não fazem sentido."""


def _numero(texto: str) -> float:
    return float(texto.replace(",", "."))


def _faixa(celula: str | None) -> tuple[float, float] | None:
    m = _FAIXA.match((celula or "").strip())
    return (_numero(m.group(1)), _numero(m.group(2))) if m else None


def extrair_quadro1(pdf: str | Path | bytes) -> tuple[dict[str, list[tuple[float, float]]],
                                                     list[tuple[int, int]]]:
    """Lê o Quadro 1 do PDF. Devolve ({poluente: [(c_ini, c_fim) x 5]}, [(i_ini, i_fim) x 5]).

    O pdfplumber devolve a tabela como linhas e colunas, mas o PDF quebra algumas
    células em duas linhas (ex.: "N2 –" / "Moderada", e a faixa do índice numa linha
    e as concentrações na seguinte). Por isso a leitura é pela COLUNA: primeiro
    descobre em que coluna está cada poluente (pelo cabeçalho) e depois junta, para
    cada categoria, a faixa do índice com as 6 faixas de concentração.
    """
    import pdfplumber

    fonte = io.BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    with pdfplumber.open(fonte) as documento:
        for pagina in documento.pages:
            if "Quadro 1" not in (pagina.extract_text() or ""):
                continue
            for tabela in pagina.extract_tables():
                resultado = _ler_tabela(tabela)
                if resultado:
                    return resultado
    raise TabelaInvalida("Quadro 1 com as faixas do IQAr não encontrado no PDF.")


def _ler_tabela(linhas: list[list[str | None]]):
    colunas = {}  # índice da coluna -> sigla do poluente
    for linha in linhas:
        for j, celula in enumerate(linha):
            sigla = _SIGLA_PDF.get((celula or "").strip())
            if sigla:
                colunas[j] = sigla
    if sorted(colunas.values()) != sorted(POLUENTES):
        return None

    limites = {p: [] for p in POLUENTES}
    indices, indice_pendente = [], None
    for linha in linhas:
        concentracoes = {colunas[j]: _faixa(linha[j]) for j in colunas if j < len(linha)}
        concentracoes = {p: f for p, f in concentracoes.items() if f}
        # a faixa do índice fica na coluna anterior à do primeiro poluente
        col_indice = min(colunas) - 1
        indice = _faixa(linha[col_indice]) if col_indice < len(linha) else None
        if len(concentracoes) == len(POLUENTES):
            indice = indice or indice_pendente
            if indice is None:
                raise TabelaInvalida("Linha de concentrações sem a faixa do índice.")
            indices.append((int(indice[0]), int(indice[1])))
            for p in POLUENTES:
                limites[p].append(concentracoes[p])
            indice_pendente = None
        elif indice and not concentracoes:
            indice_pendente = indice  # índice numa linha, concentrações na seguinte
    return limites, indices


def validar(limites: dict[str, list[tuple[float, float]]], indices: list[tuple[int, int]]) -> None:
    """Levanta TabelaInvalida se a tabela não tiver a estrutura do IQAr."""
    if [tuple(i) for i in indices] != FAIXAS_INDICE:
        raise TabelaInvalida(f"Faixas do índice inesperadas: {indices}")
    if sorted(limites) != sorted(POLUENTES):
        raise TabelaInvalida(f"Poluentes inesperados: {sorted(limites)}")
    for pol, faixas in limites.items():
        if len(faixas) != len(FAIXAS_INDICE):
            raise TabelaInvalida(f"{pol}: {len(faixas)} faixas (esperado 5).")
        if faixas[0][0] != 0:
            raise TabelaInvalida(f"{pol}: a primeira faixa deveria começar em 0.")
        for (ci, cf), (prox_ci, _) in zip(faixas, faixas[1:] + [(None, None)]):
            if not cf > ci:
                raise TabelaInvalida(f"{pol}: faixa {ci}–{cf} não é crescente.")
            if prox_ci is not None and prox_ci != cf:
                raise TabelaInvalida(f"{pol}: faixas não contínuas ({cf} -> {prox_ci}).")
