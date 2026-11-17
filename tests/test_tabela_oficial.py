"""Testes da extração do Quadro 1 (PDF do MMA) e do cálculo do IQAr pela CONAMA 506/2024."""
from __future__ import annotations

from pathlib import Path

import pytest

from src import iqar
from src.tabela_oficial import TabelaInvalida, extrair_quadro1, validar

PDF = Path(__file__).resolve().parents[1] / "data" / "external" / "mma_orientacao_tecnica_iqar_jan2025.pdf"


@pytest.mark.skipif(not PDF.exists(), reason="cópia do PDF do MMA ausente")
def test_extrai_o_quadro_1_do_pdf_do_mma():
    pytest.importorskip("pdfplumber")
    limites, indices = extrair_quadro1(PDF)
    validar(limites, indices)
    assert indices == [(0, 40), (41, 80), (81, 120), (121, 200), (201, 400)]
    assert limites["MP10"][0] == (0, 45)          # mudou em relação à CONAMA 491/2018 (0–50)
    assert limites["MP2.5"][:2] == [(0, 15), (15, 50)]
    assert limites["SO2"] == [(0, 40), (40, 50), (50, 125), (125, 800), (800, 2620)]
    assert limites["CO"][1] == (9, 11)
    assert limites == iqar.LIMITES_REFERENCIA      # a reserva no código bate com o PDF


def test_exemplo_resolvido_da_orientacao_tecnica():
    """Item 1.2 do PDF: MP10 = 210, O3 = 135 e NO2 = 220 dão 168, 86 e 60 (Muito Ruim)."""
    assert iqar.calcular_indice("MP10", 210) == 168
    assert iqar.calcular_indice("O3", 135) == 86
    assert iqar.calcular_indice("NO2", 220) == 60
    av = iqar.avaliar({"MP10": 210, "O3": 135, "NO2": 220})
    assert av["indice"] == 168 and av["categoria"] == "Muito Ruim"


@pytest.mark.parametrize("poluente, conc, indice", [
    ("MP10", 45, 40), ("MP10", 46, 41),          # fim de "Boa" e início de "Moderada"
    ("MP2.5", 15, 40), ("MP2.5", 20, 46),        # MP2.5 = 20 já é Moderada na 506/2024
    ("CO", 9.0, 40), ("CO", 9.04, 40), ("CO", 9.1, 41),  # CO arredondado com 1 casa
    ("SO2", 900, 212), ("MP10", 700, 400),       # SO2 Péssima; acima do teto -> 400
])
def test_bordas_das_faixas(poluente, conc, indice):
    assert iqar.calcular_indice(poluente, conc) == indice


def test_validacao_recusa_tabela_sem_sentido():
    limites = {p: list(f) for p, f in iqar.LIMITES_REFERENCIA.items()}
    indices = [(0, 40), (41, 80), (81, 120), (121, 200), (201, 400)]
    validar(limites, indices)  # a correta passa
    with pytest.raises(TabelaInvalida):
        validar({**limites, "MP10": limites["MP10"][:4]}, indices)              # faixa faltando
    with pytest.raises(TabelaInvalida):
        validar({**limites, "O3": [(0, 100), (110, 130)] + limites["O3"][2:]}, indices)  # buraco
    with pytest.raises(TabelaInvalida):
        validar(limites, indices[:4] + [(201, 500)])                            # índice errado


def test_csv_invalido_usa_a_reserva_do_codigo(tmp_path):
    ruim = tmp_path / "tabela_iqar.csv"
    ruim.write_text("poluente,conc_min,conc_max,indice_min,indice_max\nMP10,0,45,0,40\n",
                    encoding="utf-8")
    limites, fonte = iqar.carregar_limites(ruim)
    assert limites == iqar.LIMITES_REFERENCIA and fonte == iqar.FONTE_REFERENCIA
    limites, fonte = iqar.carregar_limites(tmp_path / "nao_existe.csv")
    assert fonte == iqar.FONTE_REFERENCIA
