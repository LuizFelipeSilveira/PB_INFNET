"""Cálculo do Índice de Qualidade do Ar (IQAr), Resolução CONAMA 506/2024.

IQAr = Ii + (If - Ii) / (Cf - Ci) * (C - Ci)

De onde vêm as faixas (LIMITES):
  1. data/raw/tabela_iqar.csv, gerada por scripts/scraping/scrape_iqar.py a partir do
     Quadro 1 do PDF do MMA ("Orientação técnica: índice de qualidade do ar", jan/2025).
     Só é usada se passar na validação de src/tabela_oficial.py.
  2. Se o arquivo não existir ou estiver inválido: LIMITES_REFERENCIA, abaixo, uma
     cópia das mesmas faixas mantida no código como reserva.
`FONTE_LIMITES` diz qual das duas está em uso (o app e a API mostram essa informação).

Observação: os limites oficiais usam médias (1h, 8h ou 24h). Como a API entrega
leituras horárias, o índice calculado aqui é uma APROXIMAÇÃO para fins educativos.
"""
from __future__ import annotations

import csv
from pathlib import Path

from src.tabela_oficial import TabelaInvalida, validar

# Faixas do índice e categorias
FAIXAS_INDICE = [(0, 40), (41, 80), (81, 120), (121, 200), (201, 400)]
CATEGORIAS = ["Boa", "Moderada", "Ruim", "Muito Ruim", "Péssima"]
CORES = {
    "Boa": "#2E9E5B",
    "Moderada": "#E3B505",
    "Ruim": "#E8772E",
    "Muito Ruim": "#C62E3A",
    "Péssima": "#6B2C91",
}

# Reserva no código: Quadro 1 da orientação técnica do MMA (jan/2025), CONAMA 506/2024.
# Faixas de concentração (Ci, Cf) por poluente; µg/m³, exceto CO em ppm.
# (Até o TP3 o projeto usava a CONAMA 491/2018, que tinha MP10 "Boa" até 50,
# MP2.5 "Boa" até 25 e SO2 com faixas 20/40/365/800.)
LIMITES_REFERENCIA = {
    "MP10": [(0, 45), (45, 100), (100, 150), (150, 250), (250, 600)],
    "MP2.5": [(0, 15), (15, 50), (50, 75), (75, 125), (125, 300)],
    "O3": [(0, 100), (100, 130), (130, 160), (160, 200), (200, 800)],
    "CO": [(0, 9), (9, 11), (11, 13), (13, 15), (15, 50)],
    "NO2": [(0, 200), (200, 240), (240, 320), (320, 1130), (1130, 3750)],
    "SO2": [(0, 40), (40, 50), (50, 125), (125, 800), (800, 2620)],
}
FONTE_REFERENCIA = "Referência no código (Quadro 1 do MMA, CONAMA 506/2024)"
ARQ_TABELA = Path(__file__).resolve().parents[1] / "data" / "raw" / "tabela_iqar.csv"


def carregar_limites(caminho: Path = ARQ_TABELA) -> tuple[dict, str]:
    """Lê as faixas da tabela gerada pelo scraping; usa a reserva se ela faltar ou for inválida."""
    try:
        with caminho.open(encoding="utf-8") as arquivo:
            linhas = list(csv.DictReader(arquivo))
        limites = {}
        for linha in linhas:
            limites.setdefault(linha["poluente"], []).append(
                (float(linha["conc_min"]), float(linha["conc_max"])))
        indices = list(dict.fromkeys((int(linha["indice_min"]), int(linha["indice_max"]))
                                     for linha in linhas))
        validar(limites, indices)
        return limites, (linhas[0].get("fonte") or str(caminho.name))
    except (OSError, KeyError, ValueError, TabelaInvalida):
        return LIMITES_REFERENCIA, FONTE_REFERENCIA


LIMITES, FONTE_LIMITES = carregar_limites()

NOMES = {
    "MP10": "Material Particulado (MP10)",
    "MP2.5": "Material Particulado (MP2.5)",
    "O3": "Ozônio (O3)",
    "CO": "Monóxido de Carbono (CO)",
    "NO2": "Dióxido de Nitrogênio (NO2)",
    "SO2": "Dióxido de Enxofre (SO2)",
}

UNIDADES = {p: ("ppm" if p == "CO" else "µg/m³") for p in LIMITES_REFERENCIA}

RECOMENDACOES = {
    "Boa": "Ar limpo. Aproveite para atividades ao ar livre sem restrições.",
    "Moderada": "Pessoas de grupos sensíveis (crianças, idosos e quem tem doenças "
                "respiratórias ou cardíacas) podem sentir tosse seca e cansaço. "
                "Para a população em geral, atividades externas seguem liberadas.",
    "Ruim": "Toda a população pode sentir tosse, ardor nos olhos, nariz e garganta. "
            "Grupos sensíveis devem reduzir esforço físico ao ar livre.",
    "Muito Ruim": "Aumento dos sintomas e falta de ar. Evite exercícios ao ar livre; "
                  "grupos sensíveis devem permanecer em ambientes internos.",
    "Péssima": "Risco sério à saúde de toda a população. Evite sair de casa e "
               "qualquer esforço físico ao ar livre.",
}


def co_ugm3_para_ppm(valor_ugm3: float) -> float:
    """Converte CO de µg/m³ para ppm (25 °C, 1 atm; massa molar 28,01 g/mol)."""
    return valor_ugm3 / 1000 / 1.145


def calcular_indice(poluente: str, concentracao: float) -> int:
    """Retorna o IQAr (0–400) de um poluente para a concentração informada."""
    if poluente not in LIMITES or concentracao is None:
        raise ValueError(f"Poluente desconhecido: {poluente}")
    # Arredondamento oficial antes do cálculo (orientação do MMA, item 1.1): CO com uma
    # casa decimal, os demais inteiros. O round() do Python já arredonda X,5 para o par
    # mais próximo, como a orientação pede.
    passo = 0.1 if poluente == "CO" else 1
    c = max(0.0, round(float(concentracao), 1 if poluente == "CO" else 0))
    for n, ((ci, cf), (ii, if_)) in enumerate(zip(LIMITES[poluente], FAIXAS_INDICE)):
        if c <= cf:
            # Da 2ª faixa em diante o Quadro 1 diz ">45 - 100": como a concentração já foi
            # arredondada, o primeiro valor possível é 46 (ou 9,1 no CO). O exemplo
            # resolvido do MMA (item 1.2) usa exatamente isso: O3 = 135 -> IQAr 86.
            c_ini = ci if n == 0 else round(ci + passo, 1)
            return round(ii + (if_ - ii) / (cf - c_ini) * (c - c_ini))
    return 400


def categoria(indice: int) -> str:
    for (ini, fim), nome in zip(FAIXAS_INDICE, CATEGORIAS):
        if indice <= fim:
            return nome
    return "Péssima"


def avaliar(concentracoes: dict[str, float]) -> dict:
    """Avalia um conjunto de leituras. O IQAr geral é o pior índice entre os poluentes."""
    detalhes = []
    for pol, valor in concentracoes.items():
        if pol in LIMITES and valor is not None:
            idx = calcular_indice(pol, valor)
            detalhes.append({
                "Poluente": NOMES[pol],
                "Sigla": pol,
                "Concentração": round(float(valor), 2),
                "Unidade": UNIDADES[pol],
                "IQAr": idx,
                "Classificação": categoria(idx),
            })
    if not detalhes:
        return {"indice": None, "categoria": None, "critico": None, "detalhes": []}
    pior = max(detalhes, key=lambda d: d["IQAr"])
    return {
        "indice": pior["IQAr"],
        "categoria": pior["Classificação"],
        "critico": pior["Poluente"],
        "detalhes": detalhes,
    }
