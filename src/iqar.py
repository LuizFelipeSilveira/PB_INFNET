"""Cálculo do Índice de Qualidade do Ar (IQAr) no padrão CETESB / CONAMA 491/2018.

IQAr = Ii + (If - Ii) / (Cf - Ci) * (C - Ci)

Observação: os limites oficiais usam médias (1h, 8h ou 24h). Como a API entrega
leituras horárias, o índice calculado aqui é uma APROXIMAÇÃO para fins educativos.
"""
from __future__ import annotations

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

# Limites de concentração por poluente (Ci, Cf) para cada faixa.
# Unidades: µg/m³, exceto CO em ppm. A última faixa usa um teto técnico para interpolação.
LIMITES = {
    "MP10": [(0, 50), (50, 100), (100, 150), (150, 250), (250, 600)],
    "MP2.5": [(0, 25), (25, 50), (50, 75), (75, 125), (125, 300)],
    "O3": [(0, 100), (100, 130), (130, 160), (160, 200), (200, 800)],
    "CO": [(0, 9), (9, 11), (11, 13), (13, 15), (15, 50)],
    "NO2": [(0, 200), (200, 240), (240, 320), (320, 1130), (1130, 3750)],
    "SO2": [(0, 20), (20, 40), (40, 365), (365, 800), (800, 2620)],
}

NOMES = {
    "MP10": "Material Particulado (MP10)",
    "MP2.5": "Material Particulado (MP2.5)",
    "O3": "Ozônio (O3)",
    "CO": "Monóxido de Carbono (CO)",
    "NO2": "Dióxido de Nitrogênio (NO2)",
    "SO2": "Dióxido de Enxofre (SO2)",
}

UNIDADES = {p: ("ppm" if p == "CO" else "µg/m³") for p in LIMITES}

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
    c = max(0.0, float(concentracao))
    for (ci, cf), (ii, if_) in zip(LIMITES[poluente], FAIXAS_INDICE):
        if c <= cf:
            return round(ii + (if_ - ii) / (cf - ci) * (c - ci))
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
