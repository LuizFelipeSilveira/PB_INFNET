"""Integração com a OpenWeather Air Pollution API (com modo simulado para testes)."""
from __future__ import annotations

import random
from datetime import datetime

import requests

from src.iqar import co_ugm3_para_ppm

URL_AIR = "https://api.openweathermap.org/data/2.5/air_pollution"

# Capitais brasileiras usadas no seletor (lat, lon)
CIDADES = {
    "Rio de Janeiro - RJ": (-22.9068, -43.1729),
    "São Paulo - SP": (-23.5505, -46.6333),
    "Belo Horizonte - MG": (-19.9167, -43.9345),
    "Brasília - DF": (-15.7939, -47.8828),
    "Salvador - BA": (-12.9777, -38.5016),
    "Fortaleza - CE": (-3.7319, -38.5267),
    "Recife - PE": (-8.0476, -34.8770),
    "Manaus - AM": (-3.1190, -60.0217),
    "Belém - PA": (-1.4558, -48.4902),
    "Curitiba - PR": (-25.4284, -49.2733),
    "Porto Alegre - RS": (-30.0346, -51.2177),
    "Florianópolis - SC": (-27.5954, -48.5480),
    "Goiânia - GO": (-16.6869, -49.2648),
    "Cuiabá - MT": (-15.6014, -56.0979),
    "Porto Velho - RO": (-8.7612, -63.9004),
}


def _normalizar(componentes: dict) -> dict[str, float]:
    """Converte o formato da OpenWeather para as siglas usadas no IQAr."""
    return {
        "MP2.5": componentes.get("pm2_5"),
        "MP10": componentes.get("pm10"),
        "O3": componentes.get("o3"),
        "NO2": componentes.get("no2"),
        "SO2": componentes.get("so2"),
        "CO": co_ugm3_para_ppm(componentes["co"]) if "co" in componentes else None,
    }


def buscar_qualidade_ar(lat: float, lon: float, api_key: str) -> dict:
    """Consulta a API real. Lança requests.HTTPError em caso de falha."""
    resp = requests.get(
        URL_AIR, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=10
    )
    resp.raise_for_status()
    item = resp.json()["list"][0]
    return {
        "concentracoes": _normalizar(item["components"]),
        "horario": datetime.fromtimestamp(item["dt"]).strftime("%d/%m/%Y %H:%M"),
        "fonte": "OpenWeather",
    }


def simular_qualidade_ar(cidade: str, cenario: str = "Aleatório") -> dict:
    """Gera leituras plausíveis. A semente usa a cidade para manter consistência."""
    faixas = {
        "Ar limpo": {"pm2_5": (3, 15), "pm10": (8, 35), "o3": (20, 80),
                     "no2": (5, 60), "so2": (1, 15), "co": (200, 600)},
        "Poluído": {"pm2_5": (55, 110), "pm10": (110, 220), "o3": (120, 190),
                    "no2": (150, 300), "so2": (20, 60), "co": (8000, 14000)},
    }
    rng = random.Random(f"{cidade}-{cenario}-{datetime.now():%Y%m%d%H}")
    if cenario == "Aleatório":
        base = {k: (lo, hi * 1.3) for k, (lo, hi) in faixas["Ar limpo"].items()}
    else:
        base = faixas[cenario]
    comp = {k: rng.uniform(lo, hi) for k, (lo, hi) in base.items()}
    return {
        "concentracoes": _normalizar(comp),
        "horario": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "fonte": f"Simulação ({cenario.lower()})",
    }
