"""Leitura dos dados extraídos (data/raw) e validação de uploads."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw"
DATA_EXTERNAL = RAIZ / "data" / "external"

ARQ_IQAR = DATA_RAW / "tabela_iqar.csv"
ARQ_NOTICIAS = DATA_RAW / "noticias_qualidade_ar.csv"
ARQ_TEXTO = DATA_RAW / "noticias_texto.txt"

COLUNAS_UPLOAD = ["data", "cidade", "poluente", "valor"]
POLUENTES_VALIDOS = {"MP10", "MP2.5", "O3", "CO", "NO2", "SO2"}

STOPWORDS_PT = set("""
a à ao aos as às até com como da das de dela dele deles depois do dos e é ela elas
ele eles em entre era essa esse esta está estão este eu foi foram há isso isto já
lhe mais mas me mesmo meu minha muito na nas nem no nos nós o os ou para pela pelas
pelo pelos por qual quando que quem se sem ser seu seus sua suas também te tem têm
um uma umas uns vai vão você sobre após diz dizem pode podem ano anos dia dias
será sendo sido ainda onde contra toda todo todos todas outro outra outros cada
g1 uol cnn folha estadão globo notícias brasil agência r7 metrópoles terra
""".split())


def ler_csv(caminho: Path) -> pd.DataFrame | None:
    return pd.read_csv(caminho) if caminho.exists() else None


def ler_texto(caminho: Path) -> str:
    return caminho.read_text(encoding="utf-8") if caminho.exists() else ""


def frequencia_palavras(texto: str, top: int = 20) -> pd.DataFrame:
    palavras = re.findall(r"[a-zà-ú]{3,}", texto.lower())
    contagem = Counter(p for p in palavras if p not in STOPWORDS_PT)
    return pd.DataFrame(contagem.most_common(top), columns=["Palavra", "Ocorrências"])


def validar_upload(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Valida o CSV enviado pelo usuário. Retorna (df_limpo, lista_de_erros)."""
    erros = []
    df.columns = [c.strip().lower() for c in df.columns]
    faltando = [c for c in COLUNAS_UPLOAD if c not in df.columns]
    if faltando:
        return None, [f"Colunas obrigatórias ausentes: {', '.join(faltando)}"]

    df = df[COLUNAS_UPLOAD].copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce", dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["poluente"] = df["poluente"].astype(str).str.strip().str.upper().str.replace("PM", "MP")
    df["cidade"] = df["cidade"].astype(str).str.strip()

    invalidos = ~df["poluente"].isin(POLUENTES_VALIDOS)
    if invalidos.any():
        erros.append(f"{invalidos.sum()} linha(s) com poluente fora de "
                     f"{sorted(POLUENTES_VALIDOS)} foram descartadas.")
    nulos = df["data"].isna() | df["valor"].isna()
    if nulos.any():
        erros.append(f"{nulos.sum()} linha(s) com data ou valor inválido foram descartadas.")

    df = df[~invalidos & ~nulos]
    return (df if not df.empty else None), erros


def modelo_upload() -> pd.DataFrame:
    return pd.DataFrame({
        "data": ["21/09/2026", "21/09/2026", "22/09/2026"],
        "cidade": ["Rio de Janeiro - RJ", "Rio de Janeiro - RJ", "São Paulo - SP"],
        "poluente": ["MP2.5", "O3", "MP10"],
        "valor": [18.4, 95.0, 62.1],
    })
