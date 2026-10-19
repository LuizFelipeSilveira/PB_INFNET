"""Coleta manchetes sobre qualidade do ar no Google Notícias (feed RSS) com BeautifulSoup.

Execute a partir da raiz do projeto:
    python scripts/scraping/scrape_noticias.py

Saídas em data/raw/:
    noticias_qualidade_ar.csv  -> título, fonte, data, link, termo de busca
    noticias_texto.txt         -> títulos + resumos (usado na nuvem de palavras)
"""
from __future__ import annotations

from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[2]
DATA_RAW = RAIZ / "data" / "raw"

TERMOS = ["qualidade do ar", "poluição do ar", "queimadas fumaça", "índice de qualidade do ar"]
URL_RSS = "https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
HEADERS = {"User-Agent": "Mozilla/5.0 (projeto academico Eter)"}


def parse_feed(xml: str, termo: str) -> list[dict]:
    soup = BeautifulSoup(xml, "xml")
    noticias = []
    for item in soup.find_all("item"):
        titulo = item.title.get_text(strip=True) if item.title else ""
        fonte = item.source.get_text(strip=True) if item.source else ""
        if fonte and titulo.endswith(f" - {fonte}"):
            titulo = titulo[: -len(fonte) - 3]
        data = None
        if item.pubDate:
            try:
                data = parsedate_to_datetime(item.pubDate.get_text()).strftime("%Y-%m-%d %H:%M")
            except (TypeError, ValueError):
                pass
        resumo = ""
        if item.description:
            resumo = BeautifulSoup(item.description.get_text(), "html.parser").get_text(" ", strip=True)
        noticias.append({
            "titulo": titulo, "fonte": fonte, "data_publicacao": data,
            "link": item.link.get_text(strip=True) if item.link else "",
            "resumo": resumo, "termo_busca": termo,
        })
    return noticias


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    todas = []
    for termo in TERMOS:
        print(f"Buscando: {termo}")
        try:
            resp = requests.get(URL_RSS.format(q=quote_plus(termo)), headers=HEADERS, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as erro:
            print(f"  ! Falha: {erro}")
            continue
        itens = parse_feed(resp.text, termo)
        print(f"  {len(itens)} notícias")
        todas.extend(itens)

    if not todas:
        print("Nenhuma notícia coletada. Verifique a conexão e tente novamente.")
        return

    df = pd.DataFrame(todas).drop_duplicates(subset="titulo")
    df["coletado_em"] = datetime.now().isoformat(timespec="seconds")
    df.to_csv(DATA_RAW / "noticias_qualidade_ar.csv", index=False)
    texto = "\n".join((df["titulo"] + ". " + df["resumo"].fillna("")).tolist())
    (DATA_RAW / "noticias_texto.txt").write_text(texto, encoding="utf-8")
    print(f"Concluído: {len(df)} notícias únicas salvas em data/raw/")


if __name__ == "__main__":
    main()
