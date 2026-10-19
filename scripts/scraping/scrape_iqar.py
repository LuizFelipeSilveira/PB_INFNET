"""Extrai tabelas e texto de páginas de referência sobre o IQAr.

Execute a partir da raiz do projeto:
    python scripts/scraping/scrape_iqar.py
    python scripts/scraping/scrape_iqar.py --sem-selenium   # força requests em tudo

Saídas em data/raw/:
    tabelas_web/<fonte>_tabela_<n>.csv   -> tabelas HTML brutas encontradas
    referencia_iqar_texto.txt            -> texto das páginas (usado na nuvem de palavras)
    tabela_iqar.csv                      -> tabela consolidada usada pelo app
    log_coleta_iqar.csv                  -> status, nº de tabelas e motor usado por fonte

Sobre o uso de Selenium (TP3, item 3)
--------------------------------------
A primeira versão deste scraper usava só `requests` + BeautifulSoup para as
duas fontes. O log gerado (`log_coleta_iqar.csv`) mostrou que a fonte CETESB
retornava sempre **0 tabelas**: o HTML bruto devolvido pela primeira resposta
não contém as faixas do IQAr em um `<table>` (o conteúdo é montado/carregado
no navegador). Em vez de ligar o Selenium para tudo "por via das dúvidas"
(que vai contra a recomendação do enunciado de manter a simplicidade), ele é
usado só para essa fonte específica, listada em `FONTES_DINAMICAS` — a
Wikipédia continua sendo lida do jeito simples, com `requests`, porque já
funciona. Se o Chrome/driver não estiver disponível no ambiente, o script cai
de volta para `requests` automaticamente (ver `baixar_com_selenium`).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))
from src.iqar import CATEGORIAS, FAIXAS_INDICE, LIMITES, NOMES, RECOMENDACOES, UNIDADES  # noqa: E402

DATA_RAW = RAIZ / "data" / "raw"
PASTA_TABELAS = DATA_RAW / "tabelas_web"

FONTES = {
    "cetesb": "https://cetesb.sp.gov.br/ar/padroes-de-qualidade-do-ar/",
    "wikipedia": "https://pt.wikipedia.org/wiki/%C3%8Dndice_de_qualidade_do_ar",
}
# Fontes cujo conteúdo relevante não aparece no HTML bruto (ver nota acima).
# Ajuste esta lista se outra fonte passar a precisar de renderização via JS.
FONTES_DINAMICAS = {"cetesb"}
HEADERS = {"User-Agent": "Mozilla/5.0 (projeto academico Eter; contato: seu-email)"}


def baixar(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as erro:
        print(f"  ! Falha ao acessar {url}: {erro}")
        return None


def baixar_com_selenium(url: str, espera_segundos: int = 10) -> BeautifulSoup | None:
    """Renderiza a página em um Chrome headless e devolve o HTML já processado pelo JS.

    Usado só para fontes em `FONTES_DINAMICAS`. Se Selenium/Chrome não
    estiverem disponíveis, avisa e cai de volta para `baixar()` (requests).
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        print("  ! selenium/webdriver-manager não instalados "
              "(pip install -r requirements.txt). Usando requests.")
        return baixar(url)

    opcoes = Options()
    opcoes.add_argument("--headless=new")
    opcoes.add_argument("--no-sandbox")
    opcoes.add_argument("--disable-dev-shm-usage")
    opcoes.add_argument(f"user-agent={HEADERS['User-Agent']}")

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opcoes)
    except Exception as erro:  # Chrome ausente, sem internet p/ baixar o driver, etc.
        print(f"  ! Não foi possível iniciar o Chrome headless ({erro}). Usando requests.")
        return baixar(url)

    try:
        driver.get(url)
        try:
            WebDriverWait(driver, espera_segundos).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
        except Exception:
            pass  # segue mesmo sem <table> aparecer: a página pode não ter uma de fato
        html = driver.page_source
    finally:
        driver.quit()
    return BeautifulSoup(html, "html.parser")


def extrair_tabelas(soup: BeautifulSoup) -> list[pd.DataFrame]:
    """Converte cada <table> em DataFrame lendo <th>/<td> com BeautifulSoup."""
    tabelas = []
    for tabela in soup.find_all("table"):
        linhas = []
        for tr in tabela.find_all("tr"):
            celulas = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if any(celulas):
                linhas.append(celulas)
        if len(linhas) < 2:
            continue
        largura = max(len(l) for l in linhas)
        linhas = [l + [""] * (largura - len(l)) for l in linhas]
        tabelas.append(pd.DataFrame(linhas[1:], columns=[f"{h or 'col'}_{i}" for i, h in enumerate(linhas[0])]))
    return tabelas


def extrair_texto(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    area = soup.find("main") or soup.find(id="mw-content-text") or soup.body or soup
    return " ".join(p.get_text(" ", strip=True) for p in area.find_all(["p", "li", "td"]))


def tabela_consolidada() -> pd.DataFrame:
    """Tabela de referência (CETESB / CONAMA 491/2018) no formato longo usado pelo app."""
    linhas = []
    for pol, faixas in LIMITES.items():
        for (ci, cf), (ii, if_), cat in zip(faixas, FAIXAS_INDICE, CATEGORIAS):
            linhas.append({
                "poluente": pol, "nome": NOMES[pol], "unidade": UNIDADES[pol],
                "categoria": cat, "indice_min": ii, "indice_max": if_,
                "conc_min": ci, "conc_max": cf if cat != "Péssima" else None,
                "recomendacao": RECOMENDACOES[cat],
            })
    return pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sem-selenium", action="store_true",
                        help="Ignora FONTES_DINAMICAS e usa requests para todas as fontes.")
    args = parser.parse_args()

    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)
    textos, log = [], []

    for nome, url in FONTES.items():
        usa_selenium = nome in FONTES_DINAMICAS and not args.sem_selenium
        motor = "selenium" if usa_selenium else "requests"
        print(f"Coletando {nome} ({motor}): {url}")
        soup = baixar_com_selenium(url) if usa_selenium else baixar(url)
        if soup is None:
            log.append({"fonte": nome, "url": url, "status": "erro", "tabelas": 0, "motor": motor})
            continue
        tabelas = extrair_tabelas(soup)
        for i, df in enumerate(tabelas, start=1):
            df.to_csv(PASTA_TABELAS / f"{nome}_tabela_{i}.csv", index=False)
        textos.append(extrair_texto(soup))
        log.append({"fonte": nome, "url": url, "status": "ok", "tabelas": len(tabelas), "motor": motor})
        print(f"  {len(tabelas)} tabela(s) salvas")

    (DATA_RAW / "referencia_iqar_texto.txt").write_text("\n\n".join(textos), encoding="utf-8")
    tabela_consolidada().to_csv(DATA_RAW / "tabela_iqar.csv", index=False)

    log_df = pd.DataFrame(log)
    log_df["coletado_em"] = datetime.now().isoformat(timespec="seconds")
    log_df.to_csv(DATA_RAW / "log_coleta_iqar.csv", index=False)
    print("Concluído. Arquivos em data/raw/")


if __name__ == "__main__":
    main()
