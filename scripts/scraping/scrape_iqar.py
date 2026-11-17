"""Extrai a tabela oficial do IQAr (PDF do MMA) e textos de referência (CETESB, Wikipédia).

Execute a partir da raiz do projeto:
    python scripts/scraping/scrape_iqar.py
    python scripts/scraping/scrape_iqar.py --sem-selenium   # força requests em tudo

Saídas em data/raw/:
    tabelas_web/<fonte>_tabela_<n>.csv   -> tabelas HTML brutas encontradas
    referencia_iqar_texto.txt            -> texto das páginas (usado na nuvem de palavras)
    tabela_iqar.csv                      -> faixas oficiais do IQAr, usadas no cálculo do app
    log_coleta_iqar.csv                  -> status, nº de tabelas e motor usado por fonte
e em data/external/:
    mma_orientacao_tecnica_iqar_jan2025.pdf -> cópia do PDF oficial (reserva offline)

De onde vem a tabela oficial (TP4)
-----------------------------------
As faixas do IQAr vêm do Quadro 1 da "Orientação técnica: índice de qualidade do ar"
do MMA (jan/2025, Resolução CONAMA 506/2024), publicada em PDF:
  1. BeautifulSoup lê a página do IQAr no site do MMA e encontra o link do PDF;
     se não achar, usa o endereço conhecido do PDF;
  2. o PDF é baixado e o Quadro 1 é extraído com pdfplumber (src/tabela_oficial.py);
  3. a tabela é validada (6 poluentes x 5 faixas contínuas) antes de ser gravada.
Se o download falhar, usa a cópia em data/external/. Se nada der certo, grava a
reserva que fica no código (src/iqar.py). O log registra qual caminho foi usado e o
CSV traz a coluna `fonte`.

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
from src.iqar import (  # noqa: E402
    CATEGORIAS, FAIXAS_INDICE, FONTE_REFERENCIA, LIMITES_REFERENCIA, NOMES, RECOMENDACOES,
    UNIDADES,
)
from src.tabela_oficial import TabelaInvalida, extrair_quadro1, validar  # noqa: E402

DATA_RAW = RAIZ / "data" / "raw"
PASTA_TABELAS = DATA_RAW / "tabelas_web"
PDF_LOCAL = RAIZ / "data" / "external" / "mma_orientacao_tecnica_iqar_jan2025.pdf"

PAGINA_MMA = ("https://www.gov.br/mma/pt-br/assuntos/meio-ambiente-urbano-recursos-hidricos-"
              "qualidade-ambiental/qualidade-do-ar/indice-de-qualidade-do-ar-iqar")
URL_PDF_MMA = PAGINA_MMA + "/orientacao-tecnica-indice-de-qualidade-do-ar-jan-25.pdf"
FONTE_MMA = "MMA, Orientação técnica IQAr (jan/2025), Quadro 1 - CONAMA 506/2024"

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


def link_do_pdf_mma() -> str:
    """Procura, na página do IQAr do MMA, o link da orientação técnica em PDF."""
    soup = baixar(PAGINA_MMA)
    if soup is not None:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "orientacao-tecnica" in href and "qualidade-do-ar" in href:
                return href if href.startswith("http") else requests.compat.urljoin(PAGINA_MMA, href)
        print("  ! Link do PDF não encontrado na página; usando o endereço conhecido.")
    return URL_PDF_MMA


def obter_tabela_oficial() -> tuple[dict, str, str]:
    """Devolve (limites, fonte, motor). Tenta o PDF online, depois a cópia local."""
    url = link_do_pdf_mma()
    print(f"Coletando mma (requests + pdfplumber): {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        limites, indices = extrair_quadro1(resp.content)
        validar(limites, indices)
        PDF_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        PDF_LOCAL.write_bytes(resp.content)  # atualiza a cópia de reserva
        return limites, FONTE_MMA, "requests+pdfplumber"
    except (requests.RequestException, TabelaInvalida) as erro:
        print(f"  ! PDF online indisponível ({erro}). Tentando a cópia local.")
    try:
        limites, indices = extrair_quadro1(PDF_LOCAL)
        validar(limites, indices)
        return limites, f"{FONTE_MMA} (cópia local)", "pdf-local+pdfplumber"
    except (OSError, TabelaInvalida) as erro:
        print(f"  ! Cópia local indisponível ({erro}). Usando a reserva do código.")
    return LIMITES_REFERENCIA, FONTE_REFERENCIA, "reserva-no-codigo"


def tabela_consolidada(limites: dict, fonte: str) -> pd.DataFrame:
    """Tabela de referência do IQAr no formato longo usado pelo app e pela API."""
    linhas = []
    for pol, faixas in limites.items():
        for (ci, cf), (ii, if_), cat in zip(faixas, FAIXAS_INDICE, CATEGORIAS):
            linhas.append({
                "poluente": pol, "nome": NOMES[pol], "unidade": UNIDADES[pol],
                "categoria": cat, "indice_min": ii, "indice_max": if_,
                "conc_min": ci, "conc_max": cf,
                "recomendacao": RECOMENDACOES[cat], "fonte": fonte,
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

    limites, fonte, motor = obter_tabela_oficial()
    tabela_consolidada(limites, fonte).to_csv(DATA_RAW / "tabela_iqar.csv", index=False)
    log.append({"fonte": "mma", "url": URL_PDF_MMA,
                "status": "ok" if motor != "reserva-no-codigo" else "erro",
                "tabelas": int(motor != "reserva-no-codigo"), "motor": motor})
    print(f"  tabela oficial: {fonte}")

    log_df = pd.DataFrame(log)
    log_df["coletado_em"] = datetime.now().isoformat(timespec="seconds")
    log_df.to_csv(DATA_RAW / "log_coleta_iqar.csv", index=False)
    print("Concluído. Arquivos em data/raw/")


if __name__ == "__main__":
    main()
