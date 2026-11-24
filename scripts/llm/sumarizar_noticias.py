"""Gera os insights do noticiário com o LLM local (TP5): sentimento e resumos por recorte.

Etapas:
  1. prepara as notícias do scraping (remove páginas de índice, detecta local e mês);
  2. classifica o sentimento de cada manchete (positivo, negativo ou neutro);
  3. resume cada recorte padrão (geral, por local, por mês, por tema) com map-reduce
     e confere a fidelidade de cada boletim contra as manchetes de origem.

Saídas em data/processed/insights/ (versionadas, para o dashboard funcionar mesmo
onde o LLM não está instalado, como no Streamlit Cloud):
    noticias_enriquecidas.csv -> notícias limpas + mes, local e sentimento
    resumos.json              -> um boletim por recorte, com estatísticas, fidelidade e fontes

Uso, na raiz do projeto (com o ambiente do requirements-llm.txt):
    python scripts/llm/sumarizar_noticias.py                    # tudo (~20 min na GTX 1050 Ti)
    python scripts/llm/sumarizar_noticias.py --sem-sentimento   # só os resumos
    python scripts/llm/sumarizar_noticias.py --max-noticias 40  # teste rápido
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

import pandas as pd  # noqa: E402

from src import sumarizacao as sz  # noqa: E402
from src.llm import LLMLocal, rotulo_sentimento  # noqa: E402


def classificar_sentimentos(df: pd.DataFrame, llm: LLMLocal) -> pd.Series:
    rotulos, inicio = [], time.perf_counter()
    for i, titulo in enumerate(df["titulo"], 1):
        rotulos.append(rotulo_sentimento(llm.processar("sentimento", titulo)["resultado"]))
        if i % 25 == 0 or i == len(df):
            print(f"  sentimento: {i}/{len(df)} ({time.perf_counter() - inicio:.0f} s)")
    return pd.Series(rotulos, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sem-sentimento", action="store_true")
    parser.add_argument("--max-noticias", type=int, default=None,
                        help="Limita as notícias de cada recorte (as mais recentes).")
    args = parser.parse_args()

    brutas = pd.read_csv(sz.ARQ_NOTICIAS)
    df = sz.preparar_noticias(brutas)
    print(f"Notícias: {len(brutas)} coletadas, {len(df)} válidas "
          f"({len(brutas) - len(df)} removidas: páginas de índice ou duplicadas).")

    llm = LLMLocal()
    print(f"Modelo: {llm.repo} (a primeira chamada carrega o modelo)")
    if args.sem_sentimento:
        anterior = (pd.read_csv(sz.ARQ_ENRIQUECIDAS) if sz.ARQ_ENRIQUECIDAS.exists()
                    else pd.DataFrame(columns=["titulo", "sentimento"]))
        df = df.merge(anterior[["titulo", "sentimento"]], on="titulo", how="left")
    else:
        df["sentimento"] = classificar_sentimentos(df, llm)

    sz.PASTA_INSIGHTS.mkdir(parents=True, exist_ok=True)
    colunas = ["titulo", "fonte", "data", "mes", "local", "termo_busca", "sentimento", "link",
               "coletado_em"]
    df[colunas].to_csv(sz.ARQ_ENRIQUECIDAS, index=False)
    print(f"Gravado: {sz.ARQ_ENRIQUECIDAS.relative_to(RAIZ)}")

    resumos = []
    for recorte in sz.recortes_padrao(df):
        print(f"Resumindo: {recorte['titulo']} ...")
        item = sz.resumir_recorte(df, llm.gerar_texto, recorte["local"], recorte["mes"],
                                  recorte["tema"], args.max_noticias)
        resumos.append({"id": recorte["id"], "titulo": recorte["titulo"], **item})
        p, f = item["processamento"], item["fidelidade"]
        print(f"  {item['estatisticas']['n_noticias']} notícias, {p['n_chamadas']} chamadas, "
              f"{p['tempo_s']} s | formato ok: {item['boletim']['formato_ok']} | "
              f"fidelidade: {f['taxa']:.0%} {f['nao_encontrados'] or ''}")
        print(f"  {item['boletim']['resumo'][:150]}")

    saida = {"gerado_em": datetime.now().isoformat(timespec="seconds"),
             "modelo": llm.repo, "formato": llm.formato,
             "noticias_coletadas": len(brutas), "noticias_validas": len(df),
             "coleta": str(brutas["coletado_em"].max()), "recortes": resumos}
    sz.ARQ_RESUMOS.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gravado: {sz.ARQ_RESUMOS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
