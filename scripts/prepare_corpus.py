"""Consolida os dados já coletados em um corpus pronto para a próxima etapa (LLMs).

Rode depois dos scrapers, a partir da raiz do projeto:
    python scripts/scraping/scrape_noticias.py
    python scripts/scraping/scrape_iqar.py
    python scripts/prepare_corpus.py

Gera data/processed/corpus_llm.jsonl. Veja src/llm_prep.py para o que cada
tipo de documento serve nos cenários de LLM da próxima entrega.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ))

from src.llm_prep import ARQ_CORPUS, resumo_corpus, salvar_corpus  # noqa: E402


def main() -> None:
    total = salvar_corpus()
    print(f"Corpus salvo em {ARQ_CORPUS.relative_to(RAIZ)} ({total} documentos).")
    if total:
        print(resumo_corpus().to_string(index=False))
    else:
        print("Nenhum documento gerado — rode os scrapers antes.")


if __name__ == "__main__":
    main()
