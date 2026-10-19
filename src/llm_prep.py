"""Preparação dos dados coletados para uso futuro com LLMs (próxima entrega, TP3 item 5).

Este módulo **não chama nenhum modelo de linguagem** — ele só organiza o que já
foi coletado (notícias, tabela oficial do IQAr, textos de referência) em um
formato padronizado ("corpus", um documento por linha em JSON), pronto para
ser consumido por técnicas de LLM na próxima etapa do projeto:

| Cenário de LLM (próxima entrega) | Documentos deste corpus usados        |
|---|---|
| Resumo automático de texto        | `tipo == "noticia"` (título + resumo) |
| Classificação de sentimento       | `tipo == "noticia"`                   |
| Perguntas e respostas (Q&A)       | `tipo in ("faixa_oficial", "referencia")` |
| Geração de texto (ex.: boletim diário) | os três tipos combinados, junto com a leitura atual do app |

Cada `Documento` tem um texto curto e autocontido (bom para embeddings/RAG) e
metadados que permitem filtrar por fonte, poluente, categoria etc.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src import data_loader as dl

ARQ_CORPUS = dl.DATA_RAW.parent / "processed" / "corpus_llm.jsonl"


@dataclass
class Documento:
    id: str
    tipo: str  # "noticia" | "faixa_oficial" | "referencia"
    texto: str
    fonte: str
    data: str | None = None
    link: str | None = None
    metadados: dict = field(default_factory=dict)


def _docs_noticias() -> list[Documento]:
    df = dl.ler_csv(dl.ARQ_NOTICIAS)
    if df is None or df.empty:
        return []
    docs = []
    for i, linha in df.iterrows():
        texto = f"{linha['titulo']}. {linha.get('resumo') or ''}".strip()
        docs.append(Documento(
            id=f"noticia-{i}", tipo="noticia", texto=texto,
            fonte=str(linha.get("fonte", "")), data=str(linha.get("data_publicacao") or "") or None,
            link=linha.get("link"), metadados={"termo_busca": linha.get("termo_busca")},
        ))
    return docs


def _docs_faixas_oficiais() -> list[Documento]:
    df = dl.ler_csv(dl.ARQ_IQAR)
    if df is None or df.empty:
        return []
    docs = []
    for i, linha in df.iterrows():
        teto = linha["conc_max"] if pd.notna(linha.get("conc_max")) else "sem limite superior"
        texto = (
            f"Para o poluente {linha['nome']} ({linha['poluente']}), a categoria "
            f"'{linha['categoria']}' do IQAr vai de {linha['indice_min']} a {linha['indice_max']} "
            f"pontos, correspondendo a uma concentração entre {linha['conc_min']} e {teto} "
            f"{linha['unidade']}. Recomendação oficial: {linha['recomendacao']}"
        )
        docs.append(Documento(
            id=f"faixa-{i}", tipo="faixa_oficial", texto=texto, fonte="CETESB / CONAMA 491/2018",
            metadados={"poluente": linha["poluente"], "categoria": linha["categoria"]},
        ))
    return docs


def _docs_referencia(tamanho_bloco: int = 600) -> list[Documento]:
    """Quebra o texto de referência (CETESB/Wikipédia) em blocos menores.

    Blocos de ~`tamanho_bloco` caracteres, cortando em fim de frase — tamanho
    pensado para um futuro índice de embeddings/RAG, não para leitura humana.
    """
    texto = dl.ler_texto(dl.DATA_RAW / "referencia_iqar_texto.txt")
    if not texto.strip():
        return []
    sentencas = re.split(r"(?<=[.!?])\s+", texto)
    blocos: list[str] = []
    atual = ""
    for sentenca in sentencas:
        if atual and len(atual) + len(sentenca) > tamanho_bloco:
            blocos.append(atual.strip())
            atual = ""
        atual += " " + sentenca
    if atual.strip():
        blocos.append(atual.strip())
    return [
        Documento(id=f"referencia-{i}", tipo="referencia", texto=bloco,
                  fonte="CETESB / Wikipédia (texto consolidado)")
        for i, bloco in enumerate(blocos)
    ]


def montar_corpus() -> list[Documento]:
    """Reúne todos os documentos disponíveis a partir dos dados já coletados."""
    return _docs_noticias() + _docs_faixas_oficiais() + _docs_referencia()


def resumo_corpus(docs: list[Documento] | None = None) -> pd.DataFrame:
    """Contagem de documentos por tipo — útil para checar o corpus rapidamente."""
    docs = docs if docs is not None else montar_corpus()
    contagem = pd.Series([d.tipo for d in docs]).value_counts()
    return contagem.rename_axis("tipo").reset_index(name="documentos")


def salvar_corpus(caminho: Path = ARQ_CORPUS) -> int:
    """Grava o corpus em JSONL (um documento por linha) e devolve a quantidade salva."""
    docs = montar_corpus()
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as arquivo:
        for doc in docs:
            arquivo.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
    return len(docs)
