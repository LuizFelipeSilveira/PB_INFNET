"""Sumarização automática do noticiário sobre qualidade do ar (TP5, Tarefa 2).

Fluxo, do dado ao insight:

1. `preparar_noticias`: limpa o CSV do scraping. Remove as páginas de índice do
   IQAir ("Qualidade do ar em Campinas"), que não são notícias, e acrescenta o mês
   e o local citado.
2. `filtrar`: escolhe o recorte (local, mês, tema) e as notícias mais recentes.
3. `sumarizar`: resume com o LLM local em duas etapas (*map-reduce*):
     * map: cada lote de até 20 manchetes vira uma lista de fatos;
     * reduce: os fatos de todos os lotes viram um boletim com Resumo, Pontos
       principais e Recomendação.
   Recortes pequenos (um lote só) vão direto para o boletim.
4. `verificar_fidelidade`: confere se os números e nomes próprios do boletim
   aparecem nas manchetes de origem. É uma checagem automática e barata de
   "alucinação": não prova que o resumo está certo, mas aponta o que conferir.
5. `estatisticas`: números calculados por código (não pelo LLM): quantidade de
   notícias, período, veículos, locais e, se houver, a distribuição de sentimento.

O LLM só escreve texto; tudo o que é contagem vem do pandas. Assim, cada número
do dashboard é exato, e o texto gerado pode ser conferido com as fontes.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.iqar import CATEGORIAS, RECOMENDACOES
from src.llm import sem_acento

TAMANHO_LOTE = 20          # manchetes por chamada na etapa map
MAX_TOKENS_FATOS = 160
MAX_TOKENS_BOLETIM = 230

# Capitais (e algumas cidades frequentes no noticiário), na ordem de busca.
LOCAIS = [
    "Manaus", "Porto Velho", "Rio Branco", "Belém", "Santarém", "Macapá", "Boa Vista",
    "Palmas", "Cuiabá", "Campo Grande", "Goiânia", "Brasília", "São Paulo", "Rio de Janeiro",
    "Belo Horizonte", "Vitória", "Curitiba", "Florianópolis", "Porto Alegre", "Salvador",
    "Recife", "Fortaleza", "São Luís", "Teresina", "Natal", "João Pessoa", "Maceió", "Aracaju",
]
OUTROS_LOCAIS = "Outros locais"
# Estados (e regiões) também são nomes próprios que o modelo pode trocar entre si.
ESTADOS = ["Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará", "Distrito Federal",
           "Espírito Santo", "Goiás", "Maranhão", "Mato Grosso", "Minas Gerais", "Pará",
           "Paraíba", "Paraná", "Pernambuco", "Piauí", "Rondônia", "Roraima", "Tocantins",
           "Sergipe", "Santa Catarina", "Amazônia", "Pantanal", "Cerrado"]

# Páginas fixas de índice do IQAir que o feed traz como se fossem notícias.
_PAGINA_INDICE = re.compile(r"^Qualidade do ar em [^:|]+$", re.IGNORECASE)


# ---------------------------------------------------------------- prompts
SISTEMA_SUMARIZACAO = (
    "Você é o analista do Éter, um app brasileiro de qualidade do ar. Você resume notícias "
    "para a população, em português do Brasil, de forma clara e fiel ao que foi noticiado."
)

PROMPT_FATOS = """Abaixo estão manchetes de notícias sobre qualidade do ar{recorte}, numeradas.

{manchetes}

Liste os 3 fatos mais importantes relatados nessas manchetes.
Regras:
- use somente informações das manchetes;
- escreva cada fato em uma linha, começando com "- ";
- cite o local e os números quando as manchetes trouxerem;
- escreva o local exatamente como na manchete (não troque um estado por uma cidade);
- ao citar um número, repita a palavra que vem junto dele na manchete;
- não repita fatos parecidos."""

PROMPT_BOLETIM = """Abaixo estão {tipo_entrada} sobre qualidade do ar{recorte}.

{conteudo}

Orientações oficiais de saúde:
{orientacoes}

Escreva um boletim exatamente neste formato:
Resumo: <duas frases sobre o que aconteceu>
Pontos principais:
- <fato 1>
- <fato 2>
- <fato 3>
Recomendação: <só o número, de 1 a 5, da orientação oficial que combina com a situação relatada>

Use somente as informações acima. Não invente números, datas nem locais. Escreva os locais e os
números exatamente como aparecem acima."""

# A recomendação de saúde vem da norma (src/iqar.py), não é escrita pelo modelo: na
# rodada 1 ele sugeriu "evite sair da cidade", um conselho inadequado. O modelo só
# escolhe qual das 5 orientações oficiais combina com o noticiário (escolha fechada,
# como na classificação de sentimento).
ORIENTACOES = "\n".join(f"{i}. {cat}: {RECOMENDACOES[cat]}"
                         for i, cat in enumerate(CATEGORIAS, 1))


# ---------------------------------------------------------------- preparo dos dados
def detectar_local(texto: str) -> str:
    alvo = sem_acento(texto)
    for local in LOCAIS:
        if sem_acento(local) in alvo:
            return local
    return OUTROS_LOCAIS


def preparar_noticias(noticias: pd.DataFrame) -> pd.DataFrame:
    """Limpa e enriquece o CSV de notícias. Não altera o DataFrame recebido."""
    df = noticias.copy()
    # Só as do IQAir: veículos de notícia também usam títulos como "Qualidade do ar em
    # Manaus" em matérias de verdade.
    pagina_indice = (df["fonte"].fillna("").str.casefold() == "iqair") & \
        df["titulo"].fillna("").str.strip().str.match(_PAGINA_INDICE)
    df = df[~pagina_indice]
    df = df.drop_duplicates("titulo")
    df["data"] = pd.to_datetime(df["data_publicacao"], errors="coerce")
    df = df.dropna(subset=["data"])
    df["mes"] = df["data"].dt.strftime("%Y-%m")
    df["local"] = df["titulo"].map(detectar_local)
    return df.sort_values("data", ascending=False).reset_index(drop=True)


def filtrar(df: pd.DataFrame, local: str | None = None, mes: str | None = None,
            tema: str | None = None, max_noticias: int | None = None) -> pd.DataFrame:
    """Aplica o recorte e mantém as `max_noticias` mais recentes."""
    if local:
        df = df[df["local"] == local]
    if mes:
        df = df[df["mes"] == mes]
    if tema:
        df = df[df["termo_busca"] == tema]
    return df.head(max_noticias) if max_noticias else df


def descrever_recorte(local: str | None, mes: str | None, tema: str | None) -> str:
    partes = []
    if local and local != OUTROS_LOCAIS:
        partes.append(f"em {local}")
    elif local == OUTROS_LOCAIS:
        partes.append("fora das capitais monitoradas")
    if mes:
        ano, m = mes.split("-")
        partes.append(f"em {m}/{ano}")
    if tema:
        partes.append(f'sobre "{tema}"')
    return (" " + ", ".join(partes)) if partes else ""


def recortes_padrao(df: pd.DataFrame, minimo: int = 10) -> list[dict]:
    """Recortes gerados em lote: geral, por local, por mês e por tema (com ao menos `minimo`)."""
    recortes = [{"id": "geral", "titulo": "Panorama geral", "local": None, "mes": None,
                 "tema": None}]
    for local, n in df["local"].value_counts().items():
        if n >= minimo:
            recortes.append({"id": f"local-{sem_acento(local).replace(' ', '-')}",
                             "titulo": f"Local: {local}", "local": local, "mes": None,
                             "tema": None})
    for mes, n in sorted(df["mes"].value_counts().items()):
        if n >= minimo:
            recortes.append({"id": f"mes-{mes}", "titulo": f"Mês: {mes[5:]}/{mes[:4]}",
                             "local": None, "mes": mes, "tema": None})
    for tema, n in df["termo_busca"].value_counts().items():
        if n >= minimo:
            recortes.append({"id": f"tema-{sem_acento(tema).replace(' ', '-')}",
                             "titulo": f"Tema: {tema}", "local": None, "mes": None,
                             "tema": tema})
    return recortes


# ---------------------------------------------------------------- estatísticas (código, não LLM)
def estatisticas(df: pd.DataFrame) -> dict:
    est = {
        "n_noticias": int(len(df)),
        "periodo_inicio": df["data"].min().strftime("%Y-%m-%d") if len(df) else None,
        "periodo_fim": df["data"].max().strftime("%Y-%m-%d") if len(df) else None,
        "veiculos": {k: int(v) for k, v in df["fonte"].value_counts().head(5).items()},
        "locais": {k: int(v) for k, v in df["local"].value_counts().head(5).items()},
    }
    if "sentimento" in df.columns:
        est["sentimento"] = {k: int(v) for k, v in
                             df["sentimento"].fillna("indefinido").value_counts().items()}
    return est


# ---------------------------------------------------------------- sumarização
@dataclass
class Boletim:
    resumo: str
    pontos: list[str]
    recomendacao: str | None           # texto oficial da orientação escolhida
    categoria_recomendacao: str | None  # categoria do IQAr dessa orientação
    texto_bruto: str
    formato_ok: bool  # o modelo seguiu o formato pedido?


@dataclass
class ResultadoSumarizacao:
    boletim: Boletim
    n_lotes: int
    n_chamadas: int
    tokens_gerados: int
    tempo_s: float
    fatos_intermediarios: list[str] = field(default_factory=list)


def _numerar(manchetes: list[str]) -> str:
    return "\n".join(f"{i}. {t}" for i, t in enumerate(manchetes, 1))


def _linhas_com_marcador(texto: str) -> list[str]:
    return [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linha).strip()
            for linha in texto.splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", linha)]


def escolher_orientacao(resposta: str | None) -> str | None:
    """Categoria escolhida pelo modelo: aceita o número (1–5) ou o nome da categoria."""
    if not resposta:
        return None
    numero = re.search(r"\b([1-5])\b", resposta)
    if numero:
        return CATEGORIAS[int(numero.group(1)) - 1]
    alvo = sem_acento(resposta)
    for cat in sorted(CATEGORIAS, key=len, reverse=True):  # "Muito Ruim" antes de "Ruim"
        if sem_acento(cat) in alvo:
            return cat
    return None


def interpretar_boletim(texto: str) -> Boletim:
    """Separa Resumo / Pontos principais / Recomendação. Tolera pequenas variações."""
    norm = texto.replace("**", "")
    resumo = re.search(r"Resumo\s*:\s*(.+?)(?=\n\s*Pontos|\n\s*Recomenda|\Z)", norm, re.S | re.I)
    pontos = re.search(r"Pontos principais\s*:?\s*\n(.+?)(?=\n\s*Recomenda|\Z)", norm, re.S | re.I)
    recom = re.search(r"Recomenda(?:ção|cao)\s*:\s*(.+)", norm, re.S | re.I)
    categoria = escolher_orientacao(recom.group(1) if recom else None)
    oficial = RECOMENDACOES[categoria] if categoria else None
    lista = _linhas_com_marcador(pontos.group(1)) if pontos else []
    if resumo and lista:
        return Boletim(resumo=" ".join(resumo.group(1).split()), pontos=lista[:5],
                       recomendacao=oficial, categoria_recomendacao=categoria,
                       texto_bruto=texto, formato_ok=categoria is not None)
    # Fora do formato: aproveita o texto inteiro como resumo, mas sinaliza.
    corpo = re.split(r"\n\s*Recomenda", norm, flags=re.I)[0]
    return Boletim(resumo=" ".join(corpo.split()), pontos=_linhas_com_marcador(corpo)[:5],
                   recomendacao=oficial, categoria_recomendacao=categoria,
                   texto_bruto=texto, formato_ok=False)


def sumarizar(manchetes: list[str], gerar, recorte: str = "",
              tamanho_lote: int = TAMANHO_LOTE) -> ResultadoSumarizacao:
    """Resume uma lista de manchetes com map-reduce.

    `gerar(prompt, max_tokens) -> dict` é a função de geração (normalmente
    `LLMLocal.gerar_texto`); recebê-la como parâmetro permite testar sem o modelo.
    """
    if not manchetes:
        raise ValueError("Nenhuma manchete para resumir.")
    inicio = time.perf_counter()
    lotes = [manchetes[i:i + tamanho_lote] for i in range(0, len(manchetes), tamanho_lote)]
    tokens, chamadas, fatos = 0, 0, []

    if len(lotes) == 1:
        prompt = PROMPT_BOLETIM.format(tipo_entrada="manchetes de notícias, numeradas",
                                       recorte=recorte, conteudo=_numerar(lotes[0]),
                                       orientacoes=ORIENTACOES)
    else:
        for lote in lotes:  # etapa map
            saida = gerar(PROMPT_FATOS.format(recorte=recorte, manchetes=_numerar(lote)),
                          MAX_TOKENS_FATOS)
            tokens, chamadas = tokens + saida["tokens_gerados"], chamadas + 1
            fatos += _linhas_com_marcador(saida["resultado"]) or [saida["resultado"].strip()]
        prompt = PROMPT_BOLETIM.format(  # etapa reduce
            tipo_entrada="fatos extraídos de notícias", recorte=recorte,
            conteudo="\n".join(f"- {f}" for f in fatos), orientacoes=ORIENTACOES)
    saida = gerar(prompt, MAX_TOKENS_BOLETIM)
    tokens, chamadas = tokens + saida["tokens_gerados"], chamadas + 1
    return ResultadoSumarizacao(
        boletim=interpretar_boletim(saida["resultado"]), n_lotes=len(lotes),
        n_chamadas=chamadas, tokens_gerados=tokens,
        tempo_s=round(time.perf_counter() - inicio, 1), fatos_intermediarios=fatos)


# ---------------------------------------------------------------- fidelidade
_PALAVRAS_DE_FORMATO = {"resumo", "pontos", "principais", "recomendacao", "fato", "evite",
                        "use", "mantenha", "fique", "procure", "feche", "beba", "a", "o", "as",
                        "os", "em", "no", "na", "de", "da", "do"}


def verificar_fidelidade(texto: str, fontes: list[str]) -> dict:
    """Confere números e nomes próprios do texto gerado contra as manchetes de origem.

    Devolve quantos termos foram verificados, quais não aparecem nas fontes e a taxa
    de termos encontrados (1,0 = tudo o que foi checado aparece nas manchetes).
    """
    base = sem_acento(" ".join(fontes))
    numeros_base = set(re.findall(r"\d+", base))
    termos = set()
    for numero in re.findall(r"\d+(?:[.,]\d+)?", texto):
        termos.add(("numero", numero))
    # Palavras com inicial maiúscula que não abrem frase nem linha: prováveis nomes próprios.
    for m in re.finditer(r"(?<![.:!?\n-]\s)(?<!^)\b([A-ZÁÂÃÉÊÍÓÔÕÚÇ][\wÀ-ú]{2,})", texto,
                         flags=re.M):
        palavra = m.group(1)
        if sem_acento(palavra) not in _PALAVRAS_DE_FORMATO:
            termos.add(("nome", palavra))
    # No início de frase, a maiúscula não indica nome próprio; ali só checa locais
    # conhecidos (na rodada 1, "Manaus" no lugar de "Amazonas" passou sem ser visto).
    for local in LOCAIS + ESTADOS:
        if re.search(rf"\b{re.escape(local)}\b", texto):
            termos.add(("nome", local))
    nao_encontrados = []
    for tipo, termo in sorted(termos):
        if tipo == "numero":
            ok = all(parte in numeros_base for parte in re.findall(r"\d+", termo))
        else:
            ok = sem_acento(termo) in base
        if not ok:
            nao_encontrados.append(termo)
    verificados = len(termos)
    return {"termos_verificados": verificados, "nao_encontrados": nao_encontrados,
            "taxa": round(1 - len(nao_encontrados) / verificados, 2) if verificados else 1.0}


# ---------------------------------------------------------------- arquivos
RAIZ = Path(__file__).resolve().parents[1]
ARQ_NOTICIAS = RAIZ / "data" / "raw" / "noticias_qualidade_ar.csv"
PASTA_INSIGHTS = RAIZ / "data" / "processed" / "insights"   # versionada (ver .gitignore)
ARQ_ENRIQUECIDAS = PASTA_INSIGHTS / "noticias_enriquecidas.csv"
ARQ_RESUMOS = PASTA_INSIGHTS / "resumos.json"


def carregar_noticias() -> pd.DataFrame:
    """Notícias prontas para resumir.

    Usa as enriquecidas (com sentimento) se vierem da mesma coleta do CSV bruto; se o
    scraper rodou de novo depois, prepara o CSV bruto (sem sentimento) para não usar
    notícias velhas.
    """
    brutas = pd.read_csv(ARQ_NOTICIAS)
    if ARQ_ENRIQUECIDAS.exists():
        df = pd.read_csv(ARQ_ENRIQUECIDAS)
        if df["coletado_em"].max() == brutas["coletado_em"].max():
            df["data"] = pd.to_datetime(df["data"], errors="coerce")
            return (df.dropna(subset=["data"]).sort_values("data", ascending=False)
                      .reset_index(drop=True))
    return preparar_noticias(brutas)


# ---------------------------------------------------------------- recorte completo


def resumir_recorte(df: pd.DataFrame, gerar, local: str | None = None, mes: str | None = None,
                    tema: str | None = None, max_noticias: int | None = None) -> dict:
    """Filtra, resume, verifica e calcula as estatísticas de um recorte.

    Usado pelo script em lote e pela rota POST /sumarizar_noticias. Levanta
    ValueError se o recorte não tiver notícias.
    """
    selecao = filtrar(df, local, mes, tema, max_noticias)
    if selecao.empty:
        raise ValueError("Nenhuma notícia neste recorte.")
    manchetes = selecao["titulo"].tolist()
    resultado = sumarizar(manchetes, gerar, recorte=descrever_recorte(local, mes, tema))
    b = resultado.boletim
    texto_gerado = " ".join([b.resumo, *b.pontos])
    return {
        "filtros": {"local": local, "mes": mes, "tema": tema, "max_noticias": max_noticias},
        "descricao": descrever_recorte(local, mes, tema).strip() or "todas as notícias",
        "estatisticas": estatisticas(selecao),
        "boletim": {"resumo": b.resumo, "pontos": b.pontos, "recomendacao": b.recomendacao,
                    "categoria_recomendacao": b.categoria_recomendacao,
                    "formato_ok": b.formato_ok},
        # a recomendação é orientação geral de saúde, não fato noticiado: fica fora da checagem
        "fidelidade": verificar_fidelidade(" ".join([b.resumo, *b.pontos]), manchetes),
        "fontes": [{"titulo": r.titulo, "veiculo": r.fonte, "data": r.data.strftime("%Y-%m-%d"),
                    "link": r.link if isinstance(r.link, str) else None,
                    "sentimento": getattr(r, "sentimento", None)
                    if isinstance(getattr(r, "sentimento", None), str) else None}
                   for r in selecao.itertuples()],
        "processamento": {"n_lotes": resultado.n_lotes, "n_chamadas": resultado.n_chamadas,
                          "tokens_gerados": resultado.tokens_gerados,
                          "tempo_s": resultado.tempo_s,
                          "palavras_entrada": sum(len(t.split()) for t in manchetes),
                          "palavras_saida": len(texto_gerado.split())},
        "texto_bruto": b.texto_bruto,
    }
