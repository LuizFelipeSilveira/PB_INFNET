"""Sumarização automática do noticiário sobre qualidade do ar (TP5, Tarefa 2).

Fluxo, do dado ao insight:

1. `preparar_noticias`: limpa o CSV do scraping. Remove as páginas de índice do
   IQAir, que não são notícias, e acrescenta o mês e o local citado.
2. `filtrar`: escolhe o recorte (local, mês, tema) e as notícias mais recentes.
3. `sumarizar`: resume com o LLM local numa cadeia de prompts curtos, uma tarefa
   por chamada (lição das rodadas 1 a 3 de avaliação):
     * fatos: cada lote de até 20 manchetes vira 3 fatos;
     * consolidação: os fatos são reduzidos em árvore até restarem os 3 principais;
     * resumo: os fatos viram duas frases.
4. `categoria_noticiada`: a recomendação de saúde é a orientação oficial (src/iqar.py)
   da categoria mais grave citada nas manchetes. É escolhida por código, não pelo LLM.
5. `verificar_fidelidade` e `recomendacao_coerente`: conferem se números e nomes do
   texto gerado aparecem nas manchetes e se a orientação combina com o sentimento.
6. `estatisticas`: números calculados por código (quantidade, período, veículos,
   locais, sentimento).

O LLM só escreve texto; tudo o que é contagem ou orientação de saúde vem do código.
Assim, cada número do dashboard é exato, e o texto gerado pode ser conferido.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.iqar import RECOMENDACOES
from src.llm import sem_acento

TAMANHO_LOTE = 20          # manchetes por chamada na etapa map
MAX_FATOS_REDUCE = 12      # acima disso, os fatos são consolidados antes do boletim
MAX_TOKENS_FATOS = 160
MAX_TOKENS_RESUMO = 120

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

# Páginas fixas de índice do IQAir que o feed traz como se fossem notícias:
# "Qualidade do ar em Campinas" e "Índice de Qualidade do Ar (AQI) ... | IQAir Brasil".
_PAGINA_INDICE = re.compile(r"^Qualidade do ar em [^:|]+$|\|\s*IQAir", re.IGNORECASE)


# ---------------------------------------------------------------- prompts
# Rodadas 1 a 3 ensinaram que o modelo de 0,8B acerta quando cada chamada tem UMA tarefa
# simples. Pedir resumo, lista e escolha de orientação numa resposta só quebrou o formato
# nos recortes grandes. Por isso a sumarização é uma cadeia de prompts curtos:
#   fatos (por lote) -> consolidação (até restarem 3) -> resumo (duas frases).
SISTEMA_SUMARIZACAO = (
    "Você é o analista do Éter, um app brasileiro de qualidade do ar. Você resume notícias "
    "para a população, em português do Brasil, de forma clara e fiel ao que foi noticiado."
)

_REGRAS = """Regras:
- use somente as informações acima;
- escreva cada fato em uma linha, começando com "- ";
- escreva locais e números exatamente como estão acima (não troque um estado por uma cidade);
- ao citar um número, repita a palavra que vem junto dele;
- não cite a numeração das manchetes;
- não repita fatos parecidos."""

PROMPT_FATOS = """Abaixo estão manchetes de notícias sobre qualidade do ar{recorte}, numeradas.

{manchetes}

Liste os 3 fatos mais importantes relatados nessas manchetes.
""" + _REGRAS

PROMPT_CONSOLIDAR = """Abaixo estão fatos extraídos de notícias sobre qualidade do ar{recorte}.

{fatos}

Junte esses fatos nos 3 mais importantes.
""" + _REGRAS

PROMPT_RESUMO = """Abaixo estão fatos extraídos de notícias sobre qualidade do ar{recorte}.

{fatos}

Escreva um resumo de duas frases sobre o que aconteceu, para a população.
Use somente os fatos acima. Escreva locais e números exatamente como estão acima.
Responda só com as duas frases, sem título e sem lista."""

# A recomendação de saúde não é escolhida pelo modelo (rodadas 2 e 3: ele escolheu "Boa" e
# "Moderada" para recortes com mais de 90% de notícias negativas). Ela vem da norma
# (src/iqar.py), para a categoria oficial mais grave citada nas próprias manchetes.
TERMOS_CATEGORIA = [  # do mais grave para o mais brando; "insalubre" e "perigoso" vêm do AQI
    ("Péssima", r"pessim|perigos"),
    ("Muito Ruim", r"muito ruim|muito insalubre"),
    ("Ruim", r"\bruim\b|insalubre|critic"),
    ("Moderada", r"moderad"),
    ("Boa", r"qualidade do ar (?:e |esta |fica )?boa\b"),
]


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
        df["titulo"].fillna("").str.strip().str.contains(_PAGINA_INDICE)
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


# ---------------------------------------------------------------- recomendação (código)
def categoria_da_manchete(titulo: str) -> str | None:
    """Categoria oficial mais grave que a manchete menciona (ou None)."""
    alvo = sem_acento(titulo)
    for categoria, padrao in TERMOS_CATEGORIA:
        if re.search(padrao, alvo):
            return categoria
    return None


def categoria_noticiada(manchetes: list[str]) -> tuple[str | None, int]:
    """Categoria oficial citada nas manchetes e quantas manchetes a citam.

    Cada manchete conta para a categoria mais grave que menciona. Vale a mais grave
    citada por pelo menos 10% das manchetes (mínimo 2); se nenhuma chegar lá, a mais
    citada (empate: a mais grave). Sem menção a categorias, devolve (None, 0).
    """
    contagem = {}
    for titulo in manchetes:
        categoria = categoria_da_manchete(titulo)
        if categoria:
            contagem[categoria] = contagem.get(categoria, 0) + 1
    if not contagem:
        return None, 0
    ordem = [c for c, _ in TERMOS_CATEGORIA]
    limiar = max(2, -(-len(manchetes) // 10))
    for categoria in ordem:
        if contagem.get(categoria, 0) >= limiar:
            return categoria, contagem[categoria]
    categoria = max(contagem, key=lambda c: (contagem[c], -ordem.index(c)))
    return categoria, contagem[categoria]


# ---------------------------------------------------------------- sumarização (LLM)
@dataclass
class Boletim:
    resumo: str
    pontos: list[str]
    formato_ok: bool  # resumo em texto corrido, ao menos 2 pontos e fatos vindos do modelo
    texto_bruto: str


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
            for linha in texto.replace("**", "").splitlines()
            if re.match(r"^\s*(?:[-*•]|\d+[.)])\s+\S", linha)]


def _sem_repetir(fatos: list[str]) -> list[str]:
    """Remove fatos repetidos e "fatos" que só falam da numeração das manchetes."""
    vistos, unicos = set(), []
    for fato in fatos:
        chave = sem_acento(re.sub(r"\W+", " ", fato)).strip()
        if not chave or chave in vistos or re.search(r"\bmanchetes?\b", chave):
            continue
        vistos.add(chave)
        unicos.append(fato)
    return unicos


def limpar_resumo(texto: str) -> tuple[str, bool]:
    """Tira rótulos e marcadores do resumo. Devolve (texto, veio em texto corrido?)."""
    linhas = [l for l in texto.replace("**", "").splitlines() if l.strip()]
    corrido = bool(linhas) and not any(re.match(r"^\s*(?:[-*•]|\d+[.)])\s", l) for l in linhas)
    limpo = " ".join(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", l).strip() for l in linhas)
    limpo = re.sub(r"^\s*Resumo\s*:\s*", "", limpo, flags=re.I)
    return " ".join(limpo.split()), corrido


def sumarizar(manchetes: list[str], gerar, recorte: str = "",
              tamanho_lote: int = TAMANHO_LOTE) -> ResultadoSumarizacao:
    """Resume uma lista de manchetes com uma cadeia de prompts (map-reduce em árvore).

    1. fatos: cada lote de até 20 manchetes vira 3 fatos;
    2. consolidação: grupos de até 12 fatos viram 3, até restarem no máximo 12; se
       restarem mais de 3, uma última consolidação escolhe os 3 pontos principais;
    3. resumo: os fatos finais viram duas frases.

    `gerar(prompt, max_tokens) -> dict` é a função de geração (normalmente
    `LLMLocal.gerar_texto`); recebê-la como parâmetro permite testar sem o modelo.
    """
    if not manchetes:
        raise ValueError("Nenhuma manchete para resumir.")
    inicio = time.perf_counter()
    tokens = chamadas = 0
    brutos = []

    def chamar(prompt: str, max_tokens: int) -> str:
        nonlocal tokens, chamadas
        saida = gerar(prompt, max_tokens)
        tokens, chamadas = tokens + saida["tokens_gerados"], chamadas + 1
        brutos.append(saida["resultado"])
        return saida["resultado"]

    def consolidar(grupo: list[str]) -> list[str]:
        texto = chamar(PROMPT_CONSOLIDAR.format(
            recorte=recorte, fatos="\n".join(f"- {f}" for f in grupo)), MAX_TOKENS_FATOS)
        return _linhas_com_marcador(texto)[:3] or grupo[:3]

    lotes = [manchetes[i:i + tamanho_lote] for i in range(0, len(manchetes), tamanho_lote)]
    fatos = []
    for lote in lotes:  # 1. fatos (map)
        texto = chamar(PROMPT_FATOS.format(recorte=recorte, manchetes=_numerar(lote)),
                       MAX_TOKENS_FATOS)
        fatos += _linhas_com_marcador(texto)[:3]
    fatos = _sem_repetir(fatos)
    reserva = not fatos  # o modelo não devolveu nenhum fato em lista
    fatos = fatos or manchetes[:3]
    intermediarios = list(fatos)
    while len(fatos) > MAX_FATOS_REDUCE:  # 2. consolidação em árvore
        grupos = [fatos[i:i + MAX_FATOS_REDUCE] for i in range(0, len(fatos), MAX_FATOS_REDUCE)]
        fatos = _sem_repetir([f for g in grupos for f in consolidar(g)])
    pontos = fatos if len(fatos) <= 3 else (_sem_repetir(consolidar(fatos)) or fatos[:3])
    resumo, corrido = limpar_resumo(chamar(  # 3. resumo
        PROMPT_RESUMO.format(recorte=recorte, fatos="\n".join(f"- {f}" for f in fatos)),
        MAX_TOKENS_RESUMO))
    boletim = Boletim(resumo=resumo, pontos=pontos[:3],
                      formato_ok=bool(resumo) and corrido and len(pontos) >= 2 and not reserva,
                      texto_bruto="\n---\n".join(brutos))
    return ResultadoSumarizacao(boletim=boletim, n_lotes=len(lotes), n_chamadas=chamadas,
                                tokens_gerados=tokens,
                                tempo_s=round(time.perf_counter() - inicio, 1),
                                fatos_intermediarios=intermediarios)


# ---------------------------------------------------------------- fidelidade
_PALAVRAS_DE_FORMATO = {"resumo", "pontos", "principais", "recomendacao", "fato", "evite",
                        "use", "mantenha", "fique", "procure", "feche", "beba", "a", "o", "as",
                        "os", "em", "no", "na", "de", "da", "do"}


def recomendacao_coerente(categoria: str | None, sentimento: dict | None) -> bool | None:
    """Confere a orientação contra um dado calculado de outra forma: o sentimento.

    Orientação branda (Boa/Moderada) com 60% ou mais de notícias negativas, ou grave
    (Muito Ruim/Péssima) com menos de 30%, é marcada como incoerente. Foi esta checagem
    que mostrou, nas rodadas 2 e 3, que o LLM escolhia mal a orientação.
    """
    if not categoria or not sentimento:
        return None
    classificadas = sum(v for k, v in sentimento.items() if k != "indefinido")
    if not classificadas:
        return None
    negativas = sentimento.get("negativo", 0) / classificadas
    if categoria in ("Boa", "Moderada"):
        return negativas < 0.6
    if categoria in ("Muito Ruim", "Péssima"):
        return negativas >= 0.3
    return True


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
    categoria, evidencia = categoria_noticiada(manchetes)
    est = estatisticas(selecao)
    return {
        "filtros": {"local": local, "mes": mes, "tema": tema, "max_noticias": max_noticias},
        "descricao": descrever_recorte(local, mes, tema).strip() or "todas as notícias",
        "estatisticas": est,
        "boletim": {"resumo": b.resumo, "pontos": b.pontos,
                    "recomendacao": RECOMENDACOES[categoria] if categoria else None,
                    "categoria_recomendacao": categoria,
                    "manchetes_que_citam_categoria": evidencia,
                    "recomendacao_coerente": recomendacao_coerente(categoria,
                                                                   est.get("sentimento")),
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
