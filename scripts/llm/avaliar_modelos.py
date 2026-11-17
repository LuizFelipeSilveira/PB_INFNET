"""Avalia LLMs candidatos rodando LOCALMENTE, nas tarefas reais do Éter (TP4).

Para cada modelo, o script mede:
  * custo computacional: tempo de carga, VRAM/RAM de pico e velocidade (tokens/s);
  * desempenho: acerto em classificação de sentimento e em perguntas sobre a
    tabela do IQAr (respostas conferidas automaticamente), além de guardar o
    texto gerado no resumo e no boletim para avaliação manual.

Os resultados alimentam a justificativa em docs/model/ModelSelection.md.

Uso (na raiz do projeto, com o ambiente do requirements-llm.txt instalado):
    python scripts/llm/avaliar_modelos.py                 # os 3 candidatos, na GPU
    python scripts/llm/avaliar_modelos.py --modelos qwen3.5-2b
    python scripts/llm/avaliar_modelos.py --cpu           # força CPU (mais lento)
    python scripts/llm/avaliar_modelos.py --somente-prompts   # só mostra os prompts

Saídas em data/processed/avaliacao_llm/:
    resumo.csv              -> uma linha por modelo com todas as métricas
    <modelo>.json           -> prompts, respostas e métricas de cada tarefa
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.append(str(RAIZ))

import pandas as pd  # noqa: E402

from src import data_loader as dl  # noqa: E402
from src.llm import carregar, gerar, montar_prompt, rotulo_sentimento  # noqa: E402

PASTA_SAIDA = dl.DATA_RAW.parent / "processed" / "avaliacao_llm"

# ---------------------------------------------------------------- candidatos
# "quant": "4bit" = NF4 via bitsandbytes (cabe nos 4 GB da GTX 1050 Ti);
#          "fp16" = pesos em meia precisão, sem quantização.
CANDIDATOS = {
    "qwen3.5-2b": {"repo": "Qwen/Qwen3.5-2B", "quant": "4bit"},
    "qwen3.5-0.8b": {"repo": "Qwen/Qwen3.5-0.8B", "quant": "fp16"},
    "qwen3-1.7b": {"repo": "Qwen/Qwen3-1.7B", "quant": "4bit"},
}

# Frases curtas com rótulo conhecido, escritas para este teste (não vêm do scraping),
# 4 de cada classe. Nenhuma repete os exemplos do prompt (src/llm.py).
# Rodada 3: a frase "Prefeitura instala nova estação de monitoramento" (neutro) saiu
# por ser ambígua (pode ser lida como positiva) e a lista passou de 6 para 12 frases.
SENTIMENTO = [
    ("Chuva forte limpa o ar e índice volta a ficar bom na capital", "positivo"),
    ("Nova frota de ônibus elétricos reduz emissões e melhora o ar no centro", "positivo"),
    ("Fim das queimadas faz a qualidade do ar voltar ao normal em Porto Velho", "positivo"),
    ("Estudo mostra queda da poluição na cidade após ampliação do metrô", "positivo"),
    ("Fumaça das queimadas deixa qualidade do ar péssima e escolas suspendem aulas", "negativo"),
    ("Hospitais registram alta de atendimentos respiratórios por causa da poluição", "negativo"),
    ("Inversão térmica piora o ar e índice atinge nível muito ruim em São Paulo", "negativo"),
    ("Nuvem de fumaça cobre Manaus pelo terceiro dia seguido", "negativo"),
    ("Cetesb divulga boletim diário de qualidade do ar nesta segunda", "neutro"),
    ("Especialistas explicam como é calculado o índice de qualidade do ar", "neutro"),
    ("Seminário sobre monitoramento do ar acontece na próxima semana em Brasília", "neutro"),
    ("Saiba quais poluentes são medidos pelas estações automáticas", "neutro"),
]


# ---------------------------------------------------------------- tarefas
def montar_tarefas() -> list[dict]:
    """Monta as tarefas do TP4. Sentimento, Q&A e resumo usam os MESMOS prompts da rota
    POST /processar_texto (src/llm.py), para que a avaliação meça o que a API faz."""
    tarefas = []

    # 1) Classificação de sentimento (com gabarito)
    for i, (frase, rotulo) in enumerate(SENTIMENTO):
        tarefas.append({
            "id": f"sentimento-{i}", "tipo": "sentimento", "esperado": rotulo,
            "prompt": montar_prompt("sentimento", frase)[0],
        })

    # 2) Perguntas e respostas sobre a tabela oficial (com gabarito)
    tabela = dl.ler_csv(dl.ARQ_IQAR)
    if tabela is not None:
        for pol, cat in [("MP2.5", "Moderada"), ("O3", "Ruim"), ("CO", "Boa")]:
            linha = tabela[(tabela["poluente"] == pol) & (tabela["categoria"] == cat)].iloc[0]
            pergunta = f"Qual é a faixa de concentração de {pol} para a qualidade do ar '{cat}'?"
            tarefas.append({
                "id": f"qa-{pol}-{cat}", "tipo": "qa",
                "esperado": [f"{linha['conc_min']:g}", f"{linha['conc_max']:g}"],
                "prompt": montar_prompt("pergunta", pergunta)[0],
            })

    # 3) Resumo de notícia (avaliação manual)
    noticias = dl.ler_csv(dl.ARQ_NOTICIAS)
    if noticias is not None and not noticias.empty:
        n = noticias.iloc[0]
        tarefas.append({
            "id": "resumo-0", "tipo": "resumo", "esperado": None,
            "prompt": montar_prompt("resumo", n["resumo"])[0],
        })

    # 4) Geração de boletim (avaliação manual)
    tarefas.append({
        "id": "boletim-0", "tipo": "boletim", "esperado": None,
        "prompt": ("Escreva um boletim de até 3 frases para a população de Manaus. Leitura atual: "
                   "IQAr 156, qualidade 'Muito Ruim', poluente crítico MP2.5 (98 µg/m³). "
                   "Inclua uma recomendação para quem pratica esportes ao ar livre."),
    })
    return tarefas


def conferir(tarefa: dict, resposta: str) -> bool | None:
    """Confere automaticamente as tarefas com gabarito; None = avaliação manual."""
    if tarefa["tipo"] == "sentimento":
        return rotulo_sentimento(resposta) == tarefa["esperado"]
    if tarefa["tipo"] == "qa":
        # Compara como número, para aceitar "25", "25,0" ou "25.0".
        numeros = {float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", resposta)}
        return all(float(e) in numeros for e in tarefa["esperado"])
    return None


# ---------------------------------------------------------------- ambiente
def relatorio_ambiente(torch) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_disponivel": torch.cuda.is_available()}
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = None
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update(gpu=prop.name, capacidade=f"{prop.major}.{prop.minor}",
                    vram_total_gb=round(prop.total_memory / 1024**3, 2))
    return info


def avaliar_modelo(nome: str, cfg: dict, tarefas: list[dict], usar_gpu: bool,
                   max_tokens: int) -> dict:
    import torch

    print(f"\n=== {nome} ({cfg['repo']}, {cfg['quant'] if usar_gpu else 'fp32/CPU'})")
    if usar_gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        import psutil
        processo = psutil.Process()
    except ImportError:
        processo = None

    inicio = time.perf_counter()
    try:
        tokenizer, modelo, _ = carregar(cfg["repo"], "gpu" if usar_gpu else "cpu",
                                        cfg["quant"] if usar_gpu else "fp32")
    except Exception as erro:  # sem memória, sem internet, biblioteca faltando...
        print(f"  ! Não foi possível carregar: {erro}")
        return {"modelo": nome, "repo": cfg["repo"], "erro": str(erro)[:300]}
    t_carga = time.perf_counter() - inicio
    print(f"  carregado em {t_carga:.1f}s")

    resultados, tokens, segundos = [], 0, 0.0
    for t in tarefas:
        try:
            resposta, n_tok, dur = gerar(tokenizer, modelo, t["prompt"], max_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"  ! {t['id']}: memória da GPU insuficiente")
            resultados.append({**t, "resposta": None, "correto": False, "erro": "OOM"})
            torch.cuda.empty_cache()
            continue
        tokens, segundos = tokens + n_tok, segundos + dur
        ok = conferir(t, resposta)
        marca = {True: "✔", False: "✘", None: "·"}[ok]
        print(f"  {marca} {t['id']:<16} {n_tok:>4} tok  {dur:5.1f}s  {resposta[:70]!r}")
        resultados.append({**t, "resposta": resposta, "correto": ok,
                           "tokens": n_tok, "segundos": round(dur, 2)})

    def taxa(tipo):
        itens = [r for r in resultados if r["tipo"] == tipo]
        return round(sum(bool(r["correto"]) for r in itens) / len(itens), 2) if itens else None

    resumo = {
        "modelo": nome, "repo": cfg["repo"],
        "formato": cfg["quant"] if usar_gpu else "fp32-cpu",
        "carga_s": round(t_carga, 1),
        "tokens_por_s": round(tokens / segundos, 1) if segundos else None,
        "vram_pico_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if usar_gpu else None,
        "ram_processo_gb": round(processo.memory_info().rss / 1024**3, 2) if processo else None,
        "acerto_sentimento": taxa("sentimento"),
        "acerto_qa": taxa("qa"),
    }
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    (PASTA_SAIDA / f"{nome}.json").write_text(
        json.dumps({"resumo": resumo, "tarefas": resultados}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    del modelo, tokenizer
    gc.collect()
    if usar_gpu:
        torch.cuda.empty_cache()
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelos", nargs="+", choices=list(CANDIDATOS), default=list(CANDIDATOS))
    parser.add_argument("--cpu", action="store_true", help="Roda no processador, sem GPU.")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--somente-prompts", action="store_true",
                        help="Mostra as tarefas e sai (não precisa de torch).")
    args = parser.parse_args()

    tarefas = montar_tarefas()
    if args.somente_prompts:
        for t in tarefas:
            print(f"--- {t['id']} (esperado: {t['esperado']})\n{t['prompt']}\n")
        return

    import torch

    ambiente = relatorio_ambiente(torch)
    print("Ambiente:", json.dumps(ambiente, ensure_ascii=False))
    usar_gpu = ambiente["cuda_disponivel"] and not args.cpu
    if not usar_gpu and not args.cpu:
        print("! CUDA não disponível: rodando na CPU. Veja a seção de instalação em "
              "docs/model/ModelSelection.md se você esperava usar a GPU.")

    resumos = [avaliar_modelo(n, CANDIDATOS[n], tarefas, usar_gpu, args.max_tokens)
               for n in args.modelos]
    tabela = pd.DataFrame(resumos)
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    tabela.assign(**{k: v for k, v in ambiente.items() if not isinstance(v, bool)}) \
          .to_csv(PASTA_SAIDA / "resumo.csv", index=False)
    print("\nResumo:\n" + tabela.to_string(index=False))
    print(f"\nDetalhes em {PASTA_SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
