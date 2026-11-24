"""LLM local do Éter (TP4): carga do modelo e tarefas de linguagem natural.

Tudo roda na máquina local com a biblioteca Transformers da Hugging Face, sem
chamar nenhum serviço na nuvem. O modelo é baixado uma vez com
`scripts/llm/baixar_modelo.py` para a pasta `modelos/`. Depois disso, ele é
lido direto do disco e funciona sem internet.

Usado por:
  * api/main.py                    -> rota POST /processar_texto
  * scripts/llm/avaliar_modelos.py -> comparação dos modelos candidatos

Configuração por variáveis de ambiente (todas opcionais):
  ETER_LLM_MODELO       repositório do Hugging Face (padrão: Qwen/Qwen3.5-0.8B)
  ETER_LLM_DISPOSITIVO  "auto" (GPU se houver), "gpu" ou "cpu"
  ETER_LLM_QUANT        "auto", "4bit", "fp16" ou "fp32" (auto: ver QUANT_NA_GPU)
  ETER_LLM_TIMEOUT_S    tempo máximo de geração por pedido, em segundos (padrão: 120)
  ETER_LLM_ESPERA_S     quanto um pedido espera na fila se o modelo estiver ocupado (padrão: 30)

Falhas viram exceções específicas (`LLMErro` e subclasses, no fim deste arquivo), que
a API traduz em códigos HTTP e mensagens claras (ver api/erros.py).

`torch` e `transformers` só são importados quando o modelo é carregado, para
que o app Streamlit e o resto da API continuem funcionando sem eles.
"""
from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PASTA_MODELOS = RAIZ / "modelos"
# Escolha justificada em docs/model/ModelSelection.md (rodada 2: mesmo acerto do 2B,
# 3,7x mais rápido na GTX 1050 Ti).
MODELO_PADRAO = "Qwen/Qwen3.5-0.8B"
# Formato padrão na GPU: modelos que cabem inteiros em FP16 rodam sem quantização
# (em placas antigas, descompactar os pesos de 4 bits a cada passo custa caro);
# os maiores usam NF4 em 4 bits para caber nos 4 GB.
QUANT_NA_GPU = {"Qwen/Qwen3.5-0.8B": "fp16"}

INSTRUCAO_SISTEMA = (
    "Você é o assistente do Éter, um app brasileiro de qualidade do ar. "
    "Responda sempre em português do Brasil, de forma curta e direta."
)
ROTULOS_SENTIMENTO = ("positivo", "negativo", "neutro")


# ---------------------------------------------------------------- utilitários
def sem_acento(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def rotulo_sentimento(resposta: str) -> str | None:
    """Extrai o rótulo da resposta do modelo. None se vier nenhum ou mais de um."""
    achados = [r for r in ROTULOS_SENTIMENTO if r in sem_acento(resposta)]
    return achados[0] if len(achados) == 1 else None


def pasta_local(repo: str) -> Path:
    """Onde `baixar_modelo.py` guarda o repositório (ex.: modelos/Qwen3.5-2B)."""
    return PASTA_MODELOS / repo.split("/")[-1]


def origem_do_modelo(repo: str) -> str:
    """Usa a cópia em modelos/ se ela existir; senão, o nome do repositório (cache do HF)."""
    local = pasta_local(repo)
    return str(local) if (local / "config.json").exists() else repo


# ---------------------------------------------------------------- carga e geração
def carregar(repo: str, dispositivo: str = "auto", quant: str = "auto"):
    """Carrega tokenizer e modelo. Devolve (tokenizer, modelo, descrição do formato)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    usar_gpu = torch.cuda.is_available() and dispositivo != "cpu"
    if dispositivo == "gpu" and not usar_gpu:
        raise RuntimeError("ETER_LLM_DISPOSITIVO=gpu, mas o PyTorch não encontrou GPU CUDA.")
    if quant == "auto":
        quant = QUANT_NA_GPU.get(repo, "4bit") if usar_gpu else "fp32"

    origem = origem_do_modelo(repo)
    tokenizer = AutoTokenizer.from_pretrained(origem)
    if usar_gpu and quant == "4bit":
        from transformers import BitsAndBytesConfig
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            # Placas Pascal (GTX 10xx) não têm bfloat16: os cálculos ficam em float16.
            bnb_4bit_compute_dtype=torch.float16,
        )
        modelo = AutoModelForCausalLM.from_pretrained(
            origem, quantization_config=config, device_map={"": 0})
    elif usar_gpu:
        dtype = torch.float32 if quant == "fp32" else torch.float16
        modelo = AutoModelForCausalLM.from_pretrained(origem, dtype=dtype, device_map={"": 0})
    else:  # na CPU, float32 é o formato mais estável e rápido
        quant = "fp32"
        modelo = AutoModelForCausalLM.from_pretrained(origem, dtype=torch.float32)
    modelo.eval()
    return tokenizer, modelo, f"{quant}-{'gpu' if usar_gpu else 'cpu'}"


def gerar(tokenizer, modelo, prompt: str, max_tokens: int = 160,
          sistema: str = INSTRUCAO_SISTEMA,
          max_time: float | None = None) -> tuple[str, int, float]:
    """Gera uma resposta. Devolve (texto, tokens gerados, segundos).

    `max_time` (segundos) faz o próprio Transformers interromper a geração se ela
    passar do limite.
    """
    import torch

    mensagens = [{"role": "system", "content": sistema},
                 {"role": "user", "content": prompt}]
    entradas = tokenizer.apply_chat_template(
        mensagens, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,  # sem "modo raciocínio": respostas diretas e mais rápidas
    ).to(modelo.device)
    inicio = time.perf_counter()
    with torch.no_grad():
        # do_sample=False (decodificação gulosa): mesma entrada -> mesma saída
        saida = modelo.generate(**entradas, max_new_tokens=max_tokens, do_sample=False,
                                max_time=max_time,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    duracao = time.perf_counter() - inicio
    novos = saida[0][entradas["input_ids"].shape[-1]:]
    texto = tokenizer.decode(novos, skip_special_tokens=True)
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
    return texto, len(novos), duracao


# ---------------------------------------------------------------- tarefas do app
def poluentes_citados(pergunta: str) -> list[str]:
    """Siglas da tabela citadas na pergunta ("pm 2,5" conta como MP2.5)."""
    from src import data_loader as dl

    # A busca exige a sigla isolada: "CO" não casa com "COMO" nem com "CO2".
    norm = re.sub(r"(\d),(\d)", r"\1.\2", dl.normalizar_poluente(pergunta).replace("MP ", "MP"))
    return [p for p in ("MP10", "MP2.5", "O3", "CO", "NO2", "SO2")
            if re.search(rf"(?<![A-Z0-9]){re.escape(p)}(?![A-Z0-9]|\.\d)", norm)]


def contexto_iqar(pergunta: str) -> tuple[str, list[str]]:
    """Trechos da tabela oficial para a tarefa 'pergunta'.

    Devolve (texto do contexto, poluentes incluídos). Usa só os poluentes citados;
    se nenhum for citado, envia a tabela inteira.
    """
    import pandas as pd

    from src import data_loader as dl

    tabela = dl.ler_csv(dl.ARQ_IQAR)
    if tabela is None:
        return "", []
    citados = poluentes_citados(pergunta)
    if citados:
        tabela = tabela[tabela["poluente"].isin(citados)]
    # Só concentrações em cada linha. Na rodada 3, com "(índice 41–80)" na mesma linha,
    # o modelo de 0,8B respondia a faixa do índice em vez da concentração (1 de 3 certas).
    linhas = []
    for r in tabela.itertuples():
        faixa = (f"de {r.conc_min:g} a {r.conc_max:g}" if pd.notna(r.conc_max)
                 else f"acima de {r.conc_min:g}")
        linhas.append(f"- {r.poluente}, {r.categoria}: {faixa} {r.unidade}")
    # As faixas do índice entram num bloco separado, só se a pergunta for sobre o índice.
    if re.search(r"\b(índice|indice|iqar)\b", pergunta, flags=re.I):
        faixas = tabela.drop_duplicates("categoria")
        linhas.append("\nFaixas do índice IQAr (as mesmas para todos os poluentes):")
        linhas += [f"- {r.categoria}: índice de {r.indice_min} a {r.indice_max}"
                   for r in faixas.itertuples()]
    return "\n".join(linhas), list(dict.fromkeys(tabela["poluente"]))


# A definição de cada rótulo e os exemplos corrigem o erro mais comum dos modelos
# pequenos na rodada 2: chamar de "positivo"/"negativo" notícias só informativas.
PROMPT_SENTIMENTO = """Classifique o sentimento da notícia abaixo, sobre qualidade do ar:
- positivo: relata melhora do ar ou uma boa notícia para a saúde das pessoas;
- negativo: relata piora do ar, poluição, fumaça ou danos à saúde;
- neutro: só informa, explica ou divulga algo, sem indicar melhora nem piora.

Exemplos:
Ventos dispersam a fumaça e o ar volta a ficar bom -> positivo
Queimadas elevam a poluição e hospitais ficam lotados -> negativo
Órgão ambiental publica relatório anual sobre o ar -> neutro

Responda com uma única palavra: positivo, negativo ou neutro.

Notícia: {texto}"""


def montar_prompt(tarefa: str, texto: str) -> tuple[str, int]:
    """Devolve (prompt, limite de tokens) para cada tarefa suportada pela API."""
    if tarefa == "resumo":
        return (f"Resuma o texto abaixo em no máximo duas frases.\n\nTexto:\n{texto}", 120)
    if tarefa == "sentimento":
        return (PROMPT_SENTIMENTO.format(texto=texto), 5)
    if tarefa == "pergunta":
        return ("Responda à pergunta usando somente a tabela oficial do IQAr abaixo. Se a "
                "resposta não estiver na tabela, diga que não sabe.\n\n"
                f"Tabela:\n{contexto_iqar(texto)[0]}\n\nPergunta: {texto}", 150)
    raise ValueError(f"Tarefa desconhecida: {tarefa}")


# ---------------------------------------------------------------- erros do LLM
class LLMErro(Exception):
    """Base das falhas do LLM local. A API traduz cada subclasse num código HTTP."""


class LLMDependenciasAusentes(LLMErro):
    """torch/transformers (ou bitsandbytes) não estão instalados neste ambiente."""


class LLMModeloNaoEncontrado(LLMErro):
    """O modelo não está em modelos/ nem no cache, e não foi possível baixá-lo."""


class LLMMemoriaInsuficiente(LLMErro):
    """Faltou memória na GPU (ou RAM) para carregar o modelo ou gerar a resposta."""


class LLMFalhaCarga(LLMErro):
    """Qualquer outra falha ao carregar o modelo."""


class LLMOcupado(LLMErro):
    """O modelo está atendendo outro pedido e a espera passou do limite."""

    def __init__(self, espera_s: float):
        super().__init__(f"o modelo está ocupado há mais de {espera_s:g} s")
        self.espera_s = espera_s


class LLMTempoEsgotado(LLMErro):
    """A geração passou do tempo máximo configurado."""

    def __init__(self, limite_s: float):
        super().__init__(f"a geração passou do limite de {limite_s:g} s")
        self.limite_s = limite_s


class LLMFalhaGeracao(LLMErro):
    """Erro inesperado durante a geração do texto."""


def _sem_memoria(erro: BaseException) -> bool:
    return type(erro).__name__ == "OutOfMemoryError" or "out of memory" in str(erro).lower()


def _liberar_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class LLMLocal:
    """Mantém UM modelo carregado na memória e atende um pedido por vez.

    A carga só acontece no primeiro uso (leva de segundos a alguns minutos).
    O cadeado (`threading.Lock`) evita que dois pedidos simultâneos gerem texto
    ao mesmo tempo na mesma GPU, o que estouraria a memória de uma placa de 4 GB.
    """

    def __init__(self) -> None:
        self.repo = os.getenv("ETER_LLM_MODELO", MODELO_PADRAO)
        self.dispositivo = os.getenv("ETER_LLM_DISPOSITIVO", "auto")
        self.quant = os.getenv("ETER_LLM_QUANT", "auto")
        self._tokenizer = self._modelo = None
        self.formato: str | None = None
        self.tempo_carga_s: float | None = None
        self.timeout_s = float(os.getenv("ETER_LLM_TIMEOUT_S", "120"))
        self.espera_s = float(os.getenv("ETER_LLM_ESPERA_S", "30"))
        self.ultimo_erro: str | None = None  # aparece em GET /llm/status, ajuda no diagnóstico
        self._cadeado = threading.Lock()

    @property
    def carregado(self) -> bool:
        return self._modelo is not None

    def status(self) -> dict:
        return {"modelo": self.repo, "origem": origem_do_modelo(self.repo),
                "carregado": self.carregado, "ocupado": self._cadeado.locked(),
                "formato": self.formato, "tempo_carga_s": self.tempo_carga_s,
                "timeout_s": self.timeout_s, "ultimo_erro": self.ultimo_erro}

    def _falhou(self, erro: LLMErro) -> LLMErro:
        self.ultimo_erro = f"{type(erro).__name__}: {erro}"
        return erro

    def _garantir_carregado(self) -> None:
        """Carrega o modelo no primeiro uso. Se falhar, o próximo pedido tenta de novo."""
        if self._modelo is not None:
            return
        inicio = time.perf_counter()
        try:
            self._tokenizer, self._modelo, self.formato = carregar(
                self.repo, self.dispositivo, self.quant)
        except ImportError as erro:
            raise self._falhou(LLMDependenciasAusentes(str(erro))) from erro
        except OSError as erro:  # modelo inexistente, sem internet e sem cópia local...
            raise self._falhou(LLMModeloNaoEncontrado(str(erro).split("\n")[0])) from erro
        except Exception as erro:  # noqa: BLE001 - qualquer outra falha vira erro de carga
            _liberar_gpu()
            self._tokenizer = self._modelo = None
            if _sem_memoria(erro):
                raise self._falhou(LLMMemoriaInsuficiente(str(erro).split("\n")[0])) from erro
            raise self._falhou(LLMFalhaCarga(str(erro).split("\n")[0])) from erro
        self.tempo_carga_s = round(time.perf_counter() - inicio, 1)
        self.ultimo_erro = None

    def processar(self, tarefa: str, texto: str) -> dict:
        """Executa uma das tarefas da rota /processar_texto (resumo, sentimento, pergunta)."""
        prompt, max_tokens = montar_prompt(tarefa, texto)
        return self.gerar_texto(prompt, max_tokens)

    def gerar_texto(self, prompt: str, max_tokens: int) -> dict:
        """Gera texto para um prompt qualquer, com fila, limites de tempo e erros tratados.

        Usado pelas tarefas acima e pela sumarização do noticiário (src/sumarizacao.py).
        """
        # Espera a vez na fila por no máximo `espera_s`; depois desiste com LLMOcupado,
        # em vez de deixar o cliente esperando sem resposta.
        if not self._cadeado.acquire(timeout=self.espera_s):
            raise LLMOcupado(self.espera_s)
        try:
            self._garantir_carregado()
            try:
                resposta, n_tokens, segundos = gerar(
                    self._tokenizer, self._modelo, prompt, max_tokens, max_time=self.timeout_s)
            except Exception as erro:  # noqa: BLE001
                _liberar_gpu()
                if _sem_memoria(erro):
                    raise self._falhou(LLMMemoriaInsuficiente(str(erro).split("\n")[0])) from erro
                raise self._falhou(LLMFalhaGeracao(f"{type(erro).__name__}: {erro}")) from erro
        finally:
            self._cadeado.release()
        if segundos >= self.timeout_s:
            raise self._falhou(LLMTempoEsgotado(self.timeout_s))
        # Saída "crua": quem chama (a API) interpreta e valida conforme a tarefa.
        return {"resultado": resposta, "tokens_gerados": n_tokens,
                "tempo_s": round(segundos, 2)}
