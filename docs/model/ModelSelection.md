# Seleção do Modelo de Linguagem (LLM local): Éter

> Fase *Modeling* do TDSP, entrega TP4. Este documento justifica a escolha do LLM que roda
> **localmente** no projeto, pelos critérios de desempenho, custo computacional local e
> acessibilidade. A seção 6 traz a avaliação feita na máquina de desenvolvimento.

## 1. Contexto e requisitos

O corpus preparado no TP3 (`src/llm_prep.py` → `data/processed/corpus_llm.jsonl`) será usado em
quatro cenários:

| Cenário | Entrada | Saída esperada |
|---|---|---|
| Classificação de sentimento | manchete de notícia | positivo / negativo / neutro |
| Perguntas e respostas (Q&A) | pergunta + trechos da tabela oficial e do texto de referência | resposta curta e fiel ao contexto |
| Resumo automático | título + resumo de notícias | uma ou duas frases |
| Geração de texto | leitura atual do app (IQAr, poluente crítico) | boletim curto com recomendação |

Requisitos derivados:

- **Idioma:** entender e escrever bem em português do Brasil.
- **Versatilidade:** um único modelo deve cobrir as quatro tarefas por *prompt*, porque o projeto
  não tem dados rotulados para treinar um classificador.
- **Execução local:** o modelo roda na máquina de desenvolvimento, sem API paga, com a
  biblioteca Transformers da Hugging Face.
- **Hardware disponível:** GPU NVIDIA **GeForce GTX 1050 Ti com 4 GB de VRAM**. É uma placa da
  arquitetura Pascal (capacidade de computação 6.1) e isso impõe restrições (seção 4.2).

## 2. Qual tipo de arquitetura?

| Arquitetura | Exemplos | Para que foi treinada | Atende ao Éter? |
|---|---|---|---|
| Codificador (*encoder-only*) | BERT, BERTimbau | Entender texto: classificação, extração | Só o sentimento, e mesmo assim exigiria *fine-tuning* com dados rotulados. Não gera texto (sem resumo, boletim ou resposta livre). |
| Codificador-decodificador | T5, PTT5, mT5 | Transformar texto em texto (tradução, resumo) | Faz resumo bem, mas os modelos abertos em português são antigos e não seguem instruções de forma confiável sem *fine-tuning*. |
| Decodificador (*decoder-only*) com ajuste de instruções | família GPT, Qwen, Llama, Gemma | Seguir instruções e gerar texto | **Sim:** faz as quatro tarefas só com *prompt* (*zero-shot*). |

**Conclusão:** o projeto precisa de um modelo **decodificador ajustado para instruções**
(*instruct*). Modelos do tipo BERT continuam úteis como apoio (por exemplo, para gerar
*embeddings* na busca de trechos do Q&A), mas não como o LLM principal.

## 3. Candidatos e triagem

A triagem considerou modelos abertos, disponíveis no Hugging Face Hub e com suporte a português.
A estimativa de memória é a dos pesos. É preciso somar a memória do contexto (*KV cache*), o
contexto CUDA (~0,3 GB) e o que o Windows já usa da placa para a tela.

| Modelo | Parâmetros | Licença | Memória estimada | Situação |
|---|---|---|---|---|
| **Qwen3.5-0.8B** | 0,8B | Apache 2.0 | **~1,6 GB em FP16** | **Escolhido (FP16), após a avaliação local** |
| Qwen3.5-2B | 2B | Apache 2.0 | ~4 GB em FP16; ~2 GB em 4 bits | Finalista: mais fluente, mas 3,7× mais lento na placa (seção 6) |
| Qwen3-1.7B | 1,7B | Apache 2.0 | ~1,4 GB em 4 bits | Comparação (geração anterior) |
| Qwen3.5-4B | 4B | Apache 2.0 | ~3 GB em 4 bits | Descartado: sem folga nos 4 GB |
| Gemma 4 E2B | 2,3B efetivos (5,1B com *embeddings*) | Apache 2.0 | acima de 4 GB | Descartado: os *embeddings* sozinhos já ocupam a placa |
| Qwen3-8B / Llama 3.1 8B | 8B | Apache 2.0 / Llama | ~5–6 GB em 4 bits | Descartados: exigem 8 GB de VRAM |
| BERTimbau / PTT5 | 0,1–0,8B | MIT / Apache | < 1 GB | Descartados como LLM principal (seção 2) |

## 4. Critérios

### 4.1 Desempenho

Os resultados publicados pela equipe Qwen no *model card* do Qwen3.5-2B comparam os três
candidatos finais. Seguem os indicadores mais ligados ao Éter (quanto maior, melhor):

| Benchmark | O que mede | Qwen3.5-2B | Qwen3.5-0.8B | Qwen3-1.7B |
|---|---|---|---|---|
| MMLU-Pro (sem raciocínio) | conhecimento geral | **55,3** | 29,7 | 40,2 |
| MMLU-Redux (sem raciocínio) | conhecimento geral | **69,2** | 48,5 | 64,4 |
| MMMLU (sem raciocínio) | conhecimento em várias línguas | **56,9** | 34,1 | 46,7 |
| IFEval (sem raciocínio) | seguir instruções de formato | 61,2 | 52,1 | **68,2** |
| INCLUDE (com raciocínio) | conhecimento regional, multilíngue | **55,4** | 40,6 | 51,8 |
| WMT24++ (com raciocínio) | tradução multilíngue | **45,8** | 27,2 | 39,3 |

Leitura dos números:

- O **Qwen3.5-2B** lidera em conhecimento geral e multilíngue. Isso pesa para responder
  perguntas sobre qualidade do ar e escrever bem em português. A família Qwen3.5 cobre 201
  línguas e dialetos.
- O **Qwen3-1.7B** segue melhor instruções de formato (IFEval) no modo sem raciocínio. Para o
  Éter isso importa na classificação ("responda com uma palavra"). Esse ponto é conferido na
  avaliação local (seção 6), em vez de decidido só pelo benchmark.
- O **Qwen3.5-0.8B** fica bem abaixo nos *benchmarks*. Mesmo assim, entra na avaliação local por
  ser o único finalista que cabe inteiro na placa sem quantização. A seção 6 mostra que isso
  pesa mais do que os *benchmarks* sugerem.
- O Qwen3.5-2B trabalha **sem modo de raciocínio por padrão**. Isso é adequado ao app: as
  respostas são diretas, mais curtas e mais rápidas.

### 4.2 Custo computacional local

A GTX 1050 Ti define o teto. Quatro pontos guiaram a escolha:

1. **4 GB de VRAM:** em FP16 (2 bytes por parâmetro), um modelo de 2B ocupa ~4 GB só de pesos e
   não cabe. Com **quantização NF4 em 4 bits** (biblioteca `bitsandbytes`), as camadas
   lineares passam a ocupar ~0,5 byte por parâmetro. O Qwen3.5-2B cai para cerca de 2 GB e
   sobra espaço para o contexto. A perda de qualidade da quantização NF4 é pequena para esse uso.
2. **Sem bfloat16:** a arquitetura Pascal não tem esse formato numérico, que é o recomendado
   pela maioria dos modelos atuais. Os cálculos são feitos em **float16**
   (`bnb_4bit_compute_dtype=torch.float16`).
3. **Versão do PyTorch:** as versões do PyTorch compiladas para CUDA 12.8 ou mais recente
   **deixaram de suportar placas Maxwell e Pascal**. É obrigatório usar a variante **CUDA 12.6**
   (`cu126`), mantida como versão legada justamente para essas placas. Ela deixará de ser
   publicada a partir do PyTorch 2.15, por isso a instalação fixa `torch<2.15` (seção 5).
4. **Arquitetura do Qwen3.5:** o modelo combina camadas de atenção comum com camadas *Gated
   DeltaNet*. Os *kernels* otimizados dessas camadas dependem de bibliotecas extras
   (`flash-linear-attention`, `causal-conv1d`, baseadas em Triton) que não costumam funcionar
   no Windows nem em placas antigas. Sem elas, o Transformers usa automaticamente uma
   implementação em PyTorch puro. Funciona, mas fica mais lento que numa placa recente.
5. **4 bits economizam memória, não tempo:** a avaliação (seção 6) mostrou que, na 1050 Ti, o
   modelo em 4 bits é bem mais lento que um modelo menor em FP16. A cada token gerado, os pesos
   precisam ser descompactados, e placas antigas não têm instruções dedicadas para isso. Por
   isso, na GPU, o projeto usa FP16 sempre que o modelo cabe inteiro na memória.

Alternativa sem GPU: todos os candidatos também rodam na CPU (`--cpu` no script de
avaliação), em float32. Isso exige ~8 GB de RAM livres para o modelo de 2B e é bem mais lento.

### 4.3 Acessibilidade local

| Requisito | Qwen3.5-2B |
|---|---|
| Disponível no Hugging Face Hub | Sim: `Qwen/Qwen3.5-0.8B` (e `Qwen/Qwen3.5-2B`) |
| Precisa de login ou aceite de termos | Não |
| Licença | Apache 2.0 (permite uso acadêmico e comercial) |
| Carrega com Transformers | Sim: `AutoModelForCausalLM` carrega só a parte de texto (o modelo também aceita imagens, o que o Éter não usa) |
| Funciona offline | Sim, depois do primeiro download (fica no cache `~/.cache/huggingface`) |
| Tamanho do download | ~2 GB para o 0,8B; ~4–5 GB para o 2B (a quantização, quando usada, é feita na carga) |

## 5. Instalação passo a passo (Windows + GTX 1050 Ti)

Use um ambiente virtual separado do app Streamlit, porque o PyTorch é grande.

```powershell
# 1. Confirme o driver: a coluna "CUDA Version" deve ser 12.6 ou maior
nvidia-smi

# 2. Crie e ative um ambiente para a parte de LLM
python -m venv .venv-llm
.venv-llm\Scripts\activate

# 3. Instale o PyTorch na variante CUDA 12.6 (a única que ainda suporta Pascal)
pip install "torch<2.15" --index-url https://download.pytorch.org/whl/cu126

# 4. Confira: deve imprimir True e (6, 1)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"

# 5. Instale o restante (Transformers, bitsandbytes etc.) e as dependências do projeto.
#    Rode um comando por vez e só DEPOIS do passo 4: se o PyTorch ainda não estiver
#    instalado, o pip baixa a versão padrão, que no Windows só usa a CPU.
pip install -r requirements-llm.txt
pip install -r requirements.txt

# 6. Baixe o modelo para a pasta modelos/ (uma vez só, ~2 GB)
python scripts/llm/baixar_modelo.py
```

Se o passo 4 imprimir `False`, ou se aparecer o aviso *"sm_61 is not compatible with the current
PyTorch installation"*, o PyTorch instalado é de outra variante. Rode
`pip uninstall torch` e repita o passo 3.

## 6. Avaliação local

O script `scripts/llm/avaliar_modelos.py` roda os candidatos na própria máquina, com tarefas
montadas a partir dos dados do Éter:

- 6 manchetes com sentimento conhecido (acerto conferido automaticamente);
- 3 perguntas sobre a tabela do IQAr, com o trecho da tabela no *prompt* (acerto conferido
  pelos números da resposta);
- 1 resumo de notícia e 1 boletim (avaliados manualmente).

As respostas são geradas sem amostragem (`do_sample=False`), para que o teste seja
reproduzível.

```powershell
python scripts/llm/avaliar_modelos.py
```

### Rodada 1: CPU (float32)

Na primeira execução, o PyTorch instalado era a versão padrão do PyPI, que no Windows só usa o
processador. Os três modelos rodaram na CPU. Mesmo assim, a rodada serve como medida do pior caso
(máquina sem GPU compatível):

| Modelo | Formato | Carga (s) | Tokens/s | RAM do processo (GB) | Acerto sentimento | Acerto Q&A |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | FP32 / CPU | 160,8 | 1,4 | 7,59 | 0,67 (4/6) | 1,00 (3/3) |
| Qwen3.5-0.8B | FP32 / CPU | 65,1 | 3,2 | 3,37 | 0,67 (4/6) | 1,00 (3/3) |
| Qwen3-1.7B | FP32 / CPU | 115,9 | 1,6 | 7,14 | 0,50 (3/6) | 1,00 (3/3) |

Leitura:

- **Q&A com contexto:** os três acertaram todas as perguntas. Quando o trecho da tabela vai no
  *prompt*, até o menor modelo localiza e devolve a faixa correta. Isso favorece a estratégia de
  sempre enviar o contexto (como faz a rota `POST /processar_texto`).
- **Sentimento:** os dois Qwen3.5 empataram (4 de 6) e o Qwen3-1.7B ficou atrás (3 de 6). Com só
  6 frases, um acerto a mais ou a menos muda a taxa em 17 pontos. O teste não separa o 2B do
  0,8B nesse critério.
- **Custo na CPU:** o 0,8B é 2,3 vezes mais rápido que o 2B (3,2 contra 1,4 tokens/s), carrega
  em menos da metade do tempo e usa menos da metade da RAM. Um resumo de 120 tokens levaria
  cerca de 40 s no 0,8B e cerca de 85 s no 2B.
- **Qwen3-1.7B:** é mais lento e usa mais memória que o 0,8B, e acertou menos. Fica descartado.

### Rodada 2: GPU (GTX 1050 Ti)

Depois de reinstalar o PyTorch na variante `cu126`:

| Modelo | Formato | Carga (s) | Tokens/s | VRAM de pico (GB) | Acerto sentimento | Acerto Q&A |
|---|---|---|---|---|---|---|
| Qwen3.5-2B | NF4 4 bits | 27,4 | 3,1 | 1,73 | 0,67 (4/6) | 1,00 (3/3) |
| **Qwen3.5-0.8B** | **FP16** | **6,7** | **11,6** | 1,46 | 0,67 (4/6) | 1,00 (3/3) |
| Qwen3-1.7B | NF4 4 bits | 16,8 | 3,1 | 2,20 | 0,67 (4/6) | 1,00 (3/3) |

- **Memória:** os três couberam com folga nos 4 GB (pico de 2,2 GB).
- **Velocidade:** o 0,8B em FP16 foi **3,7 vezes mais rápido** que o 2B em 4 bits e carregou
  4 vezes mais rápido. Um resumo de 120 tokens leva cerca de 10 s no 0,8B e 40 s no 2B
  (ver o item 5 da seção 4.2).
- **Acerto:** empate nas tarefas com gabarito.

**Análise dos erros de sentimento:** os dois Qwen3.5 erraram exatamente as duas manchetes
neutras. O 2B classificou ambas como positivas. O 0,8B classificou uma como negativa e a outra
como positiva. O *prompt* não definia o que é "neutro", e os modelos tendem a dar um rótulo
emocional a qualquer notícia sobre poluição. Uma das frases de teste ("Prefeitura instala nova
estação de monitoramento") também é ambígua e pode ser lida como positiva.

**Avaliação manual dos textos livres:**

| | Qwen3.5-2B | Qwen3.5-0.8B |
|---|---|---|
| Resumo | Fluente, mas **inverte uma causa**: diz que as chuvas provocaram a suspensão das aulas, quando na notícia a chuva *melhorou* o ar | Português travado ("redes... suspensas... estão suspendendo"), mas **fatos corretos** |
| Boletim | Claro e curto, mas **não cumpre** o pedido de recomendação específica para quem pratica esporte | Cumpre todos os pontos pedidos. Tem deslizes de vocabulário ("ar extremamente poluente") e uma recomendação de esporte mais branda que a regra do próprio app para "Muito Ruim" |

O 2B escreve melhor, mas foi o único a errar um fato. Para um app de saúde, fidelidade ao
conteúdo pesa mais que estilo. Em qualidade, os dois ficam próximos, cada um com um tipo de
falha.

### Rodada 3: *prompt* de sentimento revisado

A partir dos erros da rodada 2:

- **Prompt (`src/llm.py`, `PROMPT_SENTIMENTO`):** agora define cada rótulo e traz três exemplos
  (*few-shot*), um por classe, diferentes das frases de teste.
- **Conjunto de teste:** passou de 6 para 12 frases (4 por classe). A frase ambígua foi
  substituída. Com 12 itens, cada erro pesa 8 pontos em vez de 17.
- **Avaliação = API:** o script agora usa exatamente os mesmos *prompts* da rota
  `POST /processar_texto`, então mede o que o app de fato entrega.

| Modelo | Formato | Carga (s) | Tokens/s | VRAM de pico (GB) | Acerto sentimento (12) | Acerto Q&A (3) |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | FP16 | 14,2 | 11,1 | 1,47 | **0,92** (11/12) | **0,33** (1/3) |
| Qwen3.5-2B | NF4 4 bits | 18,6 | 3,0 | 1,74 | **1,00** (12/12) | 1,00 (3/3) |

- **Sentimento:** o *prompt* com definições e exemplos resolveu o problema das manchetes
  neutras. O 0,8B subiu de 67% para 92% e o 2B acertou todas, num conjunto de teste duas vezes
  maior.
- **Q&A:** o 0,8B caiu de 100% para 33%. A única mudança nessa tarefa foi o *prompt*, que
  passou a ser o mesmo da API e acrescenta as faixas do índice ("índice 41–80") ao lado das
  faixas de concentração. *Diagnóstico em andamento:* a hipótese é que o modelo menor confunde
  os dois conjuntos de números. O 2B não teve esse problema.

## 7. Decisão

**Modelo escolhido: `Qwen/Qwen3.5-0.8B`, em FP16, rodando na GTX 1050 Ti.**

A análise teórica (seções 3 e 4.1) apontava o Qwen3.5-2B, que tem os melhores *benchmarks*. A
avaliação local mudou a decisão:

- **Desempenho:** nas tarefas do Éter, o 0,8B **empatou** com o 2B em sentimento e Q&A (rodadas
  1 e 2). Nos textos livres, foi menos fluente, mas não errou fatos, enquanto o 2B inverteu uma
  causa no resumo. A vantagem do 2B nos *benchmarks* (conhecimento geral, raciocínio) quase não
  aparece em tarefas curtas e com contexto fornecido no *prompt*, que são as do app.
- **Custo computacional local:** 1,5 GB de VRAM, carga em 7 s e **11,6 tokens/s**, contra
  3,1 tokens/s do 2B. Para quem usa o app, é a diferença entre esperar 10 s e 40 s por um
  resumo. Por caber inteiro em FP16, dispensa a quantização, que nesta placa custa caro.
- **Acessibilidade:** mesma família, licença (Apache 2.0) e forma de carga do 2B; download de
  ~2 GB; roda também em máquinas sem GPU (3,2 tokens/s na CPU, rodada 1).

**Alternativa configurável:** o `Qwen/Qwen3.5-2B` em 4 bits continua suportado
(`ETER_LLM_MODELO`, seção 9). Ele faz sentido para gerar textos fora do fluxo interativo, como
boletins pré-gerados, onde a espera não incomoda. O `Qwen/Qwen3-1.7B` foi descartado: tem a
mesma velocidade do 2B e não supera nenhum dos dois.

## 8. Integração com a API (FastAPI)

O modelo escolhido é servido pela API do projeto, sem nenhum serviço externo:

| Arquivo | Papel |
|---|---|
| `scripts/llm/baixar_modelo.py` | baixa o modelo uma vez para `modelos/<nome>` (fora do Git) |
| `src/llm.py` | carrega o modelo com Transformers (4 bits na GPU, FP32 na CPU), monta os *prompts* e gera o texto |
| `api/main.py` | `POST /processar_texto` (tarefas `resumo`, `sentimento`, `pergunta`) e `GET /llm/status` |
| `pages/api_llm.py` | aba **LLM local** do app: envia texto (ou uma notícia coletada) para a rota |
| `tests/test_processar_texto.py` | testes da rota com um LLM falso (`pytest -q`) |

Decisões de implementação:

- **Carga preguiçosa e única:** o modelo é carregado no primeiro pedido e fica na memória. Os
  pedidos seguintes respondem sem recarregar.
- **Um pedido por vez:** um cadeado (`threading.Lock`) evita que duas gerações simultâneas
  estourem os 4 GB da placa.
- **Rota síncrona (`def`):** o FastAPI executa a geração numa thread separada, então as outras
  rotas continuam respondendo enquanto o modelo trabalha.
- **Decodificação gulosa (`do_sample=False`):** o mesmo texto gera sempre a mesma resposta,
  o que é desejável para classificação e facilita os testes.
- **Offline de verdade:** se `modelos/<nome>` existe, o Transformers lê dali e não acessa a
  internet. Para garantir isso também com o cache do Hugging Face, defina
  `HF_HUB_OFFLINE=1` antes de subir a API.
- **API funciona sem o LLM:** `torch` e `transformers` só são importados no primeiro pedido.
  Sem eles, as outras rotas funcionam normalmente e `/processar_texto` responde `503` com
  instruções.

## 9. Configuração

| Variável | Padrão | Uso |
|---|---|---|
| `ETER_LLM_MODELO` | `Qwen/Qwen3.5-0.8B` | troca o modelo (ex.: `Qwen/Qwen3.5-2B` para textos mais fluentes, aceitando ~40 s por resumo) |
| `ETER_LLM_DISPOSITIVO` | `auto` | `gpu` ou `cpu` para forçar |
| `ETER_LLM_QUANT` | `auto` | `4bit`, `fp16` ou `fp32` (auto: FP16 na GPU para o 0,8B, 4 bits para os maiores, FP32 na CPU) |
| `HF_HUB_OFFLINE` | (vazio) | `1` impede qualquer acesso ao Hugging Face |

## 10. Referências

- Qwen Team. *Qwen3.5-2B* (model card, benchmarks e arquitetura). https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen Team. *Qwen3.5-0.8B* (model card). https://huggingface.co/Qwen/Qwen3.5-0.8B
- Google DeepMind. *Gemma 4 E2B/E4B* (model card). https://huggingface.co/google/Gemma-4-E4B
- PyTorch. *Maxwell and Pascal architecture support removed in CUDA 12.8 and 12.9 builds*.
  https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128
- PyTorch. *CUDA 12.6 wheels will no longer be published from PyTorch 2.15*.
  https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
- bitsandbytes. Requisitos de GPU (NF4 a partir da capacidade 6.0, Pascal).
  https://github.com/bitsandbytes-foundation/bitsandbytes
