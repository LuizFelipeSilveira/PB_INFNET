# 🌬️ Éter: Monitor Inteligente de Qualidade do Ar

Aplicação Streamlit que traduz índices de poluição do ar em recomendações simples de saúde,
combinando dados em tempo real (OpenWeather) com tabelas oficiais de referência (CETESB / CONAMA 491/2018)
e notícias coletadas por web scraping.

## Estrutura do projeto (TDSP)

O projeto segue o Team Data Science Process, com cada pasta ligada a uma fase do ciclo de vida:

```
eter/
├── app.py                        # Deployment: ponto de entrada do Streamlit (multipage)
├── pages/                        #   uma página por aba (menu de navegação automático)
│   ├── monitor.py, noticias.py, referencia.py, dados.py, api_llm.py, sobre.py
├── api/                          # Deployment: API HTTP independente (FastAPI)
│   ├── main.py                   #   rotas (GET /iqar/tabela, /iqar/calcular, /cidades,
│   │                              #        GET+POST /leituras, POST /processar_texto,
│   │                              #        GET /llm/status)
│   └── schemas.py                #   modelos Pydantic de entrada/saída
├── src/                          # Código reutilizável (usado pelo app E pela API)
│   ├── iqar.py                   #   cálculo do IQAr e recomendações
│   ├── api.py                    #   OpenWeather + modo simulado
│   ├── data_loader.py            #   leitura de dados e validação de uploads
│   ├── state.py                  #   sessão e cache compartilhados entre páginas
│   ├── ui.py                     #   componentes visuais compartilhados
│   ├── llm_prep.py               #   Modeling: consolida o corpus para os LLMs
│   └── llm.py                    #   Modeling: carrega o LLM local e executa as tarefas (TP4)
├── scripts/
│   ├── scraping/                 # Data Acquisition: coletores com Beautiful Soup (+ Selenium
│   │   ├── scrape_iqar.py        #   para a página da CETESB — ver Charter §6)
│   │   └── scrape_noticias.py
│   ├── prepare_corpus.py         # gera data/processed/corpus_llm.jsonl
│   └── llm/                      # Modeling (TP4)
│       ├── baixar_modelo.py      #   baixa o modelo do Hugging Face para modelos/
│       └── avaliar_modelos.py    #   compara os candidatos (velocidade, memória, acerto)
├── data/
│   ├── raw/                      #   saídas dos scrapers (CSV/TXT)
│   ├── processed/                #   corpus_llm.jsonl, leituras_api.csv (não versionados)
│   └── external/                 #   dados de terceiros
├── notebooks/                    # Data Understanding: análises exploratórias
├── docs/
│   ├── project/Charter.md        # Business Understanding
│   ├── data_report/DataSummaryReport.md
│   └── model/ModelSelection.md   # Modeling: escolha e justificativa do LLM local
├── modelos/                      # pesos do LLM baixados localmente (não versionados)
├── tests/                        # pytest (rota /processar_texto com LLM falso)
├── .streamlit/                   # tema e segredos (exemplo)
├── requirements.txt
└── requirements-llm.txt          # PyTorch/Transformers só para a parte de LLM (TP4)
```

## Como rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Coletar os dados da web (rodar separadamente, antes do app)
python scripts/scraping/scrape_iqar.py       # usa Selenium só para a CETESB (ver Charter §6)
python scripts/scraping/scrape_noticias.py

# 2. (Opcional) preparar o corpus para a próxima entrega (LLM)
python scripts/prepare_corpus.py

# 3. (Opcional) configurar a chave da OpenWeather
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # e edite a chave

# 4. Abrir o app (multipage, com menu de navegação na barra lateral)
streamlit run app.py
```

Sem chave de API o app funciona em modo simulado.

### Rodar a API (opcional, processo separado)

```bash
uvicorn api.main:app --reload
```

Documentação interativa (Swagger) em http://127.0.0.1:8000/docs. A aba **API & IA** do app tem um
mini-formulário para testar as rotas sem sair do Streamlit (basta a API estar rodando).

Sugestão de fluxo: uma branch por funcionalidade (`git checkout -b feat/upload-csv`), commits
pequenos com prefixos (`feat:`, `fix:`, `docs:`, `data:`) e merge na `main` via pull request.
O `.gitignore` já impede que `secrets.toml` seja enviado.

### LLM local (TP4, opcional)

A rota `POST /processar_texto` resume textos, classifica sentimento e responde perguntas sobre a
tabela do IQAr com um modelo do Hugging Face rodando **na sua máquina** (sem nuvem). A escolha
do modelo e a instalação para GPUs antigas (ex.: GTX 1050 Ti) estão em
`docs/model/ModelSelection.md`.

```bash
# depois de instalar o PyTorch (ver docs/model/ModelSelection.md, seção 5)
pip install -r requirements-llm.txt
python scripts/llm/baixar_modelo.py            # uma vez: baixa para modelos/
uvicorn api.main:app                           # o 1º pedido carrega o modelo
```

Exemplo de chamada:

```bash
curl -X POST http://127.0.0.1:8000/processar_texto -H "Content-Type: application/json" \
     -d '{"texto": "Fumaça deixa o ar de Manaus péssimo", "tarefa": "sentimento"}'
```

No app, a aba **API & IA → LLM local** faz o mesmo com um formulário. Testes: `pytest -q`.

### Formato das respostas e dos erros

Todas as rotas declaram um `response_model` (em `api/schemas.py`), e o FastAPI valida cada
resposta antes de enviá-la. Exemplo de resposta de `POST /processar_texto`:

```json
{
  "tarefa": "sentimento",
  "texto_entrada": "Fumaça deixa o ar de Manaus péssimo",
  "analise": {"tipo": "sentimento", "rotulo": "negativo", "rotulo_valido": true,
              "resposta_modelo": "negativo"},
  "metadados": {"modelo": "Qwen/Qwen3.5-0.8B", "formato": "fp16-gpu", "tokens_gerados": 2,
                "tempo_s": 0.4, "processado_em": "2026-11-16T21:30:00"}
}
```

O conteúdo de `analise` depende de `analise.tipo`:
- `resumo`: traz `resumo`, `palavras_entrada`, `palavras_resumo` e `taxa_compressao`;
- `sentimento`: traz `rotulo`, `rotulo_valido` e `resposta_modelo`;
- `pergunta`: traz `resposta`, `poluentes_consultados` e `fonte`.

### Tratamento de erros

Todos os erros seguem um único formato (`ErroAPI`), com o que aconteceu, o que fazer e um id
para o diagnóstico:

```json
{
  "status": 503,
  "erro": "llm_modelo_nao_encontrado",
  "mensagem": "O modelo de linguagem não foi encontrado nesta máquina (...).",
  "sugestao": "Baixe o modelo uma vez com `python scripts/llm/baixar_modelo.py` (...)",
  "campos": [],
  "id_requisicao": "19aeecc7"
}
```

O `id_requisicao` volta também no cabeçalho `X-Request-ID` e aparece na linha de log do
terminal da API. Com ele, dá para achar exatamente qual pedido falhou. Erros inesperados
(500) nunca expõem detalhes internos ao cliente: o *traceback* fica só no log.

| Status | `erro` | Quando |
|---|---|---|
| 400 | `poluente_invalido` | poluente inexistente em `/iqar/calcular` ou `POST /leituras` |
| 404 | `nao_encontrado`, `poluente_nao_encontrado` | rota inexistente; poluente fora da tabela |
| 422 | `entrada_invalida` | campo ausente, vazio, fora dos limites ou desconhecido (lista em `campos`) |
| 500 | `dados_corrompidos` | CSV de dados ilegível ou fora do formato |
| 500 | `llm_falha_geracao`, `erro_interno` | falha inesperada (detalhe no log, pelo id) |
| 502 | `resposta_invalida_do_modelo` | o LLM devolveu algo fora do contrato (ex.: resumo vazio) |
| 503 | `llm_dependencias_ausentes` | PyTorch/Transformers não instalados |
| 503 | `llm_modelo_nao_encontrado` | modelo não baixado (e sem internet para baixar) |
| 503 | `llm_memoria_insuficiente` | faltou memória na GPU ao carregar ou gerar |
| 503 | `llm_falha_carga` | outra falha ao carregar o modelo |
| 503 | `llm_ocupado` | outro pedido em processamento por mais de `ETER_LLM_ESPERA_S` (cabeçalho `Retry-After`) |
| 503 | `dados_indisponiveis`, `armazenamento_indisponivel` | tabela IQAr não gerada; falha ao gravar leituras |
| 504 | `llm_tempo_esgotado` | a geração passou de `ETER_LLM_TIMEOUT_S` (padrão 120 s) |

`GET /llm/status` mostra se o modelo está carregado ou ocupado e a última falha
(`ultimo_erro`). No app, a aba **LLM local** mostra cada erro com um título claro, a
sugestão e o id. Ela também avisa quando a API está fora do ar.

## Deploy (Streamlit Community Cloud)

1. Envie o repositório ao GitHub.
2. Em share.streamlit.io, clique em **New app**, escolha o repositório e o arquivo `app.py`.
3. Em **Advanced settings > Secrets**, cole `OPENWEATHER_API_KEY = "sua_chave"`.
4. Para atualizar as notícias em produção, rode os scrapers localmente e faça commit de `data/raw/`.

## Funcionalidades

| Requisito | Onde está |
|---|---|
| Interface dinâmica, múltiplas páginas | `app.py` + `pages/`, com menu de navegação (`st.navigation`) |
| Seletor de cidade, perfil, cenários, comparação, filtros, calculadora | Páginas **Monitor** e **Tabela IQAr** |
| Web scraping (Beautiful Soup + Selenium) | `scripts/scraping/` → `data/raw/*.csv` e `*.txt` (Selenium só para a CETESB, ver Charter §6) |
| Nuvem de palavras e estatísticas | Página **Notícias e palavras** |
| API própria (FastAPI) | `api/main.py`: `GET /cidades`, `GET /iqar/tabela`, `GET /iqar/calcular`, `GET`+`POST /leituras` |
| Preparação para LLM | `src/llm_prep.py` → `data/processed/corpus_llm.jsonl`; visão geral na página **API & IA** |
| LLM local + FastAPI (TP4) | `src/llm.py` + `POST /processar_texto` (resumo, sentimento, pergunta); escolha do modelo em `docs/model/ModelSelection.md` |
| Cache | `@st.cache_data` em `src/state.py` (API OpenWeather, leitura de arquivos, nuvem de palavras) |
| Estado de sessão | Histórico de consultas, última avaliação, dados enviados, arquivos já importados |
| Upload/Download | Página **Meus dados**: modelo CSV, upload múltiplo validado, download com IQAr calculado |
