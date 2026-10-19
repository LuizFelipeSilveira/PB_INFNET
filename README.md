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
│   │                              #        GET+POST /leituras)
│   └── schemas.py                #   modelos Pydantic de entrada/saída
├── src/                          # Código reutilizável (usado pelo app E pela API)
│   ├── iqar.py                   #   cálculo do IQAr e recomendações
│   ├── api.py                    #   OpenWeather + modo simulado
│   ├── data_loader.py            #   leitura de dados e validação de uploads
│   ├── state.py                  #   sessão e cache compartilhados entre páginas
│   ├── ui.py                     #   componentes visuais compartilhados
│   └── llm_prep.py               #   Modeling: consolida o corpus para a próxima entrega (LLM)
├── scripts/
│   ├── scraping/                 # Data Acquisition: coletores com Beautiful Soup (+ Selenium
│   │   ├── scrape_iqar.py        #   para a página da CETESB — ver Charter §6)
│   │   └── scrape_noticias.py
│   └── prepare_corpus.py         # gera data/processed/corpus_llm.jsonl
├── data/
│   ├── raw/                      #   saídas dos scrapers (CSV/TXT)
│   ├── processed/                #   corpus_llm.jsonl, leituras_api.csv (não versionados)
│   └── external/                 #   dados de terceiros
├── notebooks/                    # Data Understanding: análises exploratórias
├── docs/
│   ├── project/Charter.md        # Business Understanding
│   └── data_report/DataSummaryReport.md
├── .streamlit/                   # tema e segredos (exemplo)
└── requirements.txt
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

## Controle de versão com Git

```bash
git init
git add .
git commit -m "feat: estrutura TDSP, scrapers e interface interativa"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/eter.git
git push -u origin main
```

Sugestão de fluxo: uma branch por funcionalidade (`git checkout -b feat/upload-csv`), commits
pequenos com prefixos (`feat:`, `fix:`, `docs:`, `data:`) e merge na `main` via pull request.
O `.gitignore` já impede que `secrets.toml` seja enviado.

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
| Preparação para LLM (próxima entrega) | `src/llm_prep.py` → `data/processed/corpus_llm.jsonl`; visão geral na página **API & IA** |
| Cache | `@st.cache_data` em `src/state.py` (API OpenWeather, leitura de arquivos, nuvem de palavras) |
| Estado de sessão | Histórico de consultas, última avaliação, dados enviados, arquivos já importados |
| Upload/Download | Página **Meus dados**: modelo CSV, upload múltiplo validado, download com IQAr calculado |
