# Data Summary Report: Éter

> Após rodar os scrapers, atualize os campos entre colchetes com os números reais (quantidade de
> linhas, datas de coleta), consultando `data/raw/log_coleta_iqar.csv` e `noticias_qualidade_ar.csv`.

## 1. Visão geral das fontes

| # | Fonte | Tipo de acesso | Arquivo gerado | Atualização |
|---|---|---|---|---|
| 1 | OpenWeather Air Pollution API | API REST (JSON) | não persistido; cache em memória por 10 min | Horária |
| 2 | CETESB: padrões e IQAr | Web scraping (Selenium + Beautiful Soup — página renderizada via JS, ver Charter §6) | `data/raw/tabelas_web/cetesb_tabela_*.csv`, `referencia_iqar_texto.txt` | Sob demanda |
| 3 | Wikipédia: Índice de qualidade do ar | Web scraping (Beautiful Soup) | `data/raw/tabelas_web/wikipedia_tabela_*.csv`, `referencia_iqar_texto.txt` | Sob demanda |
| 4 | Google Notícias (RSS) | Web scraping (Beautiful Soup, parser XML) | `data/raw/noticias_qualidade_ar.csv`, `noticias_texto.txt` | Sob demanda |
| 5 | Resolução CONAMA 491/2018 + faixas CETESB | Referência normativa consolidada | `data/raw/tabela_iqar.csv` | Estática |
| 6 | Usuário (app) | Upload de CSV na aplicação | mantido em `st.session_state`; download opcional | Por sessão |
| 7 | Usuário/sistemas externos (API) | `POST /leituras` (FastAPI) | `data/processed/leituras_api.csv` (não versionado) | Sob demanda |
| 8 | Corpus consolidado para LLM | Reaproveita as fontes 2–4 (`src/llm_prep.py`) | `data/processed/corpus_llm.jsonl` (não versionado) | Sob demanda |

## 2. Dicionário de dados

### 2.1 OpenWeather (resposta de `/data/2.5/air_pollution`)

| Campo | Descrição | Unidade | Transformação no app |
|---|---|---|---|
| `components.pm2_5` | Material particulado fino | µg/m³ | renomeado para `MP2.5` |
| `components.pm10` | Material particulado inalável | µg/m³ | renomeado para `MP10` |
| `components.o3` | Ozônio | µg/m³ | `O3` |
| `components.no2` | Dióxido de nitrogênio | µg/m³ | `NO2` |
| `components.so2` | Dióxido de enxofre | µg/m³ | `SO2` |
| `components.co` | Monóxido de carbono | µg/m³ | convertido para ppm (÷ 1145) |
| `dt` | Horário da medição | Unix timestamp | formatado dd/mm/aaaa hh:mm |

### 2.2 `tabela_iqar.csv` (30 linhas: 6 poluentes × 5 categorias)

| Coluna | Tipo | Descrição |
|---|---|---|
| `poluente` | texto | Sigla (MP10, MP2.5, O3, CO, NO2, SO2) |
| `nome` | texto | Nome por extenso |
| `unidade` | texto | µg/m³ ou ppm |
| `categoria` | texto | Boa, Moderada, Ruim, Muito Ruim, Péssima |
| `indice_min`, `indice_max` | inteiro | Faixa do IQAr |
| `conc_min`, `conc_max` | decimal | Faixa de concentração (vazio = sem limite superior) |
| `recomendacao` | texto | Mensagem de saúde para a categoria |

### 2.3 `noticias_qualidade_ar.csv` ([N] linhas, coletado em [data])

| Coluna | Tipo | Descrição |
|---|---|---|
| `titulo` | texto | Manchete (sem o sufixo com o nome do veículo) |
| `fonte` | texto | Veículo de imprensa |
| `data_publicacao` | data/hora | Data do item no feed |
| `link` | URL | Endereço da notícia |
| `resumo` | texto | Descrição do feed, sem HTML |
| `termo_busca` | texto | Termo que trouxe a notícia |
| `coletado_em` | data/hora | Momento da coleta |

### 2.4 Upload do usuário

| Coluna | Obrigatória | Regras de validação |
|---|---|---|
| `data` | sim | convertida com dia primeiro; linhas inválidas são descartadas |
| `cidade` | sim | texto livre |
| `poluente` | sim | um de MP10, MP2.5, O3, CO, NO2, SO2 (aceita PM2.5 e minúsculas) |
| `valor` | sim | numérico, na unidade da tabela de referência |

Na saída, o app acrescenta `iqar` e `classificacao`.

### 2.5 `leituras_api.csv` (gerado por `POST /leituras`)

| Coluna | Tipo | Descrição |
|---|---|---|
| `cidade`, `poluente`, `valor`, `data`, `origem` | — | mesmos campos enviados no corpo do POST |
| `recebido_em` | data/hora | quando a API recebeu a leitura |
| `iqar`, `categoria` | inteiro, texto | calculados pela própria API no momento do recebimento |

### 2.6 `corpus_llm.jsonl` (gerado por `src/llm_prep.py` / `scripts/prepare_corpus.py`)

Um documento por linha, em JSON, com `id`, `tipo` (`noticia` | `faixa_oficial` | `referencia`),
`texto`, `fonte`, `data`, `link` e `metadados` (campos extras conforme o tipo, ex.: `poluente` e
`categoria` para `faixa_oficial`). Pensado para a próxima entrega (técnicas de LLM), não é
consumido pelo app nem pela API nesta etapa.

## 3. Qualidade dos dados

| Aspecto | Observação | Tratamento |
|---|---|---|
| Representatividade temporal | Limites oficiais usam médias de 1 h, 8 h ou 24 h; a API fornece valores horários | Índice apresentado como aproximação |
| Cobertura espacial | A OpenWeather estima valores por modelo, não por estação física | Mencionado na aba Sobre |
| Duplicidade de notícias | O mesmo título aparece em mais de um termo de busca | `drop_duplicates` por título |
| Ruído textual | Nomes de veículos e stopwords dominam a nuvem | Lista de stopwords em português e remoção do sufixo da fonte |
| Estrutura HTML instável | Páginas podem mudar sem aviso | Extração genérica de `<table>` e log de coleta com status |
| Dados do usuário | Colunas, datas e valores podem vir errados | Validação com mensagens e descarte de linhas inválidas |

## 4. Estatísticas iniciais

| Indicador | Valor |
|---|---|
| Notícias coletadas | [N] |
| Veículos distintos | [N] |
| Período das notícias | [data inicial] a [data final] |
| Tabelas HTML extraídas | [N] |
| Palavras mais frequentes | [top 5, ver aba Notícias e palavras] |
| Documentos no corpus para LLM | [N] (ver `python scripts/prepare_corpus.py`, que imprime a contagem por tipo) |

## 5. Considerações éticas e legais

Os scrapers se identificam no cabeçalho User-Agent, fazem poucas requisições e coletam apenas
conteúdo público. As notícias são exibidas com título, veículo e link para o original, sem
reprodução do texto integral. A chave de API fica fora do repositório. Os dados enviados pelo
usuário **pela interface Streamlit** (upload de CSV) permanecem apenas na sessão do navegador e
não são gravados no servidor. Já os dados enviados **pela API** (`POST /leituras`) são, por
natureza, gravados em `data/processed/leituras_api.csv` no servidor onde a API roda — esse arquivo
fica fora do controle de versão (`.gitignore`) e, numa implantação real, mereceria autenticação e
uma política de retenção, o que está fora do escopo acadêmico desta entrega.
