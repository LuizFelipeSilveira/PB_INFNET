# Data Summary Report: Éter

Este relatório descreve os dados do projeto: de onde vêm, como foram coletados, como são
transformados e **onde cada um é usado** no ciclo de ciência de dados, da coleta até os insights
gerados por IA.

## 1. Visão geral das fontes

| # | Fonte | Tipo de acesso | Arquivo gerado | Atualização |
|---|---|---|---|---|
| 1 | OpenWeather Air Pollution API | API REST (JSON) | não persistido; cache em memória por 10 min | Horária |
| 2 | CETESB: padrões e IQAr | Web scraping (Selenium + Beautiful Soup, ver Charter §7); só o texto é aproveitado | `referencia_iqar_texto.txt` | Sob demanda |
| 3 | Wikipédia: Índice de qualidade do ar | Web scraping (Beautiful Soup) | `data/raw/tabelas_web/wikipedia_tabela_*.csv`, `referencia_iqar_texto.txt` | Sob demanda |
| 4 | Google Notícias (RSS) | Web scraping (Beautiful Soup, parser XML) | `data/raw/noticias_qualidade_ar.csv`, `noticias_texto.txt` | Sob demanda |
| 5 | MMA: Orientação técnica IQAr (jan/2025), CONAMA 506/2024 | Scraping de PDF: link encontrado com Beautiful Soup, Quadro 1 extraído com pdfplumber e validado; reservas: cópia em `data/external/` e tabela no código | `data/raw/tabela_iqar.csv` | Sob demanda |
| 6 | Usuário (app) | Upload de CSV na aplicação | mantido em `st.session_state`; download opcional | Por sessão |
| 7 | Usuário/sistemas externos (API) | `POST /leituras` (FastAPI) | `data/processed/leituras_api.csv` (não versionado) | Sob demanda |
| 8 | Corpus consolidado para LLM | Reaproveita as fontes 2 a 5 (`src/llm_prep.py`) | `data/processed/corpus_llm.jsonl` (não versionado) | Sob demanda |
| 9 | Saídas do LLM local | Geradas pelo modelo Qwen3.5-0.8B (`src/llm.py`, `src/sumarizacao.py`) | respostas da API; `data/processed/insights/` (sentimento e boletins, versionado); `data/processed/avaliacao_llm/` (avaliação, não versionado) | Sob demanda |

## 2. Integração e uso dos dados ao longo do projeto

Cada fonte entrou no projeto numa etapa e passou a alimentar outras partes depois. A tabela mostra
o caminho de cada dado até o usuário.

| Dado | Processamento | Onde é usado | Desde |
|---|---|---|---|
| Concentrações da OpenWeather | Renomeação dos campos, CO de µg/m³ para ppm, cache de 10 min; modo simulado se a API falhar | Página **Monitor** (IQAr atual, histórico, comparação entre cidades, recomendação por perfil) | TP1/TP2 |
| Faixas oficiais do IQAr (PDF do MMA) | Extração do Quadro 1, validação (6 poluentes × 5 faixas contínuas), gravação com a coluna `fonte` | Cálculo do IQAr em todo o projeto (`src/iqar.py`); página **Tabela IQAr**; rotas `/iqar/*`; contexto das perguntas ao LLM; documentos `faixa_oficial` do corpus | TP3 (tabela no código); TP4 (extraída do PDF) |
| Notícias | Limpeza de HTML, remoção do sufixo do veículo, deduplicação por título, contagem de palavras sem *stopwords* | Página **Notícias e palavras** (lista, nuvem de palavras, estatísticas); documentos `noticia` do corpus; textos de entrada do LLM na aba **LLM local** | TP3 |
| Texto de referência (CETESB, Wikipédia) | Extração do texto visível, quebra em blocos de ~600 caracteres | Nuvem de palavras; documentos `referencia` do corpus | TP3 |
| Medições do usuário (upload) | Validação de colunas, datas, poluentes e valores; cálculo do IQAr por linha | Página **Meus dados** (gráficos e download) | TP3 |
| Leituras enviadas por API | Validação do corpo, normalização do poluente, cálculo do IQAr | `GET/POST /leituras` | TP3 |
| Corpus para LLM | União padronizada das fontes acima, em JSON Lines | Referência de textos para LLM | TP3 |
| Notícias enriquecidas e boletins | Remoção das páginas de índice do IQAir, local e mês detectados, sentimento por LLM, resumos *map-reduce* por recorte, checagem de fidelidade | `POST /sumarizar_noticias`, `GET /noticias/recortes`; dashboard final | TP5 |
| Saídas do LLM | Validação por esquema (*response models*); rótulo de sentimento padronizado | Rota `POST /processar_texto`; aba **LLM local**; dashboard final (TP5) | TP4 |

### Do dado ao insight

1. **Coleta:** API, scraping de páginas, de feed e de PDF, upload e API própria.
2. **Limpeza e validação:** cada fonte tem regras próprias (seção 4). Nada chega ao app sem passar
   por elas, e o que falha é registrado em log ou devolvido com mensagem clara.
3. **Transformação:** o cálculo oficial do IQAr transforma concentrações em índice, categoria e
   recomendação; o corpus padroniza os textos num só formato.
4. **Modelagem com IA:** o LLM local classifica o sentimento das notícias, responde perguntas
   usando a tabela oficial e resume textos. Na TP5, resume o noticiário por recorte.
5. **Apresentação:** o app e o dashboard mostram os números, os resumos e as fontes lado a lado,
   para que cada insight possa ser conferido.

## 3. Dicionário de dados

### 3.1 OpenWeather (resposta de `/data/2.5/air_pollution`)

| Campo | Descrição | Unidade | Transformação no app |
|---|---|---|---|
| `components.pm2_5` | Material particulado fino | µg/m³ | renomeado para `MP2.5` |
| `components.pm10` | Material particulado inalável | µg/m³ | renomeado para `MP10` |
| `components.o3` | Ozônio | µg/m³ | `O3` |
| `components.no2` | Dióxido de nitrogênio | µg/m³ | `NO2` |
| `components.so2` | Dióxido de enxofre | µg/m³ | `SO2` |
| `components.co` | Monóxido de carbono | µg/m³ | convertido para ppm (÷ 1145) |
| `dt` | Horário da medição | Unix timestamp | formatado dd/mm/aaaa hh:mm |

### 3.2 `tabela_iqar.csv` (30 linhas: 6 poluentes × 5 categorias)

| Coluna | Tipo | Descrição |
|---|---|---|
| `poluente` | texto | Sigla (MP10, MP2.5, O3, CO, NO2, SO2) |
| `nome` | texto | Nome por extenso |
| `unidade` | texto | µg/m³ ou ppm |
| `categoria` | texto | Boa, Moderada, Ruim, Muito Ruim, Péssima |
| `indice_min`, `indice_max` | inteiro | Faixa do IQAr |
| `conc_min`, `conc_max` | decimal | Faixa de concentração (na última faixa, o teto técnico do Quadro 1) |
| `recomendacao` | texto | Mensagem de saúde para a categoria |
| `fonte` | texto | Origem: PDF do MMA (online ou cópia local) ou reserva no código |

### 3.3 `noticias_qualidade_ar.csv` (281 linhas, coletado em 19/10/2026)

| Coluna | Tipo | Descrição |
|---|---|---|
| `titulo` | texto | Manchete (sem o sufixo com o nome do veículo) |
| `fonte` | texto | Veículo de imprensa |
| `data_publicacao` | data/hora | Data do item no feed |
| `link` | URL | Endereço da notícia |
| `resumo` | texto | Descrição do feed, sem HTML |
| `termo_busca` | texto | Termo que trouxe a notícia |
| `coletado_em` | data/hora | Momento da coleta |

### 3.4 Upload do usuário

| Coluna | Obrigatória | Regras de validação |
|---|---|---|
| `data` | sim | convertida com dia primeiro; linhas inválidas são descartadas |
| `cidade` | sim | texto livre |
| `poluente` | sim | um de MP10, MP2.5, O3, CO, NO2, SO2 (aceita PM2.5 e minúsculas) |
| `valor` | sim | numérico, na unidade da tabela de referência |

Na saída, o app acrescenta `iqar` e `classificacao`.

### 3.5 `leituras_api.csv` (gerado por `POST /leituras`)

| Coluna | Tipo | Descrição |
|---|---|---|
| `cidade`, `poluente`, `valor`, `data`, `origem` | — | mesmos campos enviados no corpo do POST |
| `recebido_em` | data/hora | quando a API recebeu a leitura |
| `iqar`, `categoria` | inteiro, texto | calculados pela própria API no momento do recebimento |

### 3.6 `corpus_llm.jsonl` (gerado por `src/llm_prep.py` / `scripts/prepare_corpus.py`)

Um documento por linha, em JSON, com `id`, `tipo` (`noticia` | `faixa_oficial` | `referencia`),
`texto`, `fonte`, `data`, `link` e `metadados` (campos extras conforme o tipo, ex.: `poluente` e
`categoria` para `faixa_oficial`). Reúne todos os textos do projeto num formato único para uso com LLMs. A sumarização da TP5
lê as notícias direto do CSV, porque precisa das colunas de data, veículo e termo de busca.

### 3.7 Saídas do LLM (`POST /processar_texto`)

| Campo | Descrição |
|---|---|
| `tarefa` | `resumo`, `sentimento` ou `pergunta` |
| `texto_entrada` | Texto processado |
| `analise` | Resultado estruturado conforme `analise.tipo`: resumo com contagem de palavras e taxa de compressão; sentimento com rótulo padronizado e resposta bruta; pergunta com resposta, poluentes consultados e fonte da tabela |
| `metadados` | Modelo, formato (ex.: `fp16-gpu`), tokens gerados, tempo de geração e horário |

### 3.8 Avaliação do LLM (`data/processed/avaliacao_llm/`)

Gerada por `scripts/llm/avaliar_modelos.py`: `resumo.csv` (uma linha por modelo, com carga,
tokens/s, memória e acertos) e `<modelo>.json` (cada *prompt*, resposta e acerto). Os resultados
das quatro rodadas estão em `docs/model/ModelSelection.md`.

### 3.9 Insights do noticiário (`data/processed/insights/`, TP5)

Gerados por `scripts/llm/sumarizar_noticias.py` e versionados, para o dashboard funcionar onde o
LLM não está instalado.

| Arquivo | Conteúdo |
|---|---|
| `noticias_enriquecidas.csv` | notícias válidas (sem páginas de índice e duplicatas) com `mes`, `local` detectado e `sentimento` classificado pelo LLM |
| `resumos.json` | um boletim por recorte padrão (geral, por local, por mês, por tema): resumo, pontos principais, recomendação, `formato_ok`, fidelidade, estatísticas, processamento e lista de fontes |

## 4. Qualidade dos dados

| Aspecto | Observação | Tratamento |
|---|---|---|
| Representatividade temporal | Limites oficiais usam médias de 1 h, 8 h ou 24 h; a API fornece valores horários | Índice apresentado como aproximação |
| Cobertura espacial | A OpenWeather estima valores por modelo, não por estação física | Mencionado na aba Sobre |
| Norma vigente | A CONAMA 506/2024 mudou faixas de MP10, MP2.5 e SO2 | Tabela extraída do documento oficial e validada; teste com o exemplo resolvido da norma |
| Duplicidade de notícias | O mesmo título aparece em mais de um termo de busca | `drop_duplicates` por título |
| Ruído textual | Nomes de veículos e *stopwords* dominam a nuvem | Lista de *stopwords* em português e remoção do sufixo da fonte |
| Campo `resumo` das notícias | O feed do Google repete a manchete e junta o nome do veículo ou manchetes de outras notícias | A sumarização usa só a manchete |
| Páginas que não são notícias | 48 itens do IQAir são páginas fixas "Qualidade do ar em <cidade>" | Removidos antes da sumarização (281 → 233 válidas) |
| Concentração de fontes | Poucos veículos e poucas cidades respondem por boa parte das notícias (seção 5) | Os resumos e o dashboard mostram veículo e quantidade de notícias de cada recorte |
| Estrutura HTML instável | Páginas podem mudar sem aviso | Extração genérica de `<table>`, validação e log de coleta com status |
| Dados do usuário | Colunas, datas e valores podem vir errados | Validação com mensagens e descarte de linhas inválidas |
| Saídas do LLM | O modelo pode responder fora do formato ou errar | *Prompts* avaliados, validação por esquema (erro 502 se fora do contrato), rótulo inválido sinalizado |

## 5. Estatísticas

| Indicador | Valor |
|---|---|
| Notícias coletadas | 281 (únicas por título; 91 de "qualidade do ar", 82 de "poluição do ar", 71 de "queimadas fumaça", 37 de "índice de qualidade do ar") |
| Veículos distintos | 103 (mais frequentes: IQAir 52, G1 46, Amazonas Atual 14) |
| Período das notícias | 12/01/2026 a 19/10/2026 |
| Palavras mais frequentes | qualidade (279), manaus (273), fumaça (214), queimadas (159), poluição (129) |
| Tabelas extraídas | MMA: 1 (Quadro 1 do PDF, validado); Wikipédia: 1 (caixa de navegação, sem faixas do IQAr); CETESB: 0 |
| Última coleta da tabela oficial | 16/11/2026, PDF baixado do site do MMA (`log_coleta_iqar.csv`) |
| Documentos no corpus para LLM | 322: 281 notícias, 30 faixas oficiais, 11 blocos de referência |
| Avaliação do LLM escolhido (Qwen3.5-0.8B, GPU GTX 1050 Ti) | sentimento 92% (11 de 12); perguntas sobre a tabela 100% (3 de 3); 11,5 tokens/s; 1,46 GB de VRAM |

As palavras mais frequentes já indicam um padrão que a sumarização deve tornar explícito: o
noticiário do período é dominado por **Manaus** e por **queimadas e fumaça**.

## 6. Considerações éticas e legais

Os scrapers se identificam no cabeçalho User-Agent, fazem poucas requisições e coletam apenas
conteúdo público. As notícias são exibidas com título, veículo e link para o original, sem
reprodução do texto integral; no dashboard final (TP5), cada resumo gerado pelo LLM aparece
ao lado das notícias de origem. A chave de
API fica fora do repositório.

Os dados enviados pelo usuário **pela interface Streamlit** (upload de CSV) permanecem apenas na
sessão do navegador e não são gravados no servidor. Os dados enviados **pela API**
(`POST /leituras`) são gravados em `data/processed/leituras_api.csv` no servidor onde a API roda;
esse arquivo fica fora do controle de versão e, numa implantação real, mereceria autenticação e uma
política de retenção, o que está fora do escopo acadêmico.

O LLM roda **localmente**: nenhum texto é enviado a serviços de terceiros. As respostas geradas por
IA são identificadas como tais, mostram o modelo usado e não substituem boletins oficiais nem
orientação médica.
