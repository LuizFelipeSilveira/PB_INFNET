# Project Charter: Éter - Monitor Inteligente de Qualidade do Ar

> Preencha os campos entre colchetes com os dados da sua equipe.

## 1. Entendimento do negócio

### Contexto
A poluição do ar está entre os principais fatores ambientais de risco à saúde. No Brasil, episódios
de queimadas, tráfego intenso e atividade industrial elevam a concentração de poluentes, mas os
boletins oficiais usam índices técnicos pouco acessíveis ao público e cobrem poucas cidades.

### Problema
Muitas pessoas não sabem avaliar se o ar da sua cidade está adequado para atividades externas e
correm riscos desnecessários em dias de alta poluição, sobretudo grupos sensíveis (crianças,
idosos, pessoas com asma, DPOC ou doenças cardíacas).

### Alinhamento com a Agenda 2030
O projeto contribui para o ODS 3 (Saúde e Bem-Estar), meta 3.9, que trata da redução de doenças
causadas por poluição do ar, e para o ODS 11 (Cidades Sustentáveis), meta 11.6, que trata da
qualidade do ar nas cidades.

## 2. Escopo

### Dentro do escopo
O projeto entrega uma aplicação web em Streamlit, organizada em múltiplas páginas com menu de
navegação, que consulta dados horários de poluentes (MP2.5, MP10, O3, NO2, SO2 e CO) para
capitais brasileiras, calcula o IQAr segundo a metodologia CETESB / CONAMA 491/2018 e apresenta
recomendações de saúde adaptadas ao perfil do usuário. A aplicação também exibe notícias recentes
sobre qualidade do ar coletadas por web scraping, com nuvem de palavras e estatísticas, e permite
que o usuário envie e baixe medições próprias em CSV.

Além da interface, uma API em FastAPI (`api/`) expõe os mesmos dados e regras de negócio (tabela
oficial, cálculo do IQAr e registro de novas leituras) para outros sistemas, independente da
interface Streamlit. Os dados coletados também são consolidados em um corpus estruturado
(`src/llm_prep.py`) que prepara o terreno — sem ainda usar nenhum modelo — para as técnicas de LLM
(resumo, classificação de sentimento, Q&A e geração de texto) da próxima entrega.

### Fora do escopo
O projeto não substitui boletins oficiais nem orientação médica, não faz previsão de qualidade do
ar em dias futuros nesta etapa e não instala sensores físicos.

## 3. Objetivos e métricas de sucesso

| Objetivo | Métrica | Meta |
|---|---|---|
| Traduzir índices técnicos em linguagem simples | Recomendação exibida para 100% das consultas válidas | 100% |
| Fidelidade ao método oficial | Divergência do IQAr calculado vs. tabela CETESB para valores de teste | 0 pontos |
| Performance | Tempo de resposta de uma consulta repetida (cache) | < 1 s |
| Atualidade do conteúdo | Idade máxima das notícias exibidas | ≤ 7 dias |
| Usabilidade | Tarefas concluídas sem ajuda em teste com [N] usuários | ≥ 80% |

## 4. Stakeholders

| Papel | Quem | Interesse |
|---|---|---|
| Equipe de desenvolvimento | [Nomes e funções] | Construção, dados e deploy |
| Orientador(a) / professor(a) | [Nome] | Avaliação técnica e acadêmica |
| Usuários finais | Moradores de centros urbanos, praticantes de esportes ao ar livre | Decidir se é seguro sair e se exercitar |
| Grupos sensíveis | Crianças, idosos, pessoas com doenças respiratórias ou cardíacas | Alertas mais conservadores |
| Parceiros potenciais | Escolas, secretarias de saúde e meio ambiente, academias | Comunicação de risco à população |
| Provedores de dados | OpenWeather, CETESB, Google Notícias | Uso adequado conforme termos de serviço |

## 5. Metodologia (TDSP)

| Fase | Entregas | Status |
|---|---|---|
| Business Understanding | Este Charter, definição do problema e métricas | Concluído |
| Data Acquisition & Understanding | Scrapers (tabela IQAr, notícias), integração com API, Data Summary Report | Concluído |
| Modeling | Cálculo do IQAr e regras de recomendação; preparação do corpus para LLM (`src/llm_prep.py`) | Em andamento |
| Deployment | App Streamlit multipage, API em FastAPI (`api/`); deploy do app no Streamlit Cloud | Em andamento |
| Customer Acceptance | Testes com usuários e ajustes de interface | Planejado |
| TP4 (próxima entrega) | Aplicação das técnicas de LLM sobre o corpus já preparado (resumo, sentimento, Q&A, geração de texto) | Planejado |

## 6. Arquitetura

Os scrapers em `scripts/scraping/` rodam separadamente e gravam CSV/TXT em `data/raw/`. O app
Streamlit (`app.py` + `pages/`) lê esses arquivos com cache (`src/state.py`), consulta a API
OpenWeather (cache de 10 minutos), calcula o IQAr com o módulo `src/iqar.py` e mantém histórico e
uploads no `st.session_state`. Uma API HTTP independente (`api/`, FastAPI) reaproveita os mesmos
módulos de `src/` para expor a tabela oficial, o cálculo do IQAr e o registro de novas leituras a
outros sistemas — ela roda em outro processo (`uvicorn api.main:app`) e não depende do Streamlit
estar no ar. `src/llm_prep.py` consolida notícias, tabela oficial e texto de referência em um
corpus (`data/processed/corpus_llm.jsonl`) que será a entrada das técnicas de LLM da próxima etapa.

### Decisão sobre scraping dinâmico (Selenium)
O log de coleta (`data/raw/log_coleta_iqar.csv`) mostrou que a página da CETESB retornava **0
tabelas** extraídas só com `requests` + BeautifulSoup — sinal de conteúdo montado no navegador.
Em vez de adotar Selenium para todas as fontes (o que contraria a recomendação de manter a
simplicidade quando possível), `scripts/scraping/scrape_iqar.py` passou a renderizar só essa fonte
com um Chrome headless (Selenium + `webdriver-manager`), listada em `FONTES_DINAMICAS`; a
Wikipédia continua sendo lida da forma simples, com `requests`. Se o Chrome/driver não estiver
disponível no ambiente, o script cai de volta para `requests` automaticamente.

## 7. Cronograma

| Etapa | Período |
|---|---|
| Proposta e protótipo inicial | [datas] |
| Ambiente, scraping, interface dinâmica, cache e upload | [datas] |
| Modelo de previsão / IA generativa para recomendações | [datas] |
| Testes com usuários e entrega final | [datas] |

## 8. Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| API fora do ar ou limite de requisições excedido | Sem dados em tempo real | Cache com TTL e modo simulado automático |
| Mudança no HTML das páginas coletadas | Scraper quebra | Extração genérica de tabelas, log de coleta e tabela de referência consolidada |
| Leitura horária difere das médias oficiais | Índice impreciso | Aviso explícito de aproximação na interface |
| Upload com formato incorreto | Erros na aplicação | Validação de colunas, tipos e poluentes com mensagens claras |
| Exposição da chave de API | Uso indevido | `secrets.toml` no `.gitignore` e Secrets do Streamlit Cloud |
