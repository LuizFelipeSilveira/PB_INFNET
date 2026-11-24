# Exit Report: Éter - Monitor Inteligente de Qualidade do Ar

> Relatório de encerramento do TDSP (fase *Customer Acceptance*), entrega final TP5.
> Repositório: https://github.com/LuizFelipeSilveira/PB_INFNET

## 1. Problema e resultado

**Problema de negócio:** a informação sobre qualidade do ar está espalhada (números em APIs, faixas
oficiais em resoluções em PDF, contexto em centenas de notícias) e é técnica demais para a maioria
das pessoas, sobretudo para grupos sensíveis.

**Pergunta:** como transformar esses dados públicos e dispersos em orientação simples, confiável e
acionável?

**Resultado:** um app Streamlit cuja página inicial, **"Do dado ao insight"**, mostra o ciclo
completo: as notícias coletadas e limpas, o sentimento de cada uma classificado por um LLM local,
boletins resumidos por local, mês e tema com a orientação oficial de saúde e as fontes para
conferir, e a geração de novos boletins em tempo real. O app calcula o IQAr pela norma vigente
(CONAMA 506/2024) e expõe tudo também por uma API FastAPI.

## 2. Ciclo de ciência de dados

| Etapa TDSP | O que foi feito | Evidência |
|---|---|---|
| Business Understanding | Problema, pergunta de negócio, métricas, ODS e ESG | `docs/project/Charter.md` |
| Data Acquisition & Understanding | OpenWeather (API), PDF do MMA (Quadro 1 extraído com pdfplumber e validado), 281 notícias (RSS), textos da CETESB e da Wikipédia, upload e API de leituras | `docs/data_report/DataSummaryReport.md`, `scripts/scraping/` |
| Limpeza | 50 páginas de índice do IQAir removidas (281 → 231 notícias válidas); local e mês detectados; tabela oficial validada | `src/sumarizacao.py`, `src/tabela_oficial.py` |
| Modeling | IQAr pela norma (reproduz o exemplo oficial: 168, 86, 60); LLM local escolhido e avaliado em 4 rodadas; sumarização avaliada em 4 rodadas | `docs/model/ModelSelection.md`, `docs/model/Sumarizacao.md` |
| Deployment | App multipage com dashboard final; API com 10 rotas, *response models* e erros padronizados | `pages/dashboard.py`, `api/` |
| Customer Acceptance | 87 testes automatizados; validação com dados reais; este relatório | `tests/`, `docs/project/ExitReport.md` |

## 3. Funcionalidade de IA escolhida: sumarização

O noticiário é a fonte que mais pede síntese (231 manchetes válidas) e a sumarização é a tarefa mais
confiável para um modelo pequeno, porque cada resumo parte de um contexto fornecido no *prompt*.

**Como funciona:** cadeia de *prompts* curtos, uma tarefa por chamada (fatos por lote de 20
manchetes → consolidação em árvore até 3 pontos → resumo de duas frases). Números e a orientação de
saúde vêm do código, não do LLM: a orientação é o texto oficial da categoria do IQAr mais grave
citada nas manchetes.

**Avaliação em 4 rodadas** (9 recortes cada):

| Rodada | Abordagem | Formato correto | Fidelidade 100% | Orientação coerente |
|---|---|---|---|---|
| 1 | um *prompt* com tudo; recomendação em texto livre | 9/9 | 8/9 | texto inadequado em 3 |
| 2 | escolha da orientação por número | 7/9 | 8/9 | 6/7 |
| 3 | consolidação em árvore; escolha por nome | 7/9 | 8/9 | 4/7 |
| 4 | cadeia de *prompts* de uma tarefa; orientação por código | **9/9** | **9/9** | **9/9** |

## 4. Insights gerados com dados reais (coleta de 19/10/2026)

- **88%** das notícias classificadas pela IA relatam piora do ar ou danos à saúde.
- **Manaus** aparece em **56%** das notícias válidas; o pico foi em **17/10/2026**, com 62
  notícias num só dia.
- Quando as manchetes citam uma categoria do IQAr, a mais citada é **Péssima** (24 manchetes).
- Boletim de Manaus: suspensão de aulas em escolas e universidades por causa da fumaça de
  queimadas; orientação oficial "Péssima" (citada em 23 manchetes): risco sério à saúde de toda a
  população, evitar sair de casa e qualquer esforço físico ao ar livre.

## 5. Impacto em ESG e ODS

| Dimensão | Contribuição |
|---|---|
| ODS 3.9 | Orientação de saúde por perfil e boletins com a orientação oficial |
| ODS 11.6 | Monitoramento da qualidade do ar em capitais |
| ODS 13 | Torna visíveis os episódios de queimadas que dominam o noticiário |
| ODS 16.10 | Norma oficial extraída do documento do governo; cada dado com a sua fonte |
| ESG: Ambiental | Monitorar e comunicar a qualidade do ar |
| ESG: Social | Proteger grupos sensíveis com linguagem simples |
| ESG: Governança | Fontes rastreáveis, logs de coleta, avaliação documentada da IA, decisões de saúde fora do LLM |

## 6. Uso eficiente e justificado do modelo

- **Modelo:** Qwen3.5-0.8B (Hugging Face, Apache 2.0), local, via Transformers, em FP16 numa GTX
  1050 Ti (4 GB): 1,46 GB de VRAM e 11,5 tokens/s.
- **Por que local:** custo zero por consulta, nenhum texto sai da máquina, resultados reproduzíveis
  (decodificação determinística), funciona offline.
- **Por que este tamanho:** empatou com o de 2B nas tarefas com gabarito e foi 3,7 vezes mais rápido.
- **Eficiência:** boletins pré-gerados ficam versionados (o dashboard funciona sem o modelo); só a
  geração sob demanda chama o LLM.

## 7. Limitações e próximos passos

- O índice usa leituras horárias, não as médias oficiais de 1 h, 8 h e 24 h; é uma aproximação.
- A sumarização ainda tem exageros pontuais ("pescadores perdem vidas"); por isso o dashboard mostra
  as manchetes de origem ao lado de cada boletim.
- O noticiário é concentrado em Manaus e em outubro/2026; novas coletas ampliariam a cobertura.
- Próximos passos: coleta agendada das notícias, testes com usuários e um modelo maior quando houver
  hardware disponível.
