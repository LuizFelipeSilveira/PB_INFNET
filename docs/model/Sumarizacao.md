# Sumarização Automática do Noticiário: Éter

> Fase *Modeling* do TDSP, entrega TP5 (Tarefa 2, opção 2). Este documento descreve como o
> projeto transforma o noticiário coletado por web scraping em boletins curtos e acionáveis
> com o LLM local, as decisões de engenharia de *prompts* e como a qualidade é medida.
> O modelo usado é o escolhido em `docs/model/ModelSelection.md` (Qwen3.5-0.8B, local).

## 1. Problema

O scraper do Google Notícias trouxe 281 itens sobre qualidade do ar. Ninguém lê 281 manchetes
para decidir se é seguro sair de casa, e a contagem de palavras da TP3 só mostra *quais* termos
aparecem, não *o que aconteceu*. A sumarização responde, para qualquer recorte (local, mês,
tema): o que foi noticiado, quais os fatos principais e o que a população deve fazer.

## 2. Preparação dos dados (`src/sumarizacao.py`)

| Etapa | Por quê | Efeito no conjunto coletado em 19/10/2026 |
|---|---|---|
| Remover páginas de índice do IQAir ("Qualidade do ar em Campinas", "Índice de Qualidade do Ar (AQI) ... \| IQAir Brasil") | São páginas fixas de monitoramento que o feed traz como notícia; não relatam fatos. Títulos iguais vindos de veículos de notícia são mantidos | 281 → 231 notícias válidas (50 removidas) |
| Usar só a manchete como texto | O campo `resumo` do feed repete a manchete e acrescenta o nome do veículo ou manchetes de outras notícias | Entrada mais curta e sem ruído |
| Detectar o local citado | Permite recortar por cidade | Manaus: 130; outros locais: 88; Santarém: 8; Porto Velho: 4 |
| Extrair o mês | Permite recortar por período | outubro/2026: 203; setembro/2026: 12 |

## 3. Estratégia: cadeia de *prompts* curtos (*map-reduce* em árvore)

A lição central das rodadas de avaliação (seção 7) é que o modelo de 0,8B acerta quando **cada
chamada tem uma tarefa simples só**. Pedir numa resposta um resumo, uma lista de pontos e a escolha
de uma orientação funcionou nos recortes pequenos, mas quebrou nos grandes. Por isso a
sumarização é uma cadeia de chamadas curtas:

1. **Fatos (*map*):** as manchetes do recorte são divididas em lotes de até 20. Cada lote vira 3
   fatos.
2. **Consolidação (*reduce* em árvore):** se houver mais de 12 fatos, eles são agrupados de 12 em
   12 e cada grupo vira 3, até sobrarem no máximo 12. Uma última consolidação escolhe os **3
   pontos principais**. Fatos repetidos, ou que só falam da numeração das manchetes, são
   descartados a cada passo.
3. **Resumo:** os fatos finais viram **duas frases** em texto corrido.

Um recorte de 231 notícias passa por 12 chamadas de fatos, 4 de consolidação e 1 de resumo.

**O que não vem do LLM:**
- **Números:** quantidade de notícias, período, veículos, locais e distribuição de sentimento são
  calculados com pandas.
- **Recomendação de saúde:** é o texto oficial (`src/iqar.py`) da **categoria mais grave citada
  nas próprias manchetes** ("péssima", "perigosa", "muito insalubre"...). Vale a categoria citada
  por pelo menos 10% das manchetes (mínimo 2); se nenhuma chegar lá, a mais citada. O boletim
  informa quantas manchetes sustentam a escolha.

## 4. Engenharia de *prompts*

| Decisão | Motivo (evidência) |
|---|---|
| Papel fixo na mensagem de sistema ("analista do Éter", português, fiel ao noticiado) | Mantém idioma e tom em todas as chamadas |
| **Uma tarefa por *prompt*** (fatos, consolidação, resumo) | Rodadas 2 e 3: com tudo numa resposta, os dois maiores recortes saíram fora do formato |
| Manchetes **numeradas** no *prompt* de fatos e regra "não cite a numeração das manchetes" | A numeração separa os itens; a regra evita "fatos" como "Manchetes 3, 4, 5 citam Santarém" (rodada 3) |
| "Use somente as informações acima" | *Grounding*: na TP4, o contexto no *prompt* levou as perguntas a 100% de acerto |
| "Escreva locais e números exatamente como estão" e "repita a palavra que vem junto do número" | Rodada 1: o modelo trocou Amazonas por Manaus e transformou "964% na poluição" em "964% na qualidade do ar" |
| Pedido de exatamente 3 fatos, um por linha, começando com "- " | Saída curta e fácil de separar por código |
| Resumo "só com as duas frases, sem título e sem lista" | O código confere se veio texto corrido (`formato_ok`) |
| Recomendação **fora** do LLM (categoria citada nas manchetes) | Rodadas 1 a 3: texto livre deu "evite sair da cidade"; escolha por número confundiu listas; escolha por nome deu "Moderada" para recortes com mais de 90% de notícias negativas |
| Decodificação determinística (`do_sample=False`) | Mesma entrada, mesmo boletim: resultados reproduzíveis |

Os *prompts* completos estão em `src/sumarizacao.py` (`PROMPT_FATOS`, `PROMPT_CONSOLIDAR` e
`PROMPT_RESUMO`).

## 5. Como a qualidade é medida

| Métrica | Como é calculada | Para que serve |
|---|---|---|
| `formato_ok` | resumo em texto corrido, ao menos 2 pontos e fatos vindos do modelo (sem usar a reserva de manchetes) | mostra se o modelo seguiu as instruções |
| Fidelidade (`taxa`) | fração dos números, nomes próprios e locais conhecidos (capitais e estados, mesmo no início da frase) do texto gerado que aparecem nas manchetes de origem | aponta possíveis invenções (`nao_encontrados`) para conferir |
| `recomendacao_coerente` | orientação branda (Boa/Moderada) com 60% ou mais de notícias negativas, ou grave (Muito Ruim/Péssima) com menos de 30%, é marcada como incoerente | confere a orientação com um dado calculado de outra forma (o sentimento) |
| Taxa de compressão | palavras do boletim / palavras das manchetes | mede o quanto a leitura foi reduzida |
| Revisão manual | leitura dos boletins ao lado das fontes | detecta erros que a checagem automática não pega |

A checagem de fidelidade é simples de propósito: não prova que o resumo está certo, mas é barata,
roda em todo boletim e diz exatamente o que conferir. Ela não pega uma palavra trocada ao lado de
um número correto (ex.: "poluição" por "qualidade do ar"); por isso a revisão manual continua
fazendo parte da avaliação.

## 6. Como rodar

```powershell
# 1. (opcional) atualizar o noticiário
python scripts/scraping/scrape_noticias.py

# 2. gerar sentimento de todas as notícias e os boletins dos recortes padrão
python scripts/llm/sumarizar_noticias.py
```

Saídas em `data/processed/insights/` (versionadas, para o dashboard funcionar sem o LLM):
`noticias_enriquecidas.csv` (notícias limpas com mês, local e sentimento) e `resumos.json` (um
boletim por recorte, com estatísticas, fidelidade, processamento e fontes).

Sob demanda, pela API:

```bash
curl -X POST http://127.0.0.1:8000/sumarizar_noticias -H "Content-Type: application/json" \
     -d '{"local": "Manaus", "mes": "2026-10", "max_noticias": 30}'
```

`GET /noticias/recortes` lista os locais, meses e temas aceitos nos filtros.

## 7. Resultados

### Rodada 1 (23/11/2026, Qwen3.5-0.8B em FP16 na GTX 1050 Ti, 233 notícias)

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade (checagem da rodada) | Compressão |
|---|---|---|---|---|---|---|
| Panorama geral | 233 | 13 | 147,3 | sim | 100% | 2% |
| Local: Manaus | 130 | 8 | 74,6 | sim | 75% ("República") | 3% |
| Local: outros locais | 88 | 6 | 63,1 | sim | 100% | 6% |
| Mês: 09/2026 | 12 | 1 | 6,8 | sim | 100% | 39% |
| Mês: 10/2026 | 203 | 12 | 103,4 | sim | 100% | 3% |
| Tema: qualidade do ar | 91 | 6 | 44,8 | sim | 100% | 6% |
| Tema: queimadas fumaça | 71 | 5 | 35,5 | sim | 100% | 9% |
| Tema: poluição do ar | 59 | 4 | 35,5 | sim | 100% | 9% |
| Tema: índice de qualidade do ar | 12 | 1 | 9,7 | sim | 100% | 55% |

Os 9 boletins seguiram o formato e foram gerados em cerca de 9 minutos no total, sem contar a
classificação de sentimento. O sentimento das 233 notícias válidas ficou em 204 negativas, 23
positivas e 6 neutras.

**Revisão manual** (cada afirmação conferida nas manchetes de origem):

- **Corretos:** "pior qualidade do ar do mundo", "índice de 637", "38 mil focos", "23 vezes acima
  do limite da OMS", "11 suspeitos presos", "Aparecida tem a maior concentração de partículas",
  "Manaus cancela eventos ao ar livre", "governo investiga a origem dos incêndios".
- **Erros encontrados:**

| Recorte | Erro | Manchete de origem | Pego pela checagem automática? |
|---|---|---|---|
| Local: Manaus | "Avenida da República" | "avenida na Zona Norte de Manaus" | sim |
| Local: outros locais | "Manaus tem 10 municípios sob alerta" | "Amazonas tem 10 municípios sob alerta severo" | não (passou a pegar, ver abaixo) |
| Mês: 09/2026 | "964% de aumento na qualidade do ar" | "alta de 964% na poluição do ar" | não |
| Manaus, qualidade do ar, índice | recomendação "evite sair da cidade" | — | não se aplica |

**Correções para a rodada 2:**

1. **Recomendação por escolha fechada:** o modelo escolhe uma das 5 orientações oficiais (responde
   o número), e o boletim mostra o texto da norma. Uma resposta fora da lista vira
   `formato_ok = false`.
2. **Regras novas nos *prompts*:** manter o local exatamente como na manchete e repetir a palavra
   que acompanha cada número.
3. **Checagem de fidelidade ampliada:** passa a conferir capitais e estados mesmo no início da
   frase. Reaplicada aos boletins da rodada 1, ela agora aponta "Manaus" no recorte "outros
   locais" (fidelidade de 100% para 75%).

### Rodada 2 (23/11/2026)

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade | Orientação escolhida |
|---|---|---|---|---|---|---|
| Panorama geral | 233 | 13 | 163,4 | **não** | 100% | — |
| Local: Manaus | 130 | 8 | 58,6 | sim | 100% | Péssima |
| Local: outros locais | 88 | 6 | 65,3 | **não** | 83% ("Manaus") | — |
| Mês: 09/2026 | 12 | 1 | 8,8 | sim | 100% | **Boa** |
| Mês: 10/2026 | 203 | 12 | 118,2 | sim | 100% | Péssima |
| Tema: qualidade do ar | 91 | 6 | 66,9 | sim | 100% | Péssima |
| Tema: queimadas fumaça | 71 | 5 | 38,8 | sim | 100% | Ruim |
| Tema: poluição do ar | 59 | 4 | 35,8 | sim | 100% | Péssima |
| Tema: índice de qualidade do ar | 12 | 1 | 7,3 | sim | 100% | Péssima |

**O que melhorou:** nenhuma recomendação inventada; a troca "poluição/qualidade do ar" não se
repetiu; a checagem ampliada apontou "Manaus" no recorte de outros locais, onde nenhuma manchete
cita a cidade.

**Problemas novos e correções para a rodada 3:**

| Problema | Causa | Correção |
|---|---|---|
| `formato_ok = false` nos dois maiores recortes | Com 12 ou 13 lotes, a etapa final recebia até 39 fatos; o modelo copiava a lista e esgotava o limite de tokens antes da recomendação | Consolidação em árvore (no máximo 12 fatos na etapa final), remoção de fatos repetidos e recomendação pedida **antes** dos pontos |
| Orientação "Boa" para "poluição acima do péssimo" (setembro) | O *prompt* tinha manchetes numeradas de 1 a 12 e orientações numeradas de 1 a 5; o modelo confundiu as listas | Orientações sem número; o modelo responde o **nome** da categoria. Verificação de coerência com o sentimento: reaplicada à rodada 2, marca só esse caso |
| Página "Índice de Qualidade do Ar (AQI) ... \| IQAir Brasil" virou ponto principal | O filtro só reconhecia "Qualidade do ar em X" | Filtro ampliado: 231 notícias válidas |

### Rodada 3 (23/11/2026)

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade | Orientação (LLM) | Coerente |
|---|---|---|---|---|---|---|---|
| Panorama geral | 231 | 23 | 280,2 | **não** | 71% | — | — |
| Local: Manaus | 130 | 11 | 115,9 | sim | 100% | Muito Ruim | sim |
| Local: outros locais | 86 | 9 | 104,9 | **não** | 100% | — | — |
| Mês: 09/2026 | 12 | 1 | 8,7 | sim | 100% | Muito Ruim | sim |
| Mês: 10/2026 | 203 | 21 | 228,7 | sim | 100% | **Moderada** | **não** |
| Tema: qualidade do ar | 91 | 9 | 83,2 | sim | 100% | **Moderada** | **não** |
| Tema: queimadas fumaça | 71 | 7 | 66,2 | sim | 100% | **Moderada** | **não** |
| Tema: poluição do ar | 58 | 6 | 54,4 | sim | 100% | Muito Ruim | sim |
| Tema: índice de qualidade do ar | 11 | 1 | 7,2 | sim | 100% | Muito Ruim | sim |

**Diagnóstico:** a consolidação em árvore resolveu o excesso de fatos, mas os dois maiores
recortes ainda saíram fora do formato: o modelo devolveu só uma lista, sem resumo e sem
recomendação, e uma das linhas era um "fato" sobre a numeração ("Manchetes 3, 4, 5 ... citam
Santarém"), que a checagem de fidelidade apontou (71%). Escolhida pelo nome, a orientação foi
"Moderada" em 3 recortes com mais de 90% de notícias negativas; a verificação de coerência marcou
os três.

**Mudança de abordagem (rodada 4):** três rodadas mostraram o mesmo limite: o modelo de 0,8B não
sustenta várias tarefas numa resposta só. A sumarização passou a ser uma **cadeia de *prompts* de
uma tarefa cada** (seção 3), e a recomendação saiu do LLM: passou a ser a orientação oficial da
categoria mais grave citada nas manchetes. Aplicada às fontes da rodada 3, essa regra dá:

| Recorte | Categoria citada | Manchetes que a citam | Coerente com o sentimento |
|---|---|---|---|
| Panorama geral | Péssima | 24 de 231 | sim |
| Local: Manaus | Péssima | 23 de 130 | sim |
| Local: outros locais | Ruim | 2 de 86 | sim |
| Mês: 09/2026 | Péssima | 1 de 12 | sim |
| Mês: 10/2026 | Péssima | 23 de 203 | sim |
| Tema: qualidade do ar | Péssima | 18 de 91 | sim |
| Tema: queimadas fumaça | Péssima | 4 de 71 | sim |
| Tema: poluição do ar | Ruim | 2 de 58 | sim |
| Tema: índice de qualidade do ar | Péssima (empate de 1 citação entre três categorias; vale a mais grave) | 1 de 11 | sim |

### Rodada 4 (a preencher)

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade | Orientação | Coerente |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
