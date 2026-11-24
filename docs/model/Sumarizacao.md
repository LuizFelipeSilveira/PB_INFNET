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

## 3. Estratégia: *map-reduce*

Um resumo de muitas manchetes de uma vez sobrecarrega um modelo pequeno e fica lento. O projeto
resume em duas etapas:

1. **Map:** as manchetes do recorte são divididas em lotes de até 20. Cada lote vira uma lista de
   3 fatos.
2. **Consolidação:** se houver mais de 12 fatos, eles são agrupados de 12 em 12 e cada grupo é
   reduzido aos 3 mais importantes, até sobrarem no máximo 12 (*reduce* em árvore). Fatos
   repetidos são removidos em cada passo.
3. **Reduce:** os fatos de todos os lotes viram um boletim com formato fixo: **Resumo** (duas
   frases), **Pontos principais** (três itens) e **Recomendação**. A recomendação não é escrita
   pelo modelo: ele escolhe qual das 5 orientações oficiais de saúde (uma por categoria do IQAr)
   combina com o noticiário, e o boletim traz o texto da norma (seção 7).

Recortes com até 20 manchetes vão direto para o boletim, numa chamada só.

O LLM só escreve texto. **Tudo o que é número vem do código**: quantidade de notícias, período,
veículos, locais e distribuição de sentimento são calculados com pandas. Assim, cada número
mostrado no dashboard é exato.

## 4. Engenharia de *prompts*

As lições das quatro rodadas de avaliação da TP4 orientaram os *prompts* da sumarização:

| Decisão | Motivo (evidência da TP4) |
|---|---|
| Papel fixo na mensagem de sistema ("analista do Éter", português, fiel ao noticiado) | Mantém idioma e tom em todas as chamadas |
| Manchetes **numeradas** no *prompt* | Deixa claro onde começa e termina cada item; reduz mistura de notícias |
| Instrução explícita "use somente as informações acima; não invente números, datas nem locais" | *Grounding*: na TP4, o contexto no *prompt* levou as perguntas a 100% de acerto |
| Formato de saída exato, com rótulos (`Resumo:`, `Pontos principais:`, `Recomendação:`) | Permite separar as partes por código e verificar se o modelo obedeceu (`formato_ok`) |
| Pedido de exatamente 3 fatos por lote | Limita o tamanho da saída e força a escolha do que é mais importante |
| "Cite o local e os números quando as manchetes trouxerem" | Resumos sem local e sem números são vagos e pouco acionáveis |
| "Escreva o local exatamente como na manchete" e "repita a palavra que vem junto do número" | Rodada 1: o modelo trocou Amazonas por Manaus e transformou "964% na poluição" em "964% na qualidade do ar" |
| Recomendação por **escolha fechada** entre as 5 orientações oficiais (responde o nome da categoria), pedida **antes** dos pontos | Rodada 1: o modelo recomendou "evite sair da cidade". Escolher numa lista, como na classificação de sentimento, é o que um modelo pequeno faz com mais acerto, e o texto final é o da norma |
| Decodificação determinística (`do_sample=False`) | Mesma entrada, mesmo boletim: resultados reproduzíveis |

Os *prompts* completos estão em `src/sumarizacao.py` (`PROMPT_FATOS` e `PROMPT_BOLETIM`).

## 5. Como a qualidade é medida

| Métrica | Como é calculada | Para que serve |
|---|---|---|
| `formato_ok` | o código encontrou Resumo, Pontos principais e uma orientação oficial válida | mostra se o modelo seguiu as instruções |
| `recomendacao_coerente` | orientação branda (Boa/Moderada) com 60% ou mais de notícias negativas, ou grave (Muito Ruim/Péssima) com menos de 30%, é marcada como incoerente | confere a escolha do modelo com um dado calculado de outra forma (o sentimento) |
| Fidelidade (`taxa`) | fração dos números, nomes próprios e locais conhecidos (capitais e estados, mesmo no início da frase) do boletim que aparecem nas manchetes de origem | aponta possíveis invenções (`nao_encontrados`) para conferir |
| Taxa de compressão | palavras do boletim / palavras das manchetes | mede o quanto a leitura foi reduzida |
| Revisão manual | leitura dos boletins ao lado das fontes | detecta erros que a checagem automática não pega (ex.: causa invertida) |

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

### Rodada 3 (a preencher)

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade | Orientação | Coerente |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
