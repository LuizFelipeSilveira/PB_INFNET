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
| Remover páginas de índice do IQAir ("Qualidade do ar em Campinas") | São páginas fixas de monitoramento que o feed traz como notícia; não relatam fatos. Títulos iguais vindos de veículos de notícia são mantidos | 281 → 233 notícias válidas (48 removidas) |
| Usar só a manchete como texto | O campo `resumo` do feed repete a manchete e acrescenta o nome do veículo ou manchetes de outras notícias | Entrada mais curta e sem ruído |
| Detectar o local citado | Permite recortar por cidade | Manaus: 130; outros locais: 88; Santarém: 8; Porto Velho: 4 |
| Extrair o mês | Permite recortar por período | outubro/2026: 203; setembro/2026: 12 |

## 3. Estratégia: *map-reduce*

Um resumo de muitas manchetes de uma vez sobrecarrega um modelo pequeno e fica lento. O projeto
resume em duas etapas:

1. **Map:** as manchetes do recorte são divididas em lotes de até 20. Cada lote vira uma lista de
   3 fatos.
2. **Reduce:** os fatos de todos os lotes viram um boletim com formato fixo: **Resumo** (duas
   frases), **Pontos principais** (três itens) e **Recomendação** (uma frase).

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
| Decodificação determinística (`do_sample=False`) | Mesma entrada, mesmo boletim: resultados reproduzíveis |

Os *prompts* completos estão em `src/sumarizacao.py` (`PROMPT_FATOS` e `PROMPT_BOLETIM`).

## 5. Como a qualidade é medida

| Métrica | Como é calculada | Para que serve |
|---|---|---|
| `formato_ok` | o código encontrou Resumo e Pontos principais na saída | mostra se o modelo seguiu as instruções |
| Fidelidade (`taxa`) | fração dos números e nomes próprios do boletim que aparecem nas manchetes de origem | aponta possíveis invenções (`nao_encontrados`) para conferir |
| Taxa de compressão | palavras do boletim / palavras das manchetes | mede o quanto a leitura foi reduzida |
| Revisão manual | leitura dos boletins ao lado das fontes | detecta erros que a checagem automática não pega (ex.: causa invertida) |

A checagem de fidelidade é simples de propósito: não prova que o resumo está certo, mas é barata,
roda em todo boletim e diz exatamente o que conferir. A recomendação fica fora dela, porque é uma
orientação de saúde, e não um fato noticiado.

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

*A preencher com a execução de `scripts/llm/sumarizar_noticias.py` na GPU de desenvolvimento.*

| Recorte | Notícias | Chamadas | Tempo (s) | `formato_ok` | Fidelidade | Compressão |
|---|---|---|---|---|---|---|
| | | | | | | |

Revisão manual: *a preencher.*
