"""Modelos Pydantic da API: o "contrato" de entrada e saída de cada rota.

Toda rota declara `response_model=...`. Com isso, o FastAPI **valida a resposta antes
de enviá-la**: se algum campo vier fora do tipo ou das restrições abaixo (ex.: IQAr
negativo, categoria inexistente, resumo vazio), a resposta não sai com dados
errados. Os mesmos modelos geram a documentação em /docs.

Restrições usadas:
  * `Literal[...]`  -> só aceita os valores listados (categorias, tarefas, rótulos)
  * `Field(ge=, le=)` -> limites numéricos (ex.: IQAr entre 0 e 400)
  * `min_length`    -> textos não podem vir vazios
  * `extra="forbid"` nos pedidos -> campos desconhecidos são rejeitados (evita erro de digitação
    silencioso, como mandar "tarfea" e cair no valor padrão sem perceber)
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.data_loader import POLUENTES_VALIDOS

Poluente = Literal["MP10", "MP2.5", "O3", "CO", "NO2", "SO2"]
Categoria = Literal["Boa", "Moderada", "Ruim", "Muito Ruim", "Péssima"]
Tarefa = Literal["resumo", "sentimento", "pergunta"]
Rotulo = Literal["positivo", "negativo", "neutro"]
IQAr = Annotated[int, Field(ge=0, le=400, description="Índice de Qualidade do Ar (0–400).")]


class Pedido(BaseModel):
    """Base dos corpos de requisição: rejeita campos que a rota não conhece."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------- consulta e leituras
class StatusAPI(BaseModel):
    status: Literal["ok"]
    app: str
    versao: str
    docs: str


class Cidade(BaseModel):
    nome: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class FaixaReferencia(BaseModel):
    poluente: Poluente
    nome: str
    unidade: Literal["µg/m³", "ppm"]
    categoria: Categoria
    indice_min: IQAr
    indice_max: IQAr
    conc_min: float = Field(..., ge=0)
    conc_max: float | None = Field(None, ge=0, description="Fim da faixa (na última, o teto "
                                                           "técnico do Quadro 1).")
    recomendacao: str = Field(..., min_length=1)
    fonte: str | None = Field(None, description="De onde veio a tabela (PDF do MMA ou reserva).")


class ResultadoIQAr(BaseModel):
    poluente: Poluente
    concentracao: float = Field(..., ge=0)
    iqar: IQAr
    categoria: Categoria
    recomendacao: str = Field(..., min_length=1)


class NovaLeitura(Pedido):
    """Corpo esperado por `POST /leituras`."""

    cidade: str = Field(..., min_length=2, max_length=100, examples=["Rio de Janeiro - RJ"])
    poluente: str = Field(..., description=f"Um de: {sorted(POLUENTES_VALIDOS)} (aceita PM no "
                                           "lugar de MP, ex.: PM2.5).", examples=["MP2.5"])
    valor: float = Field(..., ge=0, description="Concentração na unidade oficial do poluente.")
    data: date | None = Field(None, description="Data da medição. Se omitida, usa a data atual.")
    origem: str = Field("api", max_length=50,
                        description="De onde veio a medição (sensor, estação, etc.).")


class LeituraRegistrada(BaseModel):
    """O que a API devolve depois de registrar uma leitura (inclui o IQAr calculado)."""

    cidade: str
    poluente: Poluente
    valor: float = Field(..., ge=0)
    data: date
    origem: str
    recebido_em: datetime
    iqar: IQAr
    categoria: Categoria


# ---------------------------------------------------------------- LLM local (TP4)
class PedidoTexto(Pedido):
    """Corpo esperado por `POST /processar_texto`."""

    texto: str = Field(..., min_length=3, max_length=4000,
                       examples=["Fumaça das queimadas deixa o ar de Manaus em nível péssimo e "
                                 "escolas suspendem as aulas nesta segunda-feira."])
    tarefa: Tarefa = Field(
        "resumo",
        description="resumo: resume o texto em até duas frases; sentimento: positivo, negativo "
                    "ou neutro; pergunta: responde usando a tabela oficial do IQAr.")

    @field_validator("texto")
    @classmethod
    def texto_com_conteudo(cls, valor: str) -> str:
        """Espaços nas pontas não contam: "   oi   " não passa só por ter 8 caracteres."""
        valor = valor.strip()
        if len(valor) < 3:
            raise ValueError("deve ter pelo menos 3 caracteres além de espaços")
        return valor


class AnaliseResumo(BaseModel):
    tipo: Literal["resumo"] = "resumo"
    resumo: str = Field(..., min_length=1, description="Resumo gerado pelo modelo.")
    palavras_entrada: int = Field(..., ge=1)
    palavras_resumo: int = Field(..., ge=1)
    taxa_compressao: float = Field(..., ge=0, description="palavras_resumo / palavras_entrada.")


class AnaliseSentimento(BaseModel):
    tipo: Literal["sentimento"] = "sentimento"
    rotulo: Rotulo | None = Field(..., description="Rótulo padronizado; vazio se o modelo não "
                                                   "respondeu exatamente um rótulo válido.")
    rotulo_valido: bool = Field(..., description="False quando `rotulo` está vazio.")
    resposta_modelo: str = Field(..., description="Texto exato devolvido pelo modelo.")


class AnalisePergunta(BaseModel):
    tipo: Literal["pergunta"] = "pergunta"
    pergunta: str
    resposta: str = Field(..., min_length=1)
    poluentes_consultados: list[Poluente] = Field(
        ..., description="Poluentes da tabela enviados como contexto ao modelo.")
    fonte: str = Field(..., description="Origem da tabela usada como contexto.")


# O campo `tipo` diz qual dos três formatos veio ("união discriminada"): o cliente
# olha `analise.tipo` e já sabe quais campos existem.
Analise = Annotated[Union[AnaliseResumo, AnaliseSentimento, AnalisePergunta],
                    Field(discriminator="tipo")]


class MetadadosLLM(BaseModel):
    modelo: str = Field(..., examples=["Qwen/Qwen3.5-0.8B"])
    formato: str = Field(..., description="Precisão e dispositivo, ex.: fp16-gpu ou fp32-cpu.")
    tokens_gerados: int = Field(..., ge=0)
    tempo_s: float = Field(..., ge=0, description="Tempo de geração (sem contar a carga).")
    processado_em: datetime


class RespostaProcessamento(BaseModel):
    """Resposta de `POST /processar_texto`."""

    tarefa: Tarefa
    texto_entrada: str = Field(..., description="Texto recebido, como foi processado.")
    analise: Analise
    metadados: MetadadosLLM


class StatusLLM(BaseModel):
    modelo: str
    origem: str = Field(..., description="Pasta local (modelos/...) ou nome no Hugging Face.")
    carregado: bool
    ocupado: bool = Field(..., description="True enquanto um pedido está sendo processado.")
    formato: str | None = None
    tempo_carga_s: float | None = Field(None, ge=0)
    timeout_s: float = Field(..., gt=0, description="Tempo máximo de geração por pedido.")
    ultimo_erro: str | None = Field(None, description="Última falha do modelo (diagnóstico); "
                                                      "volta a vazio após uma carga bem-sucedida.")


# ---------------------------------------------------------------- erros
class CampoInvalido(BaseModel):
    campo: str = Field(..., examples=["texto"])
    mensagem: str = Field(..., examples=["deve ter pelo menos 3 caracteres"])


class ErroAPI(BaseModel):
    """Formato único de TODOS os erros da API (4xx e 5xx). Ver api/erros.py."""

    status: int = Field(..., ge=400, le=599)
    erro: str = Field(..., description="Código curto e estável, bom para tratar no cliente.",
                      examples=["llm_modelo_nao_encontrado"])
    mensagem: str = Field(..., description="O que aconteceu, em português.")
    sugestao: str | None = Field(None, description="O que fazer para resolver.")
    campos: list[CampoInvalido] = Field(default_factory=list,
                                        description="Campos com problema (erros 422).")
    id_requisicao: str = Field(..., description="Também no cabeçalho X-Request-ID e no log do "
                                                "servidor; informe-o ao relatar um problema.")


# ---------------------------------------------------------------- sumarização (TP5)
class PedidoSumarizacao(Pedido):
    """Corpo de `POST /sumarizar_noticias`. Todos os filtros são opcionais."""

    local: str | None = Field(None, examples=["Manaus"],
                              description="Cidade citada (ver GET /noticias/recortes).")
    mes: str | None = Field(None, pattern=r"^\d{4}-\d{2}$", examples=["2026-10"],
                            description="Mês no formato AAAA-MM.")
    tema: str | None = Field(None, examples=["queimadas fumaça"],
                             description="Termo de busca que trouxe a notícia.")
    max_noticias: int = Field(30, ge=3, le=60, description="Usa as N notícias mais recentes "
                                                          "do recorte (mais notícias = mais lento).")


class EstatisticasNoticias(BaseModel):
    n_noticias: int = Field(..., ge=1)
    periodo_inicio: date
    periodo_fim: date
    veiculos: dict[str, int] = Field(..., description="Até 5 veículos com mais notícias.")
    locais: dict[str, int]
    sentimento: dict[str, int] | None = Field(None, description="Contagem por rótulo, se o "
                                                                "sentimento já foi calculado.")


class BoletimNoticias(BaseModel):
    resumo: str = Field(..., min_length=1)
    pontos: list[str]
    recomendacao: str | None = Field(None, description="Orientação oficial de saúde (texto da "
                                                       "norma) da categoria citada nas manchetes.")
    categoria_recomendacao: Categoria | None = Field(
        None, description="Categoria oficial mais grave citada nas manchetes (por código).")
    manchetes_que_citam_categoria: int = Field(0, ge=0, description="Evidência da escolha.")
    recomendacao_coerente: bool | None = Field(
        None, description="False se a orientação não combina com o sentimento das notícias "
                          "(ex.: 'Boa' com maioria negativa). Vazio sem sentimento calculado.")
    formato_ok: bool = Field(..., description="False se o resumo não veio em texto corrido ou "
                                              "faltaram pontos.")


class Fidelidade(BaseModel):
    termos_verificados: int = Field(..., ge=0)
    nao_encontrados: list[str] = Field(..., description="Números ou nomes do boletim que não "
                                                        "aparecem nas manchetes: conferir.")
    taxa: float = Field(..., ge=0, le=1)


class FonteNoticia(BaseModel):
    titulo: str
    veiculo: str
    data: date
    link: str | None = None
    sentimento: Rotulo | None = None


class ProcessamentoSumarizacao(BaseModel):
    n_lotes: int = Field(..., ge=1)
    n_chamadas: int = Field(..., ge=1)
    tokens_gerados: int = Field(..., ge=0)
    tempo_s: float = Field(..., ge=0)
    palavras_entrada: int = Field(..., ge=1)
    palavras_saida: int = Field(..., ge=1)
    modelo: str
    formato: str
    processado_em: datetime


class RespostaSumarizacao(BaseModel):
    """Resposta de `POST /sumarizar_noticias`."""

    descricao: str = Field(..., examples=["em Manaus, em 10/2026"])
    filtros: dict[str, str | int | None]
    estatisticas: EstatisticasNoticias
    boletim: BoletimNoticias
    fidelidade: Fidelidade
    fontes: list[FonteNoticia]
    processamento: ProcessamentoSumarizacao


class RecortesDisponiveis(BaseModel):
    """Valores aceitos nos filtros de `POST /sumarizar_noticias`, com a contagem de notícias."""

    total: int
    locais: dict[str, int]
    meses: dict[str, int]
    temas: dict[str, int]
