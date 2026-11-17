"""Tratamento de erros da API do Éter.

Objetivo: qualquer problema vira uma resposta HTTP com o código certo e um corpo
no formato `ErroAPI` (api/schemas.py), que traz:
  * `erro`: código curto e estável, para o cliente tratar por programa;
  * `mensagem`: o que aconteceu, em português;
  * `sugestao`: o que a pessoa pode fazer para resolver;
  * `id_requisicao`: identificador que também aparece no log do servidor (e no
    cabeçalho X-Request-ID), para achar o erro exato na hora do diagnóstico.

Peças:
  * `ErroHTTP`: uma HTTPException do FastAPI com `codigo` e `sugestao`;
  * `erro_do_llm()`: traduz as exceções de src/llm.py em ErroHTTP (503, 504...);
  * `registrar(app)`: instala o middleware de id e os tratadores de exceção.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.schemas import CampoInvalido, ErroAPI
from src import llm as llm_mod

log = logging.getLogger("eter.api")
if not log.handlers:  # mensagens visíveis no terminal do uvicorn
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)

# Código padrão por status, usado quando quem levantou o erro não informou um.
CODIGOS_PADRAO = {400: "requisicao_invalida", 404: "nao_encontrado", 405: "metodo_nao_permitido",
                  422: "entrada_invalida", 500: "erro_interno",
                  502: "resposta_invalida_do_modelo", 503: "servico_indisponivel",
                  504: "tempo_esgotado"}


class ErroHTTP(HTTPException):
    """HTTPException do FastAPI com um código estável e uma sugestão para o usuário."""

    def __init__(self, status: int, codigo: str, mensagem: str, sugestao: str | None = None,
                 headers: dict | None = None):
        super().__init__(status_code=status, detail=mensagem, headers=headers)
        self.codigo, self.sugestao = codigo, sugestao


# ---------------------------------------------------------------- LLM -> HTTP
def erro_do_llm(erro: llm_mod.LLMErro) -> ErroHTTP:
    """Escolhe o código HTTP e a orientação certos para cada falha do LLM local."""
    if isinstance(erro, llm_mod.LLMDependenciasAusentes):
        return ErroHTTP(503, "llm_dependencias_ausentes",
                        f"As bibliotecas do modelo de linguagem não estão instaladas ({erro}).",
                        "Instale o PyTorch (docs/model/ModelSelection.md, seção 5) e depois "
                        "`pip install -r requirements-llm.txt`.")
    if isinstance(erro, llm_mod.LLMModeloNaoEncontrado):
        return ErroHTTP(503, "llm_modelo_nao_encontrado",
                        f"O modelo de linguagem não foi encontrado nesta máquina ({erro}).",
                        "Baixe o modelo uma vez com `python scripts/llm/baixar_modelo.py` e "
                        "confira a variável ETER_LLM_MODELO.")
    if isinstance(erro, llm_mod.LLMMemoriaInsuficiente):
        return ErroHTTP(503, "llm_memoria_insuficiente",
                        "Faltou memória na placa de vídeo para processar o texto.",
                        "Feche outros programas que usam a GPU, envie um texto menor ou use "
                        "um modelo menor (ETER_LLM_MODELO=Qwen/Qwen3.5-0.8B).")
    if isinstance(erro, llm_mod.LLMOcupado):
        return ErroHTTP(503, "llm_ocupado",
                        f"O modelo está processando outro pedido ({erro}).",
                        "Aguarde alguns segundos e tente de novo.",
                        headers={"Retry-After": str(int(erro.espera_s))})
    if isinstance(erro, llm_mod.LLMTempoEsgotado):
        return ErroHTTP(504, "llm_tempo_esgotado",
                        f"O processamento demorou demais e foi interrompido ({erro}).",
                        "Envie um texto menor ou aumente ETER_LLM_TIMEOUT_S. Na CPU, a geração "
                        "é bem mais lenta que na GPU.")
    if isinstance(erro, llm_mod.LLMFalhaCarga):
        return ErroHTTP(503, "llm_falha_carga",
                        f"O modelo de linguagem não pôde ser carregado ({erro}).",
                        "Veja o log da API e o campo `ultimo_erro` em GET /llm/status.")
    return ErroHTTP(500, "llm_falha_geracao",
                    f"Erro inesperado ao gerar o texto ({erro}).",
                    "Tente novamente; se persistir, informe o id_requisicao.")


# ---------------------------------------------------------------- respostas de erro
def _id(request: Request) -> str:
    return getattr(request.state, "id_requisicao", "-")


def resposta_erro(request: Request, status: int, codigo: str, mensagem: str,
                  sugestao: str | None = None, campos: list[CampoInvalido] | None = None,
                  headers: dict | None = None) -> JSONResponse:
    corpo = ErroAPI(status=status, erro=codigo, mensagem=mensagem, sugestao=sugestao,
                    campos=campos or [], id_requisicao=_id(request))
    nivel = logging.ERROR if status >= 500 else logging.WARNING
    log.log(nivel, "[%s] %s %s -> %s %s: %s", corpo.id_requisicao, request.method,
            request.url.path, status, codigo, mensagem)
    cabecalhos = {"X-Request-ID": corpo.id_requisicao, **(headers or {})}
    return JSONResponse(status_code=status, content=corpo.model_dump(), headers=cabecalhos)


def _traduzir(e: dict) -> str:
    """Mensagens do Pydantic em português, para os casos mais comuns."""
    ctx = e.get("ctx", {})
    return {
        "missing": "campo obrigatório",
        "string_too_short": f"deve ter pelo menos {ctx.get('min_length')} caracteres",
        "string_too_long": f"deve ter no máximo {ctx.get('max_length')} caracteres",
        "literal_error": f"valor inválido; use um de: {ctx.get('expected')}",
        "greater_than_equal": f"deve ser maior ou igual a {ctx.get('ge')}",
        "less_than_equal": f"deve ser menor ou igual a {ctx.get('le')}",
        "float_parsing": "deve ser um número",
        "int_parsing": "deve ser um número inteiro",
        "date_from_datetime_parsing": "deve ser uma data no formato AAAA-MM-DD",
        "extra_forbidden": "campo não reconhecido por esta rota",
        "json_invalid": "o corpo não é um JSON válido",
        "model_attributes_type": "o corpo deve ser um objeto JSON",
        # erros levantados pelos nossos próprios validadores (field_validator)
        "value_error": e["msg"].removeprefix("Value error, "),
    }.get(e["type"], e["msg"])


# ---------------------------------------------------------------- instalação no app
def registrar(app: FastAPI) -> None:
    @app.middleware("http")
    async def id_e_tempo(request: Request, chamar_proximo):
        """Dá um id curto a cada requisição e o devolve no cabeçalho X-Request-ID."""
        request.state.id_requisicao = uuid.uuid4().hex[:8]
        inicio = time.perf_counter()
        resposta = await chamar_proximo(request)
        resposta.headers["X-Request-ID"] = request.state.id_requisicao
        resposta.headers["X-Tempo-Processamento-s"] = f"{time.perf_counter() - inicio:.3f}"
        return resposta

    @app.exception_handler(RequestValidationError)
    def _validacao(request: Request, exc: RequestValidationError) -> JSONResponse:
        campos = [CampoInvalido(campo=".".join(str(p) for p in e["loc"][1:]) or str(e["loc"][0]),
                                mensagem=_traduzir(e)) for e in exc.errors()]
        return resposta_erro(request, 422, "entrada_invalida",
                             "A requisição tem campos inválidos. Veja a lista em 'campos'.",
                             "Corrija os campos indicados e envie de novo. O formato esperado "
                             "de cada rota está em /docs.", campos)

    @app.exception_handler(StarletteHTTPException)
    def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codigo = getattr(exc, "codigo", None) or CODIGOS_PADRAO.get(exc.status_code, "erro")
        sugestao = getattr(exc, "sugestao", None)
        if exc.status_code == 404 and sugestao is None:
            sugestao = "Confira o endereço. A lista de rotas está em /docs."
        return resposta_erro(request, exc.status_code, codigo, str(exc.detail), sugestao,
                             headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    def _inesperado(request: Request, exc: Exception) -> JSONResponse:
        # Último recurso: nunca devolve o traceback ao cliente, mas registra no log
        # com o mesmo id para o diagnóstico.
        log.exception("[%s] erro não tratado em %s", _id(request), request.url.path)
        return resposta_erro(request, 500, "erro_interno",
                             "Ocorreu um erro interno inesperado na API.",
                             "Tente novamente. Se persistir, informe o id_requisicao a quem "
                             "mantém a API (o detalhe está no log do servidor).")
