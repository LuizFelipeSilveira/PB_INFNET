"""Testes da rota POST /processar_texto, dos formatos de resposta e dos erros da API.

O LLM de verdade é substituído por um "falso", então os testes rodam em segundos e
não precisam de GPU, torch nem do modelo baixado. Rode na raiz do projeto:
    pytest -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.schemas import ErroAPI, RespostaProcessamento
from src.llm import montar_prompt, rotulo_sentimento

cliente = TestClient(api_main.app)


class LLMFalso:
    repo = "modelo/falso"
    formato = "teste-cpu"

    def __init__(self, resposta: str = "negativo", erro: Exception | None = None):
        self.resposta, self.erro, self.pedidos = resposta, erro, []

    def processar(self, tarefa: str, texto: str) -> dict:
        if self.erro:
            raise self.erro
        self.pedidos.append((tarefa, texto))
        return {"resultado": self.resposta, "tokens_gerados": 3, "tempo_s": 0.1}


@pytest.fixture
def llm_falso(monkeypatch):
    falso = LLMFalso()
    monkeypatch.setattr(api_main, "llm", falso)
    return falso


def processar(**corpo):
    return cliente.post("/processar_texto", json=corpo)


# ---------------------------------------------------------------- respostas estruturadas
def test_sentimento_estruturado(llm_falso):
    llm_falso.resposta = "Negativo."
    r = processar(texto="  Fumaça deixa o ar péssimo  ", tarefa="sentimento")
    assert r.status_code == 200
    corpo = RespostaProcessamento.model_validate(r.json())  # o JSON segue o contrato
    assert corpo.tarefa == "sentimento"
    assert corpo.texto_entrada == "Fumaça deixa o ar péssimo"  # espaços das pontas removidos
    assert corpo.analise.tipo == "sentimento"
    assert corpo.analise.rotulo == "negativo" and corpo.analise.rotulo_valido
    assert corpo.analise.resposta_modelo == "Negativo."
    assert corpo.metadados.modelo == "modelo/falso" and corpo.metadados.tokens_gerados == 3


def test_sentimento_sem_rotulo_valido_continua_200(llm_falso):
    llm_falso.resposta = "Não consigo classificar."
    analise = processar(texto="texto qualquer", tarefa="sentimento").json()["analise"]
    assert analise["rotulo"] is None and analise["rotulo_valido"] is False


def test_resumo_estruturado_e_tarefa_padrao(llm_falso):
    llm_falso.resposta = "Manaus teve o pior ar do país."
    r = processar(texto="Texto longo sobre Manaus com doze palavras no total para o teste")
    assert r.status_code == 200
    analise = r.json()["analise"]
    assert r.json()["tarefa"] == "resumo" and analise["tipo"] == "resumo"
    assert analise["palavras_entrada"] == 12 and analise["palavras_resumo"] == 7
    assert analise["taxa_compressao"] == round(7 / 12, 2)


def test_pergunta_estruturada(llm_falso):
    llm_falso.resposta = "De 25 a 50 µg/m³."
    analise = processar(texto="Qual a faixa de pm 2,5 moderada?", tarefa="pergunta").json()["analise"]
    assert analise["tipo"] == "pergunta"
    assert analise["poluentes_consultados"] == ["MP2.5"]
    assert analise["resposta"] == "De 25 a 50 µg/m³."


def test_resposta_vazia_do_modelo_vira_502(llm_falso):
    llm_falso.resposta = ""
    r = processar(texto="Texto para resumir", tarefa="resumo")
    assert r.status_code == 502
    assert ErroAPI.model_validate(r.json()).erro == "resposta_invalida_do_modelo"


# ---------------------------------------------------------------- erros padronizados
@pytest.mark.parametrize("corpo, campo, trecho", [
    ({"texto": "oi"}, "texto", "pelo menos 3 caracteres"),
    ({"texto": "texto válido", "tarefa": "traduzir"}, "tarefa", "use um de"),
    ({"tarefa": "resumo"}, "texto", "obrigatório"),
    ({"texto": "texto válido", "tarfea": "resumo"}, "tarfea", "não reconhecido"),
])
def test_entrada_invalida_vira_422_em_portugues(llm_falso, corpo, campo, trecho):
    r = cliente.post("/processar_texto", json=corpo)
    assert r.status_code == 422
    erro = ErroAPI.model_validate(r.json())
    assert erro.erro == "entrada_invalida"
    assert any(c.campo == campo and trecho in c.mensagem for c in erro.campos)
    assert llm_falso.pedidos == []  # o modelo nem é chamado


def test_llm_indisponivel_vira_503(monkeypatch):
    monkeypatch.setattr(api_main, "llm", LLMFalso(erro=ImportError("No module named 'torch'")))
    r = processar(texto="texto qualquer", tarefa="resumo")
    assert r.status_code == 503
    erro = ErroAPI.model_validate(r.json())
    assert erro.erro == "llm_indisponivel" and "requirements-llm.txt" in erro.mensagem


def test_erros_das_outras_rotas_no_mesmo_formato():
    r = cliente.get("/iqar/calcular", params={"poluente": "XYZ", "concentracao": 10})
    assert r.status_code == 400 and r.json()["erro"] == "requisicao_invalida"
    r = cliente.get("/iqar/calcular", params={"poluente": "MP10", "concentracao": -5})
    assert r.status_code == 422 and r.json()["campos"][0]["campo"] == "concentracao"
    r = cliente.get("/rota/que/nao/existe")
    assert r.status_code == 404 and r.json()["erro"] == "nao_encontrado"


def test_rotas_de_consulta_seguem_o_contrato():
    r = cliente.get("/iqar/calcular", params={"poluente": "pm2.5", "concentracao": 30})
    assert r.status_code == 200
    assert r.json()["categoria"] == "Moderada" and 0 <= r.json()["iqar"] <= 400
    assert cliente.get("/").json()["status"] == "ok"
    assert len(cliente.get("/iqar/tabela").json()) == 30


# ---------------------------------------------------------------- funções de apoio
@pytest.mark.parametrize("resposta, esperado", [
    ("Positivo", "positivo"), ("neutro.", "neutro"), ("NEGATIVO!", "negativo"),
    ("positivo ou negativo", None), ("não sei", None),
])
def test_rotulo_sentimento(resposta, esperado):
    assert rotulo_sentimento(resposta) == esperado


def test_pergunta_usa_so_o_poluente_citado():
    prompt, _ = montar_prompt("pergunta", "Qual a faixa de pm 2,5 para qualidade ruim?")
    assert "MP2.5, Ruim" in prompt
    assert "MP10" not in prompt and "CO," not in prompt


def test_pergunta_sem_poluente_usa_tabela_toda():
    prompt, _ = montar_prompt("pergunta", "Como o índice é dividido em categorias?")
    for sigla in ("MP10", "MP2.5", "O3", "CO", "NO2", "SO2"):
        assert f"- {sigla}," in prompt
