"""Testes da rota POST /processar_texto e das funções de apoio em src/llm.py.

O LLM de verdade é substituído por um "falso", então os testes rodam em segundos e
não precisam de GPU, torch nem do modelo baixado. Rode na raiz do projeto:
    pytest -q
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
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
        return {"resultado": self.resposta,
                "sentimento": rotulo_sentimento(self.resposta) if tarefa == "sentimento" else None,
                "tokens_gerados": 3, "tempo_s": 0.1}


@pytest.fixture
def llm_falso(monkeypatch):
    falso = LLMFalso()
    monkeypatch.setattr(api_main, "llm", falso)
    return falso


def test_sentimento_devolve_rotulo_padronizado(llm_falso):
    llm_falso.resposta = "Negativo."
    r = cliente.post("/processar_texto",
                     json={"texto": "Fumaça deixa o ar péssimo", "tarefa": "sentimento"})
    assert r.status_code == 200
    assert r.json()["sentimento"] == "negativo"
    assert llm_falso.pedidos == [("sentimento", "Fumaça deixa o ar péssimo")]


def test_resumo_e_tarefa_padrao(llm_falso):
    llm_falso.resposta = "Manaus teve o pior ar do país."
    r = cliente.post("/processar_texto", json={"texto": "Texto longo sobre Manaus..."})
    assert r.status_code == 200
    assert r.json()["tarefa"] == "resumo"
    assert r.json()["sentimento"] is None


@pytest.mark.parametrize("corpo", [
    {"texto": "oi"},                                    # curto demais
    {"texto": "texto válido", "tarefa": "traduzir"},    # tarefa inexistente
    {"tarefa": "resumo"},                               # sem texto
])
def test_validacao_de_entrada(llm_falso, corpo):
    assert cliente.post("/processar_texto", json=corpo).status_code == 422
    assert llm_falso.pedidos == []


def test_llm_indisponivel_vira_503(monkeypatch):
    monkeypatch.setattr(api_main, "llm", LLMFalso(erro=ImportError("No module named 'torch'")))
    r = cliente.post("/processar_texto", json={"texto": "texto qualquer", "tarefa": "resumo"})
    assert r.status_code == 503
    assert "requirements-llm.txt" in r.json()["detail"]


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
