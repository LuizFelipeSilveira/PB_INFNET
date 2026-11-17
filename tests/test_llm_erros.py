"""Testes das falhas do LLMLocal (src/llm.py), sem precisar de torch nem do modelo.

`carregar` e `gerar` são substituídos por versões falsas que simulam cada problema.
"""
from __future__ import annotations

import threading

import pytest

from src import llm


@pytest.fixture
def modelo(monkeypatch):
    monkeypatch.setenv("ETER_LLM_ESPERA_S", "0.1")
    monkeypatch.setenv("ETER_LLM_TIMEOUT_S", "5")
    return llm.LLMLocal()


@pytest.mark.parametrize("falha, esperado", [
    (ImportError("No module named 'torch'"), llm.LLMDependenciasAusentes),
    (OSError("Qwen/X is not a local folder and is not a valid model identifier"),
     llm.LLMModeloNaoEncontrado),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), llm.LLMMemoriaInsuficiente),
    (ValueError("config.json inválido"), llm.LLMFalhaCarga),
])
def test_falhas_de_carga_viram_excecoes_especificas(monkeypatch, modelo, falha, esperado):
    def carregar_falho(*_):
        raise falha
    monkeypatch.setattr(llm, "carregar", carregar_falho)
    with pytest.raises(esperado):
        modelo.processar("resumo", "texto qualquer")
    assert modelo.status()["ultimo_erro"].startswith(esperado.__name__)
    assert not modelo.carregado
    assert not modelo.status()["ocupado"]  # o cadeado foi liberado mesmo com erro


def test_depois_de_uma_falha_o_proximo_pedido_tenta_carregar_de_novo(monkeypatch, modelo):
    tentativas = []

    def carregar(*_):
        tentativas.append(1)
        if len(tentativas) == 1:
            raise OSError("sem internet")
        return object(), object(), "fp16-gpu"
    monkeypatch.setattr(llm, "carregar", carregar)
    monkeypatch.setattr(llm, "gerar", lambda *a, **k: ("ok", 1, 0.1))
    with pytest.raises(llm.LLMModeloNaoEncontrado):
        modelo.processar("resumo", "texto qualquer")
    assert modelo.processar("resumo", "texto qualquer")["resultado"] == "ok"
    assert modelo.status()["ultimo_erro"] is None  # limpo após a carga bem-sucedida


def test_modelo_ocupado_desiste_depois_da_espera(monkeypatch, modelo):
    monkeypatch.setattr(llm, "carregar", lambda *_: (object(), object(), "fp16-gpu"))
    liberar = threading.Event()

    def gerar_lento(*a, **k):
        liberar.wait(2)
        return "ok", 1, 0.1
    monkeypatch.setattr(llm, "gerar", gerar_lento)
    primeiro = threading.Thread(target=modelo.processar, args=("resumo", "texto um"))
    primeiro.start()
    try:
        with pytest.raises(llm.LLMOcupado):
            modelo.processar("resumo", "texto dois")
    finally:
        liberar.set()
        primeiro.join()


def test_geracao_que_passa_do_limite_vira_tempo_esgotado(monkeypatch, modelo):
    monkeypatch.setattr(llm, "carregar", lambda *_: (object(), object(), "fp16-gpu"))
    limites = []

    def gerar_interrompido(*a, max_time=None, **k):
        limites.append(max_time)
        return "texto pela met", 40, max_time  # o generate parou no limite
    monkeypatch.setattr(llm, "gerar", gerar_interrompido)
    with pytest.raises(llm.LLMTempoEsgotado):
        modelo.processar("resumo", "texto qualquer")
    assert limites == [5.0]  # o limite foi repassado ao generate (max_time)


def test_falta_de_memoria_na_geracao(monkeypatch, modelo):
    monkeypatch.setattr(llm, "carregar", lambda *_: (object(), object(), "fp16-gpu"))

    def gerar_sem_memoria(*a, **k):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(llm, "gerar", gerar_sem_memoria)
    with pytest.raises(llm.LLMMemoriaInsuficiente):
        modelo.processar("resumo", "texto qualquer")
