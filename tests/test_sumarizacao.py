"""Testes da sumarização do noticiário (src/sumarizacao.py e POST /sumarizar_noticias).

O LLM é substituído por funções falsas, então os testes não precisam do modelo.
"""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src import llm
from src import sumarizacao as sz

BOLETIM_OK = """Resumo: Fumaça de queimadas piorou o ar em Manaus. Escolas suspenderam aulas.
Pontos principais:
- Manaus teve a pior qualidade do ar do Brasil
- Aulas foram suspensas
- A chuva melhorou o ar
Recomendação: 4"""


def noticias_exemplo(n: int = 45) -> pd.DataFrame:
    linhas = [{"titulo": f"Fumaça em Manaus piora o ar pela {i}ª vez", "fonte": "G1",
               "data_publicacao": f"2026-10-{(i % 28) + 1:02d} 10:00", "link": f"https://x/{i}",
               "resumo": "", "termo_busca": "queimadas fumaça", "coletado_em": "2026-10-19"}
              for i in range(n)]
    linhas += [  # páginas de índice do IQAir e uma duplicata: devem sair
        {"titulo": "Qualidade do ar em Campinas", "fonte": "IQAir",
         "data_publicacao": "2026-06-01 10:00", "link": None, "resumo": "",
         "termo_busca": "qualidade do ar", "coletado_em": "2026-10-19"},
        {**{"titulo": "Fumaça em Manaus piora o ar pela 0ª vez", "fonte": "UOL",
            "data_publicacao": "2026-10-01 10:00", "link": None, "resumo": "",
            "termo_busca": "poluição do ar", "coletado_em": "2026-10-19"}},
        {"titulo": "Santarém amanhece coberta por fumaça", "fonte": "G1",
         "data_publicacao": "2026-09-20 08:00", "link": None, "resumo": "",
         "termo_busca": "queimadas fumaça", "coletado_em": "2026-10-19"},
    ]
    return pd.DataFrame(linhas)


class GeradorFalso:
    """Registra os prompts recebidos; devolve fatos na etapa map e o boletim no fim."""

    def __init__(self, boletim: str = BOLETIM_OK):
        self.prompts, self.boletim = [], boletim

    def __call__(self, prompt: str, max_tokens: int) -> dict:
        self.prompts.append(prompt)
        texto = self.boletim if "Escreva um boletim" in prompt else "- fato A\n- fato B\n- fato C"
        return {"resultado": texto, "tokens_gerados": 10, "tempo_s": 0.1}


# ---------------------------------------------------------------- preparo
def test_preparo_remove_paginas_de_indice_e_duplicatas():
    df = sz.preparar_noticias(noticias_exemplo(5))
    assert "Qualidade do ar em Campinas" not in df["titulo"].tolist()
    assert df["titulo"].is_unique and len(df) == 6
    assert set(df["local"]) == {"Manaus", "Santarém"}
    assert df["data"].is_monotonic_decreasing  # mais recentes primeiro


@pytest.mark.parametrize("texto, local", [
    ("Fumaça cobre Manaus", "Manaus"), ("Ar de SAO PAULO piora", "São Paulo"),
    ("Queimadas no Pará", sz.OUTROS_LOCAIS),
])
def test_detectar_local(texto, local):
    assert sz.detectar_local(texto) == local


def test_recortes_padrao_respeitam_o_minimo():
    df = sz.preparar_noticias(noticias_exemplo(45))
    ids = [r["id"] for r in sz.recortes_padrao(df, minimo=10)]
    assert ids[0] == "geral" and "local-manaus" in ids and "mes-2026-10" in ids
    assert "local-santarem" not in ids  # só 1 notícia


# ---------------------------------------------------------------- map-reduce
def test_um_lote_vai_direto_para_o_boletim():
    gerar = GeradorFalso()
    r = sz.sumarizar([f"manchete {i}" for i in range(12)], gerar)
    assert r.n_lotes == 1 and r.n_chamadas == 1
    assert "1. manchete 0" in gerar.prompts[0]  # manchetes numeradas no prompt


def test_varios_lotes_fazem_map_e_depois_reduce():
    gerar = GeradorFalso()
    r = sz.sumarizar([f"manchete {i}" for i in range(45)], gerar)
    assert r.n_lotes == 3 and r.n_chamadas == 4          # 3 map + 1 reduce
    assert "Liste os 3 fatos" in gerar.prompts[0]
    assert "fatos extraídos" in gerar.prompts[-1] and "- fato A" in gerar.prompts[-1]
    assert r.boletim.formato_ok and len(r.boletim.pontos) == 3
    assert r.boletim.categoria_recomendacao == "Muito Ruim"
    assert r.boletim.recomendacao.startswith("Aumento dos sintomas")  # texto oficial, não gerado


def test_boletim_fora_do_formato_e_sinalizado():
    b = sz.interpretar_boletim("Manaus teve muita fumaça hoje.")
    assert not b.formato_ok and b.resumo == "Manaus teve muita fumaça hoje."


def test_boletim_tolera_negrito_e_numeracao():
    b = sz.interpretar_boletim("**Resumo:** Ar ruim.\n**Pontos principais:**\n1. Um\n2. Dois\n"
                               "**Recomendação:** 3")
    assert b.formato_ok and b.pontos == ["Um", "Dois"] and b.categoria_recomendacao == "Ruim"


@pytest.mark.parametrize("resposta, categoria", [
    ("4", "Muito Ruim"), ("Orientação 5.", "Péssima"), ("Muito Ruim", "Muito Ruim"),
    ("ruim", "Ruim"), ("Evite sair da cidade", None),
])
def test_recomendacao_e_escolhida_entre_as_oficiais(resposta, categoria):
    assert sz.escolher_orientacao(resposta) == categoria


def test_recomendacao_inventada_nao_passa():
    """Rodada 1: o modelo escreveu "Evite sair da cidade"; agora isso vira formato_ok=False."""
    b = sz.interpretar_boletim("Resumo: Ar ruim.\nPontos principais:\n- Um\n"
                               "Recomendação: Evite sair da cidade e use máscara.")
    assert not b.formato_ok and b.recomendacao is None


# ---------------------------------------------------------------- fidelidade
def test_fidelidade_aponta_numeros_e_nomes_ausentes_das_fontes():
    fontes = ["Manaus registra alta de 964% na poluição", "Defesa Civil envia alerta em Manaus"]
    ok = sz.verificar_fidelidade("Resumo: Em Manaus a poluição subiu 964%. A Defesa Civil "
                                 "alertou.", fontes)
    assert ok["nao_encontrados"] == [] and ok["taxa"] == 1.0
    ruim = sz.verificar_fidelidade("Resumo: Em Belém houve 12 focos e a Defesa Civil agiu.",
                                   fontes)
    assert set(ruim["nao_encontrados"]) == {"Belém", "12"}
    assert 0 < ruim["taxa"] < 1
    # Rodada 1: "Manaus" no início da frase, no lugar de "Amazonas", passava sem ser visto
    troca = sz.verificar_fidelidade("Manaus tem 10 municípios sob alerta.",
                                    ["Amazonas tem 10 municípios sob alerta severo"])
    assert troca["nao_encontrados"] == ["Manaus"]


def test_resumir_recorte_junta_texto_estatisticas_e_fontes():
    df = sz.preparar_noticias(noticias_exemplo(30))
    item = sz.resumir_recorte(df, GeradorFalso(), local="Manaus", max_noticias=25)
    assert item["estatisticas"]["n_noticias"] == 25 and len(item["fontes"]) == 25
    assert item["boletim"]["formato_ok"]
    assert item["processamento"]["n_chamadas"] == 3  # 2 lotes + reduce
    with pytest.raises(ValueError):
        sz.resumir_recorte(df, GeradorFalso(), local="Recife")


# ---------------------------------------------------------------- API
class LLMFalso:
    repo, formato = "modelo/falso", "teste-cpu"

    def __init__(self, gerar=None, erro=None):
        self.gerar, self.erro = gerar or GeradorFalso(), erro

    def gerar_texto(self, prompt, max_tokens):
        if self.erro:
            raise self.erro
        return self.gerar(prompt, max_tokens)


@pytest.fixture
def api(monkeypatch):
    df = sz.preparar_noticias(noticias_exemplo(45))
    monkeypatch.setattr(sz, "carregar_noticias", lambda: df)
    monkeypatch.setattr(api_main, "llm", LLMFalso())
    return TestClient(api_main.app)


def test_rota_recortes(api):
    r = api.get("/noticias/recortes").json()
    assert r["total"] == 46 and r["locais"]["Manaus"] == 45 and "2026-10" in r["meses"]


def test_rota_sumarizar_devolve_boletim_estruturado(api):
    r = api.post("/sumarizar_noticias", json={"local": "Manaus", "max_noticias": 10})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["descricao"] == "em Manaus"
    assert corpo["boletim"]["formato_ok"] and len(corpo["boletim"]["pontos"]) == 3
    assert corpo["estatisticas"]["n_noticias"] == 10 and len(corpo["fontes"]) == 10
    assert corpo["processamento"]["modelo"] == "modelo/falso"
    assert 0 <= corpo["fidelidade"]["taxa"] <= 1


def test_rota_sumarizar_recorte_vazio_vira_404(api):
    r = api.post("/sumarizar_noticias", json={"local": "Recife"})
    assert r.status_code == 404 and r.json()["erro"] == "sem_noticias"
    assert "/noticias/recortes" in r.json()["sugestao"]


@pytest.mark.parametrize("corpo, campo", [({"mes": "outubro"}, "mes"),
                                          ({"max_noticias": 500}, "max_noticias")])
def test_rota_sumarizar_valida_filtros(api, corpo, campo):
    r = api.post("/sumarizar_noticias", json=corpo)
    assert r.status_code == 422 and r.json()["campos"][0]["campo"] == campo


def test_rota_sumarizar_sem_modelo_vira_503(api, monkeypatch):
    monkeypatch.setattr(api_main, "llm", LLMFalso(erro=llm.LLMModeloNaoEncontrado("x")))
    r = api.post("/sumarizar_noticias", json={})
    assert r.status_code == 503 and r.json()["erro"] == "llm_modelo_nao_encontrado"


def test_rota_sumarizar_boletim_vazio_vira_502(api, monkeypatch):
    monkeypatch.setattr(api_main, "llm", LLMFalso(gerar=GeradorFalso(boletim="   ")))
    r = api.post("/sumarizar_noticias", json={"max_noticias": 5})
    assert r.status_code == 502 and r.json()["erro"] == "resposta_invalida_do_modelo"


# ---------------------------------------------------------------- correções da rodada 2
class GeradorFatosUnicos(GeradorFalso):
    """Cada chamada devolve 3 fatos diferentes, como um modelo de verdade faria."""

    def __call__(self, prompt, max_tokens):
        self.prompts.append(prompt)
        if "Escreva um boletim" in prompt:
            texto = self.boletim
        else:
            n = len(self.prompts)
            texto = "\n".join(f"- fato {n}.{i}" for i in range(3))
        return {"resultado": texto, "tokens_gerados": 10, "tempo_s": 0.1}


def test_muitos_fatos_sao_consolidados_antes_do_boletim():
    """233 manchetes = 12 lotes = 36 fatos: consolida em grupos de 12 antes do boletim."""
    gerar = GeradorFatosUnicos()
    r = sz.sumarizar([f"manchete {i}" for i in range(233)], gerar)
    consolidacoes = [p for p in gerar.prompts if p.startswith("Abaixo estão fatos") and
                     "Junte esses fatos" in p]
    assert r.n_lotes == 12 and len(consolidacoes) == 3          # 36 fatos -> 3 grupos
    assert r.n_chamadas == 12 + 3 + 1
    final = gerar.prompts[-1]
    assert "Escreva um boletim" in final and final.count("\n- fato") <= sz.MAX_FATOS_REDUCE


def test_fatos_repetidos_sao_removidos():
    assert sz._sem_repetir(["Fumaça em Manaus.", "fumaça em manaus", "Aulas suspensas"]) == \
        ["Fumaça em Manaus.", "Aulas suspensas"]


def test_recomendacao_antes_dos_pontos_e_lista_longa_cortada():
    b = sz.interpretar_boletim("Resumo: Ar péssimo.\nRecomendação: Péssima\nPontos principais:\n"
                               "- a\n- b\n- c\n- d\n- e")
    assert b.formato_ok and b.categoria_recomendacao == "Péssima" and b.pontos == ["a", "b", "c"]


def test_orientacoes_no_prompt_nao_sao_numeradas():
    """Rodada 2: orientações 1–5 ao lado de manchetes 1–12 confundiram o modelo."""
    assert not any(linha[:2] in {"1.", "2.", "3.", "4.", "5."}
                   for linha in sz.ORIENTACOES.splitlines())


@pytest.mark.parametrize("categoria, sentimento, esperado", [
    ("Boa", {"negativo": 10, "positivo": 2}, False),         # caso da rodada 2 (setembro)
    ("Péssima", {"negativo": 10, "positivo": 2}, True),
    ("Péssima", {"positivo": 9, "negativo": 1}, False),
    ("Ruim", {"negativo": 5}, True),
    ("Boa", None, None), (None, {"negativo": 3}, None),
])
def test_coerencia_da_recomendacao_com_o_sentimento(categoria, sentimento, esperado):
    assert sz.recomendacao_coerente(categoria, sentimento) is esperado


def test_preparo_remove_paginas_aqi_do_iqair():
    df = pd.DataFrame([
        {"titulo": "Índice de Qualidade do Ar (AQI) e informações sobre Poluição do Ar de India"
                   " | IQAir Brasil", "fonte": "IQAir", "data_publicacao": "2026-10-01 10:00",
         "link": None, "resumo": "", "termo_busca": "x", "coletado_em": "2026-10-19"},
        {"titulo": "Como três grandes estudos mudaram o que sabemos sobre a poluição do ar",
         "fonte": "IQAir", "data_publicacao": "2026-10-01 10:00", "link": None, "resumo": "",
         "termo_busca": "x", "coletado_em": "2026-10-19"},
    ])
    assert sz.preparar_noticias(df)["titulo"].str.startswith("Como três").tolist() == [True]
