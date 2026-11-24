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

RESUMO_OK = "Fumaça de queimadas piorou o ar em Manaus. Escolas suspenderam aulas."


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
    """Registra os prompts; cada chamada de fatos devolve 3 fatos diferentes."""

    def __init__(self, resumo: str = RESUMO_OK):
        self.prompts, self.resumo = [], resumo

    def __call__(self, prompt: str, max_tokens: int) -> dict:
        self.prompts.append(prompt)
        if "Escreva um resumo" in prompt:
            texto = self.resumo
        else:
            n = len(self.prompts)
            texto = "\n".join(f"- fato {n}.{i}" for i in range(3))
        return {"resultado": texto, "tokens_gerados": 10, "tempo_s": 0.1}


def tipos(gerar) -> list[str]:
    return ["fatos" if "Liste os 3 fatos" in p else "consolidar" if "Junte esses" in p
            else "resumo" for p in gerar.prompts]


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


# ---------------------------------------------------------------- cadeia de prompts
def test_um_lote_fatos_e_resumo():
    gerar = GeradorFalso()
    r = sz.sumarizar([f"manchete {i}" for i in range(12)], gerar)
    assert tipos(gerar) == ["fatos", "resumo"] and r.n_chamadas == 2
    assert "1. manchete 0" in gerar.prompts[0]  # manchetes numeradas no prompt
    assert r.boletim.resumo == RESUMO_OK and len(r.boletim.pontos) == 3 and r.boletim.formato_ok


def test_varios_lotes_consolidam_ate_3_pontos():
    gerar = GeradorFalso()
    r = sz.sumarizar([f"manchete {i}" for i in range(45)], gerar)   # 3 lotes = 9 fatos
    assert tipos(gerar) == ["fatos"] * 3 + ["consolidar", "resumo"]
    assert r.n_lotes == 3 and len(r.boletim.pontos) == 3 and r.boletim.formato_ok


def test_muitos_fatos_consolidam_em_arvore():
    """231 manchetes = 12 lotes = 36 fatos -> 3 grupos de 12 -> 9 fatos -> 3 pontos."""
    gerar = GeradorFalso()
    r = sz.sumarizar([f"manchete {i}" for i in range(231)], gerar)
    assert tipos(gerar) == ["fatos"] * 12 + ["consolidar"] * 4 + ["resumo"]
    assert r.n_chamadas == 17
    assert gerar.prompts[-1].count("\n- fato") <= sz.MAX_FATOS_REDUCE  # resumo recebe <= 12


def test_cada_prompt_pede_uma_tarefa_so():
    """Rodadas 1–3: pedir resumo, lista e escolha juntos quebrava o formato."""
    assert "Recomendação" not in sz.PROMPT_FATOS + sz.PROMPT_CONSOLIDAR + sz.PROMPT_RESUMO
    assert "sem lista" in sz.PROMPT_RESUMO


def test_resumo_em_lista_e_sinalizado():
    gerar = GeradorFalso(resumo="- Manaus teve fumaça\n- Aulas suspensas")
    r = sz.sumarizar([f"manchete {i}" for i in range(5)], gerar)
    assert not r.boletim.formato_ok and r.boletim.resumo == "Manaus teve fumaça Aulas suspensas"


def test_resumo_tira_rotulo():
    assert sz.limpar_resumo("**Resumo:** Ar ruim em Manaus.") == ("Ar ruim em Manaus.", True)


def test_fatos_repetidos_e_sobre_numeracao_sao_removidos():
    """Rodada 3: o modelo escreveu "Manchetes 3, 4, 5 ... citam Santarém" como se fosse fato."""
    assert sz._sem_repetir(["Fumaça em Manaus.", "fumaça em manaus", "Aulas suspensas",
                            "Manchetes 3, 4, 5 citam Santarém"]) == \
        ["Fumaça em Manaus.", "Aulas suspensas"]


# ---------------------------------------------------------------- recomendação por código
@pytest.mark.parametrize("manchetes, esperado", [
    (["Manaus tem poluição acima do que é considerado péssimo", "Fumaça em Manaus"],
     ("Péssima", 1)),                                                 # caso de setembro
    (["Ar insalubre em Itaituba", "Qualidade do ar crítica em Santarém", "ar perigoso"] +
     ["outra"] * 7, ("Ruim", 2)),             # Péssima só 1 de 10 (< limiar 2): vale Ruim
    (["Ar perigoso em Manaus", "Nível péssimo", "Qualidade muito insalubre"] + ["x"] * 7,
     ("Péssima", 2)),
    (["Sem categoria aqui"], (None, 0)),
])
def test_categoria_noticiada(manchetes, esperado):
    assert sz.categoria_noticiada(manchetes) == esperado


def test_resumir_recorte_traz_orientacao_oficial_com_evidencia():
    df = sz.preparar_noticias(pd.DataFrame([
        {"titulo": t, "fonte": "G1", "data_publicacao": "2026-10-10 10:00", "link": None,
         "resumo": "", "termo_busca": "x", "coletado_em": "2026-10-19"}
        for t in ["Ar perigoso em Manaus", "Nível péssimo em Manaus", "Fumaça em Manaus"]]))
    b = sz.resumir_recorte(df, GeradorFalso())["boletim"]
    assert b["categoria_recomendacao"] == "Péssima" and b["manchetes_que_citam_categoria"] == 2
    assert b["recomendacao"] == sz.RECOMENDACOES["Péssima"]  # texto da norma, não gerado


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
    assert item["processamento"]["n_chamadas"] == 4  # 2 lotes + consolidação + resumo
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
    monkeypatch.setattr(api_main, "llm", LLMFalso(gerar=GeradorFalso(resumo="   ")))
    r = api.post("/sumarizar_noticias", json={"max_noticias": 5})
    assert r.status_code == 502 and r.json()["erro"] == "resposta_invalida_do_modelo"


# ---------------------------------------------------------------- coerência e filtro
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


def test_sem_fatos_do_modelo_usa_manchetes_mas_sinaliza():
    class SemLista(GeradorFalso):
        def __call__(self, prompt, max_tokens):
            self.prompts.append(prompt)
            return {"resultado": "texto sem marcadores", "tokens_gerados": 3, "tempo_s": 0.1}
    r = sz.sumarizar(["Fumaça em Manaus", "Aulas suspensas", "Chuva melhora o ar"], SemLista())
    assert r.boletim.pontos == ["Fumaça em Manaus", "Aulas suspensas", "Chuva melhora o ar"]
    assert not r.boletim.formato_ok
