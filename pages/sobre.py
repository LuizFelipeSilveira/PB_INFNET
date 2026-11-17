"""Página: contexto do projeto."""
from __future__ import annotations

import streamlit as st

st.title("ℹ️ Sobre")

st.markdown("""
### O problema
Muitas pessoas não sabem avaliar se o ar da sua cidade está adequado para atividades externas
e correm riscos desnecessários de saúde em dias de alta poluição.

### Nossos objetivos
Combinar dados de poluição em tempo real (API OpenWeather) e a tabela oficial do IQAr
(extraída por scraping do PDF do Ministério do Meio Ambiente) para traduzir índices complexos
em recomendações simples e diretas.

### Como o índice é calculado
Seguimos a orientação técnica do MMA para o IQAr (Resolução CONAMA 506/2024): a concentração é
arredondada, cada poluente recebe um índice por interpolação linear entre as faixas oficiais, e o
índice geral é o pior entre eles. As faixas
oficiais usam médias de 1 h, 8 h ou 24 h; aqui usamos a leitura horária, por isso o valor é uma
aproximação educativa e não substitui boletins oficiais.

### Links e inspirações
* [Agenda 2030 (ODS 3 e 11)](https://brasil.un.org/pt-br/sdgs)
* [Índice de Qualidade do Ar - MMA](https://www.gov.br/mma/pt-br/assuntos/meio-ambiente-urbano-recursos-hidricos-qualidade-ambiental/qualidade-do-ar/indice-de-qualidade-do-ar-iqar)
* [Qualidade do ar - CETESB](https://cetesb.sp.gov.br/ar/)
* [OpenWeather Air Pollution API](https://openweathermap.org/api/air-pollution)
""")
