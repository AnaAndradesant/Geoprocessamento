import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard - Ana Andrade", layout="wide")


# ==========================================
# BARRA LATERAL (SIDEBAR) - CONTATOS
# ==========================================
st.sidebar.markdown("---")

gmail_logo_url = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg"
linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"

# IMPORTANTE: O HTML abaixo está sem indentação na esquerda para não bugar no Streamlit
html_contato_novo = f"""
<div style="text-align: center;">
<p style="font-size: 12px; color: #888; margin-bottom: 2px; margin-top: 10px;">Desenvolvido por</p>
<h2 style="font-family: serif; font-size: 20px; margin-top: 0px; margin-bottom: 2px; color: inherit;">ANA ANDRADE</h2>
<p style="font-size: 12px; color: #777; margin-top: 0px; margin-bottom: 15px;">Especialista em Geoprocessamento</p>
<hr style="border: 0; border-top: 1px solid #e0e0e0; margin-bottom: 15px;">
<div style="display: flex; justify-content: space-between; gap: 10px;">
<a href="https://mail.google.com/mail/?view=cm&fs=1&to=anacarolinasantos580@gmail.com" target="_blank" style="flex: 1; text-decoration: none; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; justify-content: center; align-items: center;">
<img src="{gmail_logo_url}" alt="Gmail Logo" style="width: 26px; height: auto; display: block; margin: 0 auto;">
</a>
<a href="https://www.linkedin.com/in/ana-carolina-santos-3920931b3" target="_blank" style="flex: 1; text-decoration: none; background-color: #0077B5; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: flex; justify-content: center; align-items: center;">
<img src="{linkedin_logo_url}" alt="LinkedIn Logo" style="width: 22px; height: auto; display: block; margin: 0 auto;">
</a>
</div>
</div>
"""
st.sidebar.markdown(html_contato_novo, unsafe_allow_html=True)


# ==========================================
# PÁGINA PRINCIPAL - MAPA
# ==========================================
st.title("🗺️ Análise Espacial")
st.markdown("Explore os dados no mapa interativo. Use o ícone de **camadas no canto superior direito do mapa** para alterar o estilo visual.")

# 1. Cria o mapa base (Aqui centralizei no Brasil como exemplo)
mapa = folium.Map(location=[-15.793889, -47.882778], zoom_start=4)

# 2. Adiciona os estilos de mapa que o usuário vai poder escolher
# Estilo 1: Claridade e minimalismo (O que você queria!)
folium.TileLayer(
    tiles='CartoDB positron',
    name='Estilo Claro (Clean)'
).add_to(mapa)

# Estilo 2: Mapa escuro (Fica super elegante)
folium.TileLayer(
    tiles='CartoDB dark_matter',
    name='Estilo Escuro'
).add_to(mapa)

# Estilo 3: O padrão tradicional
folium.TileLayer(
    tiles='OpenStreetMap',
    name='Padrão (Ruas)'
).add_to(mapa)

# 3. Adiciona o controle de camadas (Cria o botão mágico no mapa)
folium.LayerControl().add_to(mapa)

# 4. Renderiza o mapa grande na tela principal do Streamlit
st_folium(mapa, width=1000, height=600, returned_objects=[])
