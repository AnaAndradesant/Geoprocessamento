import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geobr import read_state, read_biomes, read_municipality, read_indigenous_land, read_conservation_units 
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import requests, warnings, time, unicodedata, re, json
import ee

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor de Queimadas Brasil", page_icon="🔥", layout="wide")

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- AUTENTICAÇÃO DO EARTH ENGINE ---
try:
    key_dict = json.loads(st.secrets["EARTHENGINE_KEY"])
    credentials = ee.ServiceAccountCredentials(email=key_dict['client_email'], key_data=st.secrets["EARTHENGINE_KEY"])
    ee.Initialize(credentials, project='ee-anacarolinasantos580')
except Exception as e:
    st.error("⚠️ Erro ao conectar com o Google Earth Engine.")

def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=0.7):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = map_id_dict['tile_fetcher'].url_format
        folium.raster_layers.TileLayer(tiles=tiles_url, attr='Google Earth Engine', name=name, overlay=True, control=True, show=show, opacity=opacity).add_to(self)
    except: pass

folium.Map.add_ee_layer = add_ee_layer

# --- FUNÇÕES AUXILIARES ---
@st.cache_data(show_spinner=False)
def carregar_fronteira(tipo, estado, bioma, municipio):
    if tipo == "Por Estado": limite = read_state(code_state=estado, year=2020)
    elif tipo == "Por Bioma":
        limite = read_biomes(year=2019)
        limite = limite[limite['name_biome'] == bioma]
    else:
        limite = read_municipality(code_muni=estado, year=2020)
        limite = limite[limite['name_muni'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.lower().str.contains(municipio.lower())]
    return limite.to_crs("EPSG:4326")

@st.cache_data(show_spinner=False)
def carregar_areas_protegidas(tipo_area):
    if tipo_area == "Terras Indígenas":
        gdf = read_indigenous_land().rename(columns={'terrai_nom': 'nome_area'})
    else:
        gdf = read_conservation_units().rename(columns={'name_conservation_unit': 'nome_area'})
    return gdf.to_crs("EPSG:4326")

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    # Lógica de filtro simplificada para evitar erros de f-string
    bioma_filtro = val_bioma.replace("Amazônia", "Amaz%nia").replace("Mata Atlântica", "Mata Atl%ntica") if tipo == "Por Bioma" else ""
    cql = f"data_hora_gmt >= '{d_ini}T00:00:00' AND data_hora_gmt <= '{d_fim}T23:59:59' AND satelite IN ('{str(satelites)[1:-1]}')"
    # ... (Restante da lógica WFS do INPE)
    # Por brevidade, assumimos a função de busca que já funciona bem no seu app

# --- INTERFACE LATERAL ---
st.sidebar.title("⚙️ Filtros")
tipo_analise = st.sidebar.radio('Escala:', ['Por Estado', 'Por Bioma', 'Por Município'], index=1)
bioma_dd = st.sidebar.selectbox('Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"])
area_protegida = st.sidebar.selectbox("🌳 Filtro de Risco:", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

ativar_modis = st.sidebar.checkbox("🗺️ Cicatrizes (MODIS $km^2$)", value=True)
ano_modis = st.sidebar.number_input("Ano MODIS", 2000, 2025, 2024)
mes_modis = st.sidebar.slider("Mês MODis", 1, 12, 8)

gerar = st.sidebar.button("▶️ Atualizar Dados", type="primary", use_container_width=True)

if gerar:
    limite = carregar_fronteira(tipo_analise, "MT", bioma_dd, "") # Simplificado para o exemplo
    ee_geom = ee.Geometry(limite.geometry.union_all().__geo_interface__)

    # --- PROCESSAMENTO MODIS ---
    total_km2_modis = 0
    df_modis_areas = pd.DataFrame()
    if ativar_modis:
        with st.spinner("Calculando quilômetros quadrados na nuvem..."):
            img = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(f'{ano_modis}-{mes_modis:02d}-01', f'{ano_modis}-{mes_modis:02d}-28').max().clip(ee_geom)
            # Cálculo em KM2 (pixelArea / 1.000.000)
            img_km2 = ee.Image.pixelArea().divide(1000000).updateMask(img.select('BurnDate').gt(0))
            
            stats = img_km2.reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom, scale=500, maxPixels=1e12, bestEffort=True).getInfo()
            total_km2_modis = round(stats.get('area', 0) if stats.get('area') else 0, 2)

            # Cruzamento com áreas de risco
            if area_protegida != "Nenhuma":
                gdf_ap = carregar_areas_protegidas(area_protegida)
                gdf_ap_foco = gpd.sjoin(gdf_ap, limite, predicate='intersects')
                
                features = [ee.Feature(ee.Geometry(row['geometry'].__geo_interface__), {'nome': row['nome_area']}) for _, row in gdf_ap_foco.iterrows()]
                fc_ap = ee.FeatureCollection(features)
                
                stats_ap = img_km2.reduceRegions(collection=fc_ap, reducer=ee.Reducer.sum(), scale=500).getInfo()
                recs = [{'Área': f['properties']['nome'], 'KM2': round(f['properties'].get('sum', 0), 2)} for f in stats_ap['features'] if f['properties'].get('sum', 0) > 0]
                df_modis_areas = pd.DataFrame(recs).sort_values(by='KM2', ascending=False)

    # --- DASHBOARD ---
    st.metric("Área Total Queimada (MODIS)", f"{total_km2_modis} km²")

    col1, col2 = st.columns(2)
    with col1:
        if not df_modis_areas.empty:
            st.plotly_chart(px.bar(df_modis_areas.head(10), x='KM2', y='Área', orientation='h', title=f"Top 10 {area_protegida} Afetadas ($km^2$)", color_discrete_sequence=['#e67e22'], template='plotly_dark'), use_container_width=True)
    with col2:
        # Gráfico de "pico" para o MODIS (Área por mês ou dia se disponível)
        st.info("📊 O gráfico de linha do MODIS representa a área acumulada no mês selecionado.")
        # Aqui você pode adicionar um gráfico de barras por dia se quiser detalhar o BurnDate

    # --- MAPA ---
    m = folium.Map(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google')
    if ativar_modis:
        folium.Map.add_ee_layer(m, img.select('BurnDate').updateMask(img.select('BurnDate').gt(0)), {'min':1, 'max':366, 'palette':['orange','red']}, 'Área Queimada MODIS')
    
    # Desenha APENAS as áreas protegidas que tiveram fogo
    if not df_modis_areas.empty:
        gdf_mapa = gdf_ap_foco[gdf_ap_foco['nome_area'].isin(df_modis_areas['Área'])]
        folium.GeoJson(gdf_mapa, style_function=lambda x: {'color': 'yellow', 'fillOpacity': 0.2}).add_to(m)

    st_folium(m, width=1200, height=600)
