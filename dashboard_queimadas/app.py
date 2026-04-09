import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geobr import read_state, read_biomes, read_municipality, read_indigenous_land, read_conservation_units
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests, warnings, time, unicodedata, re, json, io
import ee

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Monitor de Queimadas Brasil - Full",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTADO DA SESSÃO ---
if 'gerar_dashboard' not in st.session_state:
    st.session_state.gerar_dashboard = False

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- AUTENTICAÇÃO DO EARTH ENGINE ---
try:
    if "EARTHENGINE_KEY" in st.secrets:
        key_dict = json.loads(st.secrets["EARTHENGINE_KEY"])
        credentials = ee.ServiceAccountCredentials(
            email=key_dict['client_email'],
            key_data=st.secrets["EARTHENGINE_KEY"]
        )
        ee.Initialize(credentials, project='ee-anacarolinasantos580')
    else:
        ee.Initialize(project='ee-anacarolinasantos580')
except Exception as e:
    st.error(f"⚠️ Erro ao conectar com o Google Earth Engine: {e}")

# --- FUNÇÃO PARA ADICIONAR CAMADAS EE AO FOLIUM ---
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1.0):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = map_id_dict['tile_fetcher'].url_format
        folium.raster_layers.TileLayer(
            tiles=tiles_url, attr='Map Data © Google Earth Engine', name=name,
            overlay=True, control=True, show=show, opacity=opacity
        ).add_to(self)
    except Exception as e:
        st.error(f"🚨 Erro ao carregar camada do satélite: {e}")

folium.Map.add_ee_layer = add_ee_layer

# =============================================================
# --- FUNÇÕES DE DADOS (CACHED) ---
# =============================================================

@st.cache_data(ttl=86400)
def buscar_cidades(uf):
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    try:
        resp = requests.get(url, timeout=5)
        return sorted([d['nome'] for d in resp.json()]) if resp.status_code == 200 else ["Erro"]
    except: return ["Erro"]

@st.cache_data(show_spinner=False)
def carregar_fronteira(tipo, estado, bioma, municipio):
    if tipo == "Por Estado":
        limite = read_state(code_state=estado, year=2020)
    elif tipo == "Por Bioma":
        limite = read_biomes(year=2019)
        limite = limite[limite['name_biome'] == bioma]
    elif tipo == "Por Município":
        limite = read_municipality(code_muni=estado, year=2020)
        busca = ''.join(c for c in unicodedata.normalize('NFD', municipio) if unicodedata.category(c) != 'Mn').lower()
        limite['n'] = limite['name_muni'].apply(lambda x: ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn').lower())
        limite = limite[limite['n'].str.contains(busca)]
    return limite.to_crs("EPSG:4326")

@st.cache_data(show_spinner=False)
def carregar_areas_protegidas(tipo_area):
    if tipo_area == "Terras Indígenas":
        gdf = read_indigenous_land()
        gdf = gdf.rename(columns={'terrai_nom': 'nome_area'})
    else:
        gdf = read_conservation_units()
        gdf = gdf.rename(columns={'name_conservation_unit': 'nome_area'})
    return gdf.to_crs("EPSG:4326")[['nome_area', 'geometry']]

@st.cache_data(ttl=3600)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    # Lógica de filtro simplificada para exemplo (usa as mesmas strings do código anterior)
    # ... (mesmo código de requisição WFS do INPE já fornecido)
    return pd.DataFrame() # Placeholder para o bloco completo do INPE

@st.cache_data(ttl=86400)
def calcular_anomalia_modis(geom_json_str, ano_ref):
    # Lógica do GEE para calcular média histórica vs ano atual
    # ... (mesmo código de anomalia já fornecido)
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def calcular_nbr_sentinel(geom_json_str, ano, mes):
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    data_ref = ee.Date.fromYMD(ano, mes, 1)
    data_pre = data_ref.advance(-3, 'month')
    data_pos = data_ref.advance(2, 'month')
    
    def mask_s2(img):
        qa = img.select('QA60')
        return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))).divide(10000)

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ee_geom).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2)
    pre = col.filterDate(data_pre, data_ref).median().clip(ee_geom)
    pos = col.filterDate(data_ref, data_pos).median().clip(ee_geom)
    
    nbr_pre = pre.normalizedDifference(['B8', 'B12'])
    nbr_pos = pos.normalizedDifference(['B8', 'B12'])
    dnbr = nbr_pre.subtract(nbr_pos)
    
    sev = dnbr.where(dnbr.lt(-0.1), 0).where(dnbr.gte(-0.1).And(dnbr.lt(0.1)), 1).where(dnbr.gte(0.1).And(dnbr.lt(0.27)), 2).where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 3).where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 4).where(dnbr.gte(0.66), 5)
    return dnbr, sev

# =============================================================
# --- INTERFACE LATERAL ---
# =============================================================
st.sidebar.title("🔥 Filtros Master")

tipo_analise = st.sidebar.radio('Escala:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)
estado_dd = st.sidebar.selectbox('Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25)
bioma_dd = st.sidebar.selectbox('Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))
municipio_dd = st.sidebar.selectbox('Município:', buscar_cidades(estado_dd), disabled=(tipo_analise != 'Por Município'))

st.sidebar.markdown("---")
fonte_escolhida = st.sidebar.radio(
    "Fonte de Dados:",
    ["🔥 Focos (INPE)", "🗺️ Área Queimada (NASA MODIS)", "🔬 Severidade dNBR (Sentinel-2)"]
)

if "INPE" in fonte_escolhida:
    unidade_dd = st.sidebar.selectbox("Período:", ["Dias", "Meses"], index=1)
    quantidade_sel = st.sidebar.slider("Quantidade:", 1, 12, 1)
    satelites_sel = st.sidebar.multiselect("Satélites:", ['AQUA_M-T', 'NPP-375', 'TERRA_M-T'], default=['AQUA_M-T'])
elif "MODIS" in fonte_escolhida:
    ano_modis = st.sidebar.number_input("Ano:", 2001, 2025, 2024)
    mes_modis = st.sidebar.slider("Mês:", 1, 12, 8)
    calc_anomalia = st.sidebar.checkbox("Calcular Anomalia Histórica", value=True)
else:
    ano_s2 = st.sidebar.number_input("Ano:", 2017, 2025, 2024)
    mes_s2 = st.sidebar.slider("Mês:", 1, 12, 9)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("Cruzar com Áreas Protegidas:", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

if st.sidebar.button("▶️ GERAR DASHBOARD COMPLETO", type="primary", use_container_width=True):
    st.session_state.gerar_dashboard = True

# =============================================================
# --- PROCESSAMENTO E VISUALIZAÇÃO ---
# =============================================================

if st.session_state.gerar_dashboard:
    with st.spinner("🚀 Processando todas as camadas selecionadas..."):
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        geom_unida = limite.geometry.union_all()
        geom_json = json.dumps(geom_unida.__geo_interface__)
        
        # Mapa Base
        centro = [limite.geometry.centroid.y.mean(), limite.geometry.centroid.x.mean()]
        m = folium.Map(location=centro, zoom_start=7, tiles="cartodbpositron")
        folium.GeoJson(limite, name="Limite", style_function=lambda x:{'fillColor': 'none', 'color': 'black'}).add_to(m)

        # LÓGICA POR FONTE
        if "INPE" in fonte_escolhida:
            # (Aqui entra a lógica de busca WFS do INPE e Heatmap já enviada)
            st.info("Visualizando Focos de Calor do INPE...")
            
        elif "MODIS" in fonte_escolhida:
            data_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
            img = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ee, data_ee.advance(1, 'month')).select('BurnDate').max().clip(ee.Geometry(geom_unida.__geo_interface__))
            m.add_ee_layer(img, {'min': 1, 'max': 366, 'palette': ['orange', 'red']}, "Área Queimada MODIS")
            
            if calc_anomalia:
                st.subheader("📊 Tabela de Anomalia (MODIS)")
                df_anomalia = calcular_anomalia_modis(geom_json, ano_modis)
                st.dataframe(df_anomalia, use_container_width=True)

        elif "Sentinel-2" in fonte_escolhida:
            dnbr, sev = calcular_nbr_sentinel(geom_json, ano_s2, mes_s2)
            paleta_sev = ['#006400', '#7cfc00', '#ffff00', '#ffa500', '#ff4500', '#8b0000']
            m.add_ee_layer(sev, {'min': 0, 'max': 5, 'palette': paleta_sev}, "Severidade dNBR")
            st.write("🔬 **Legenda Severidade:** Verde (Não afetado) → Vermelho Escuro (Alta)")

        # Áreas Protegidas
        if area_protegida != "Nenhuma":
            gdf_ap = carregar_areas_protegidas(area_protegida)
            # Filtra apenas áreas que intersectam o limite
            gdf_ap_f = gdf_ap[gdf_ap.intersects(geom_unida)]
            folium.GeoJson(gdf_ap_f, name=area_protegida, style_function=lambda x:{'fillColor': 'green', 'fillOpacity': 0.2}).add_to(m)
            st.write(f"🌳 {len(gdf_ap_f)} {area_protegida} monitoradas na região.")

        # Renderizar Mapa
        st_folium(m, width=1200, height=600)

# --- RODAPÉ DE CONTATO ---
st.sidebar.markdown("---")
st.sidebar.info("Desenvolvido por Ana Andrade")
