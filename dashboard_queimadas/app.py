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
st.set_page_config(
    page_title="Monitor de Queimadas Brasil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- AUTENTICAÇÃO DO EARTH ENGINE (STREAMLIT CLOUD) ---
try:
    # 1. Lê a chave secreta que guardamos no Streamlit
    key_dict = json.loads(st.secrets["EARTHENGINE_KEY"])
    
    # 2. Cria a credencial especial para o robô
    credentials = ee.ServiceAccountCredentials(
        email=key_dict['client_email'], 
        key_data=st.secrets["EARTHENGINE_KEY"]
    )
    
    # 3. Inicializa o Earth Engine silenciosamente
    ee.Initialize(credentials, project='ee-anacarolinasantos580')
    
except Exception as e:
    st.error(f"⚠️ Erro ao conectar com o Google Earth Engine.")
    st.info("Por favor, verifique se a variável EARTHENGINE_KEY foi criada corretamente nas Settings > Secrets do Streamlit Cloud.")
    st.write(e)

# Método para o Folium renderizar GEE
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=0.7):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data © Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=show,
            opacity=opacity
        ).add_to(self)
    except:
        pass

folium.Map.add_ee_layer = add_ee_layer

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_cidades(uf):
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200: return sorted([d['nome'] for d in resp.json()])
    except: pass
    return ["Erro ao carregar cidades"]

def normalizar_texto(txt):
    if pd.isna(txt): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(txt)) if unicodedata.category(c) != 'Mn').lower()

@st.cache_data(show_spinner=False)
def carregar_fronteira(tipo, estado, bioma, municipio):
    if tipo == "Por Estado": limite = read_state(code_state=estado, year=2020)
    elif tipo == "Por Bioma":
        limite = read_biomes(year=2019)
        limite = limite[limite['name_biome'] == bioma]
    elif tipo == "Por Município":
        limite = read_municipality(code_muni=estado, year=2020)
        busca = normalizar_texto(municipio.strip())
        limite['nome_norm'] = limite['name_muni'].apply(normalizar_texto)
        limite = limite[limite['nome_norm'].str.contains(busca)]
        
    limite = limite.to_crs("EPSG:4326")
    limite['geometry'] = limite['geometry'].simplify(tolerance=0.005, preserve_topology=True)
    return limite

@st.cache_data(show_spinner=False)
def carregar_areas_protegidas(tipo_area):
    if tipo_area == "Terras Indígenas":
        gdf_areas = read_indigenous_land()
        if 'terrai_nom' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'terrai_nom': 'nome_area'})
    else:
        gdf_areas = read_conservation_units()
        if 'name_conservation_unit' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'name_conservation_unit': 'nome_area'})
    
    gdf_areas['geometry'] = gdf_areas['geometry'].make_valid()
    gdf_areas = gdf_areas.to_crs("EPSG:4326")
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    return gdf_areas[['nome_area', 'geometry']]

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    dic_estados = {"AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS", "BA": "BAHIA", "CE": "CEAR%", "DF": "DISTRITO FEDERAL", "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O", "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS", "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%", "PE": "PERNAMBUCO", "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE", "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA", "SC": "SANTA CATARINA", "SP": "S%O PAULO", "SE": "SERGIPE", "TO": "TOCANTINS"}

    if tipo == "Por Estado": filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'"
    elif tipo == "Por Bioma":
        tradutor = {"Amazônia": "Amaz%nia", "Mata Atlântica": "Mata Atl%ntica"}
        filtro_base = f"bioma ILIKE '{tradutor.get(val_bioma, val_bioma)}'"
    elif tipo == "Por Município":
        muni_curinga = re.sub(r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni).replace(' ', '%')
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{muni_curinga}%'"

    dt_ini = datetime.strptime(d_ini, "%Y-%m-%d")
    dt_fim = datetime.strptime(d_fim, "%Y-%m-%d")
    all_dfs = []
    sat_str = "','".join(satelites)
    
    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=5), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('{sat_str}') AND pais_complete_id=33 AND {filtro_base}"
        try:
            r = requests.get(url, params={"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": "bdqueimadas:focos", "outputFormat": "application/json", "CQL_FILTER": cql, "maxFeatures": 10000}, verify=False, timeout=60)
            if r.status_code == 200:
                features = r.json().get("features", [])
                if features:
                    registros = []
                    for f in features:
                        props = f["properties"]
                        props["longitude"], props["latitude"] = f["geometry"]["coordinates"]
                        registros.append(props)
                    all_dfs.append(pd.DataFrame(registros))
        except: pass
        dt_ini = dt_bloco_fim + timedelta(days=1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- INTERFACE (BARRA LATERAL) ---
st.sidebar.title("⚙️ Filtros da Análise")
tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)

estado_dd = st.sidebar.selectbox('Selecione o Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25, disabled=(tipo_analise == 'Por Bioma'))
bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))
cidades_lista = buscar_cidades(estado_dd)
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', cidades_lista, disabled=(tipo_analise != 'Por Município'))

st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Fontes de Dados")

ativar_inpe = st.sidebar.checkbox("🔥 Focos de Calor (INPE)", value=True)
if ativar_inpe:
    unidade_dd = st.sidebar.selectbox("Analisar tempo por:", ["Dias", "Meses", "Anos"], index=1)
    if unidade_dd == "Dias": op_qtd = list(range(1, 91))
    elif unidade_dd == "Meses": op_qtd = list(range(1, 61))
    else: op_qtd = list(range(1, 11))
    quantidade_sel = st.sidebar.selectbox(f"Quantidade de {unidade_dd}:", options=op_qtd, index=1)
    satelites_sel = st.sidebar.multiselect("Satélites:", ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03'], default=['AQUA_M-T', 'NPP-375', 'NPP-375D'])

st.sidebar.markdown("")
ativar_modis = st.sidebar.checkbox("🗺️ Cicatrizes (NASA MODIS)", value=True)
if ativar_modis:
    ano_modis = st.sidebar.selectbox("Ano (MODIS):", list(range(2001, datetime.now().year + 1)), index=datetime.now().year - 2002)
    mes_modis = st.sidebar.selectbox("Mês (MODIS):", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("Análise de Risco:", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

# --- TELA PRINCIPAL ---
st.title("🔥 Dashboard Integrado 🔥")

if gerar:
    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")
    df_rec = pd.DataFrame()
    areas_afetadas = gpd.GeoDataFrame()

    with st.status(f"🛰️ Processando: **{val_sel}**", expanded=True) as status:
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        
        if ativar_inpe:
            hoje = datetime.now()
            if unidade_dd == "Dias": dt_ini = hoje - timedelta(days=quantidade_sel)
            elif unidade_dd == "Meses": dt_ini = hoje - timedelta(days=30*quantidade_sel)
            else: dt_ini = hoje - timedelta(days=365*quantidade_sel)
            df = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel)
            
            if not df.empty:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                gdf = gpd.sjoin(gdf, limite, predicate="within")
                df_rec = pd.DataFrame(gdf.drop(columns="geometry"))

                if not df_rec.empty and area_protegida != "Nenhuma":
                    gdf_areas = carregar_areas_protegidas(area_protegida).to_crs(gdf.crs)
                    gdf_focos_risco = gpd.sjoin(gdf.drop(columns=['index_right', 'index_left'], errors='ignore'), gdf_areas, predicate='within')
                    if not gdf_focos_risco.empty:
                        areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])]
                        st.error(f"🚨 ALERTA: {len(gdf_focos_risco)} focos em áreas protegidas!")
            
        status.update(label="✅ Consultas finalizadas!", state="complete", expanded=False)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        centro = limite.geometry.union_all().centroid
        m = folium.Map(location=[centro.y, centro.x], zoom_start=9, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
        folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
        
        # Camada MODIS (Earth Engine)
        if ativar_modis:
            try:
                ee_geom = ee.Geometry(limite.geometry.union_all().__geo_interface__)
                data_ini = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                data_fim = data_ini.advance(1, 'month')
                area_queimada = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ini, data_fim).select('BurnDate').max().clip(ee_geom)
                m.add_ee_layer(area_queimada, {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']}, 'MODIS')
            except Exception as e:
                st.error(f"Erro MODIS: {e}")
        
        # Camada INPE
        if ativar_inpe and not df_rec.empty:
            if not areas_afetadas.empty:
                folium.GeoJson(areas_afetadas.__geo_interface__, style_function=lambda x: {'fillColor': '#e74c3c', 'color': '#c0392b', 'weight': 2, 'fillOpacity': 0.4}).add_to(m)
            HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15, blur=20).add_to(m)
            
        st_folium(m, width=700, height=700, returned_objects=[])
