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
    initial_sidebar_state="expanded",
    menu_items={'About': "### 🛰️ Monitor de Queimadas Integrado\nDesenvolvido por Ana Carolina Andrade."}
)

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- AUTENTICAÇÃO DO EARTH ENGINE ---
try:
    key_dict = json.loads(st.secrets["EARTHENGINE_KEY"])
    credentials = ee.ServiceAccountCredentials(
        email=key_dict['client_email'], 
        key_data=st.secrets["EARTHENGINE_KEY"]
    )
    ee.Initialize(credentials, project='ee-anacarolinasantos580')
except Exception as e:
    st.error("⚠️ Erro ao conectar com o Google Earth Engine. Verifique seus Secrets.")

def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=0.7):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = map_id_dict.get('tile_fetcher', {}).url_format if 'tile_fetcher' in map_id_dict else map_id_dict.get('urlFormat', map_id_dict.get('url_format', ''))
        folium.raster_layers.TileLayer(
            tiles=tiles_url, attr='Map Data © Google Earth Engine', name=name,
            overlay=True, control=True, show=show, opacity=opacity
        ).add_to(self)
    except Exception as e:
        st.error(f"🚨 Erro crítico ao desenhar a camada do Earth Engine: {e}")

folium.Map.add_ee_layer = add_ee_layer

# --- FUNÇÕES COM CACHE ---
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
        gdf_areas = gdf_areas.rename(columns={'terrai_nom': 'nome_area'}) if 'terrai_nom' in gdf_areas.columns else gdf_areas
    else:
        gdf_areas = read_conservation_units()
        gdf_areas = gdf_areas.rename(columns={'name_conservation_unit': 'nome_area'}) if 'name_conservation_unit' in gdf_areas.columns else gdf_areas
    
    gdf_areas['geometry'] = gdf_areas['geometry'].make_valid()
    gdf_areas = gdf_areas.to_crs("EPSG:4326")
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    return gdf_areas[['nome_area', 'geometry']]

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    dic_estados = {"AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS", "BA": "BAHIA", "CE": "CEAR%", "DF": "DISTRITO FEDERAL", "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O", "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS", "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%", "PE": "PERNAMBUCO", "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE", "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA", "SC": "SANTA CATARINA", "SP": "S%O PAULO", "SE": "SERGIPE", "TO": "TOCANTINS"}

    filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'" if tipo == "Por Estado" else \
                  f"bioma ILIKE '{{'Amazônia': 'Amaz%nia', 'Mata Atlântica': 'Mata Atl%ntica'}.get(val_bioma, val_bioma)}'" if tipo == "Por Bioma" else \
                  f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{re.sub(r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni).replace(' ', '%')}%'"

    dt_ini, dt_fim = datetime.strptime(d_ini, "%Y-%m-%d"), datetime.strptime(d_fim, "%Y-%m-%d")
    all_dfs = []
    
    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=5), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('{'\' , \''.join(satelites)}') AND pais_complete_id=33 AND {filtro_base}"
        try:
            r = requests.get(url, params={"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": "bdqueimadas:focos", "outputFormat": "application/json", "CQL_FILTER": cql, "maxFeatures": 10000}, verify=False, timeout=60)
            if r.status_code == 200 and r.json().get("features"):
                registros = [{"longitude": f["geometry"]["coordinates"][0], "latitude": f["geometry"]["coordinates"][1], **f["properties"]} for f in r.json()["features"]]
                all_dfs.append(pd.DataFrame(registros))
        except: pass
        dt_ini = dt_bloco_fim + timedelta(days=1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- INTERFACE (BARRA LATERAL) ---
st.sidebar.title("⚙️ Filtros da Análise")

tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)
estado_dd = st.sidebar.selectbox('Selecione o Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25, disabled=(tipo_analise == 'Por Bioma'))
bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', buscar_cidades(estado_dd), disabled=(tipo_analise != 'Por Município'))

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Fontes de Dados")

unidade_dd, quantidade_sel, satelites_sel = "Meses", 1, []
ativar_inpe = st.sidebar.checkbox("🔥 Focos de Calor (INPE)", value=True)

if ativar_inpe:
    with st.sidebar.expander("⏱️ Filtros de Tempo (INPE)", expanded=True):
        unidade_dd = st.selectbox("Analisar por:", ["Dias", "Meses", "Anos"], index=1)
        quantidade_sel = st.selectbox(f"Quantidade:", options=list(range(1, 91)) if unidade_dd == "Dias" else list(range(1, 61)) if unidade_dd == "Meses" else list(range(1, 11)), index=1)
        satelites_sel = st.multiselect("Satélites:", ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03'], default=['AQUA_M-T', 'NPP-375', 'NPP-375D'])

ativar_modis = st.sidebar.checkbox("🗺️ Cicatrizes (NASA MODIS)", value=True)
if ativar_modis:
    with st.sidebar.expander("📅 Filtros de Data (MODIS)", expanded=True):
        ano_modis = st.selectbox("Ano:", list(range(2001, datetime.now().year + 1)), index=datetime.now().year - 2002)
        mes_modis = st.selectbox("Mês:", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("🌳 Análise de Risco Espacial:", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])
gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

# --- INTERFACE PRINCIPAL ---
st.title("🔥 Dashboard Integrado 🔥")

if gerar:
    if not ativar_inpe and not ativar_modis:
        st.error("⚠️ Selecione pelo menos uma Fonte de Dados na barra lateral.")
        st.stop()

    hoje = datetime.now()
    dt_ini = hoje - timedelta(days=quantidade_sel if unidade_dd == "Dias" else 30*quantidade_sel if unidade_dd == "Meses" else 365*quantidade_sel)
    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.status(f"🛰️ Processando dados para: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando fronteiras geográficas oficiais...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        geom_unida = limite.geometry.union_all()
        ee_geom_complex = ee.Geometry(geom_unida.__geo_interface__)

        df_inpe_rec = pd.DataFrame()
        if ativar_inpe:
            st.write("📡 Consultando base de dados do INPE...")
            df = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel)
            if not df.empty:
                gdf_inpe = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                df_inpe_rec = pd.DataFrame(gpd.sjoin(gdf_inpe, limite, predicate="within").drop(columns="geometry"))

        # ESTATÍSTICAS MODIS
        total_ha_modis = 0
        df_modis_areas = pd.DataFrame()
        area_queimada_img = None
        
        if ativar_modis:
            st.write("☁️ Calculando estatísticas na nuvem do Google Earth Engine...")
            try:
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao_modis = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ini_ee, data_ini_ee.advance(1, 'month')).filterBounds(ee_geom_complex)
                
                if colecao_modis.size().getInfo() > 0:
                    area_queimada_img = colecao_modis.select('BurnDate').max().clip(ee_geom_complex)
                    mask_queimada = area_queimada_img.gt(0)
                    
                    # Cria imagem onde o valor do pixel é a área dele em hectares
                    img_area_ha = ee.Image.pixelArea().divide(10000).updateMask(mask_queimada).rename('area_ha')
                    
                    # Calcula o total na região toda
                    estatistica_total = img_area_ha.reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom_complex, scale=500, maxPixels=1e9).getInfo()
                    total_ha_modis = round(estatistica_total.get('area_ha', 0), 2)
            except Exception as e:
                st.error(f"Erro ao calcular estatísticas MODIS: {e}")

        # CRUZAMENTO COM ÁREAS PROTEGIDAS
        focos_em_areas = pd.DataFrame()
        areas_afetadas = gpd.GeoDataFrame()
        
        if area_protegida != "Nenhuma":
            st.write(f"🌳 Intersectando com {area_protegida}...")
            gdf_areas_brasil = carregar_areas_protegidas(area_protegida)
            # Filtra só as áreas que tocam a nossa região de interesse para ficar mais rápido
            gdf_areas = gpd.sjoin(gdf_areas_brasil, limite, predicate='intersects').drop(columns=['index_right'])
            
            if not gdf_areas.empty:
                # 1. Cruzamento INPE
                if ativar_inpe and not df_inpe_rec.empty:
                    gdf_inpe_bound = gpd.GeoDataFrame(df_inpe_rec, geometry=gpd.points_from_xy(df_inpe_rec["longitude"], df_inpe_rec["latitude"]), crs="EPSG:4326")
                    gdf_focos_risco = gpd.sjoin(gdf_inpe_bound, gdf_areas, predicate='within')
                    if not gdf_focos_risco.empty:
                        focos_em_areas = pd.DataFrame(gdf_focos_risco.drop(columns="geometry"))
                        areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])]

                # 2. Cruzamento MODIS (Enviando as áreas para o GEE calcular Hectares)
                if ativar_modis and total_ha_modis > 0:
                    try:
                        features_ee = [ee.Feature(ee.Geometry(row['geometry'].__geo_interface__), {'nome_area': row['nome_area']}) for _, row in gdf_areas.iterrows()]
                        if features_ee:
                            fc_areas = ee.FeatureCollection(features_ee)
                            # Pede pro GEE somar os hectares queimados dentro de cada polígono
                            stats_modis = img_area_ha.reduceRegions(collection=fc_areas, reducer=ee.Reducer.sum(), scale=500).getInfo()
                            
                            records = []
                            for f in stats_modis['features']:
                                nome = f['properties'].get('nome_area')
                                ha = round(f['properties'].get('sum', 0), 2)
                                if ha > 0: records.append({'Área Protegida': nome, 'Hectares Queimados (MODIS)': ha})
                            
                            if records:
                                df_modis_areas = pd.DataFrame(records).sort_values(by='Hectares Queimados (MODIS)', ascending=False)
                                areas_afetadas_modis = gdf_areas[gdf_areas['nome_area'].isin(df_modis_areas['Área Protegida'])]
                                areas_afetadas = pd.concat([areas_afetadas, areas_afetadas_modis]).drop_duplicates(subset=['nome_area'])
                    except Exception as e:
                        st.write("⚠️ Alerta: Ocorreu uma limitação de processamento na nuvem ao calcular TIs/UCs no MODIS para esta região específica.")

        status.update(label="✅ Processamento concluído!", state="complete", expanded=False)

    # --- CARD DE RESUMO DINÂMICO ---
    texto_resumo = []
    if ativar_inpe: texto_resumo.append(f"🔥 {len(df_inpe_rec):,} Focos de Calor (INPE)")
    if ativar_modis: texto_resumo.append(f"🗺️ {total_ha_modis:,.2f} Hectares de Cicatrizes (MODIS)")
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 8px solid #ff4b4b; margin-bottom: 15px;">
        <h3 style="color: #c0392b; margin: 0;">{' | '.join(texto_resumo)}</h3>
        <p style="color: #636e72; margin: 4px 0 0 0;">Análise de {val_sel} | Risco: {area_protegida}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- ALERTAS DE RISCO E GRÁFICOS ---
    if not focos_em_areas.empty or not df_modis_areas.empty:
        st.error(f"🚨 **ALERTA CRÍTICO:** Fogo detectado DENTRO de áreas protegidas ({area_protegida})!")
        colA, colB = st.columns(2)
        
        # Ranking INPE
        with colA:
            if not focos_em_areas.empty:
                df_ranking_inpe = focos_em_areas['nome_area'].value_counts().reset_index().head(10)
                df_ranking_inpe.columns = ['Área Protegida', 'Focos (INPE)']
                fig_inpe = px.bar(df_ranking_inpe, x='Focos (INPE)', y='Área Protegida', orientation='h', text='Focos (INPE)', color='Focos (INPE)', color_continuous_scale=px.colors.sequential.Reds, title="🔥 Focos por Área (INPE)")
                fig_inpe.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=40, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_inpe, use_container_width=True)
            elif ativar_inpe:
                st.success("✅ INPE: Nenhum foco detectado nas áreas protegidas.")

        # Ranking MODIS
        with colB:
            if not df_modis_areas.empty:
                fig_modis = px.bar(df_modis_areas.head(10), x='Hectares Queimados (MODIS)', y='Área Protegida', orientation='h', text='Hectares Queimados (MODIS)', color='Hectares Queimados (MODIS)', color_continuous_scale=px.colors.sequential.Oranges, title="🗺️ Hectares Queimados por Área (MODIS)")
                fig_modis.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=40, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_modis, use_container_width=True)
            elif ativar_modis:
                st.success("✅ MODIS: Nenhuma cicatriz detectada nas áreas protegidas.")

    st.markdown("---") 

    # --- MAPA ESPACIAL ---
    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("🗺️ Análise Espacial")
        centro = limite.geometry.union_all().centroid
        m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite
