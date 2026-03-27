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
        if 'terrai_nom' in gdf_areas.columns: gdf_areas = gdf_areas.rename(columns={'terrai_nom': 'nome_area'})
    else:
        gdf_areas = read_conservation_units()
        if 'name_conservation_unit' in gdf_areas.columns: gdf_areas = gdf_areas.rename(columns={'name_conservation_unit': 'nome_area'})
    
    gdf_areas['geometry'] = gdf_areas['geometry'].make_valid()
    gdf_areas = gdf_areas.to_crs("EPSG:4326")
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    return gdf_areas[['nome_area', 'geometry']]

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    dic_estados = {"AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS", "BA": "BAHIA", "CE": "CEAR%", "DF": "DISTRITO FEDERAL", "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O", "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS", "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%", "PE": "PERNAMBUCO", "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE", "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA", "SC": "SANTA CATARINA", "SP": "S%O PAULO", "SE": "SERGIPE", "TO": "TOCANTINS"}

    if tipo == "Por Estado":
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'"
    elif tipo == "Por Bioma":
        bioma_busca = val_bioma
        if val_bioma == "Amazônia": bioma_busca = "Amaz%nia"
        elif val_bioma == "Mata Atlântica": bioma_busca = "Mata Atl%ntica"
        filtro_base = f"bioma ILIKE '{bioma_busca}'"
    else:
        muni_curinga = re.sub(r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni).replace(' ', '%')
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{muni_curinga}%'"

    dt_ini, dt_fim = datetime.strptime(d_ini, "%Y-%m-%d"), datetime.strptime(d_fim, "%Y-%m-%d")
    all_dfs = []
    sat_str = "','".join(satelites)
    
    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=5), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('{sat_str}') AND pais_complete_id=33 AND {filtro_base}"
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

# --- EXECUÇÃO PRINCIPAL ---
st.title("🔥 Dashboard Integrado 🔥")

if gerar:
    if not ativar_inpe and not ativar_modis:
        st.error("⚠️ Selecione pelo menos uma fonte de dados.")
        st.stop()

    hoje = datetime.now()
    if ativar_inpe:
        dt_ini = hoje - timedelta(days=quantidade_sel if unidade_dd == "Dias" else 30*quantidade_sel if unidade_dd == "Meses" else 365*quantidade_sel)
    
    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.status(f"🛰️ Processando: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando limites geográficos...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        geom_unida = limite.geometry.union_all()
        ee_geom_complex = ee.Geometry(geom_unida.__geo_interface__)

        df_inpe_rec = pd.DataFrame()
        if ativar_inpe:
            st.write("📡 Consultando INPE...")
            df = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel)
            if not df.empty:
                gdf_inpe = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                df_inpe_rec = pd.DataFrame(gpd.sjoin(gdf_inpe, limite, predicate="within").drop(columns="geometry"))

        total_km2_modis = 0
        df_modis_areas = pd.DataFrame()
        df_evolucao_modis = pd.DataFrame()
        area_queimada_img = None
        if ativar_modis:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ini_ee, data_ini_ee.advance(1, 'month')).filterBounds(ee_geom_complex)
                if colecao.size().getInfo() > 0:
                    area_queimada_img = colecao.select('BurnDate').max().clip(ee_geom_complex)
                    # Cálculo direto em KM2 (divide por 1.000.000)
                    img_area_km2 = ee.Image.pixelArea().divide(1000000).updateMask(area_queimada_img.gt(0)).rename('area_km2')
                    
                    stats_total = img_area_km2.reduceRegion(
                        reducer=ee.Reducer.sum(), 
                        geometry=ee_geom_complex, 
                        scale=500, 
                        maxPixels=1e13,
                        bestEffort=True 
                    ).getInfo()
                    total_km2_modis = round(stats_total.get('area_km2', 0) if stats_total.get('area_km2') else 0, 2)

                    # Dados para gráfico de evolução (pico) do MODIS baseados no BurnDate
                    hist = area_queimada_img.reduceRegion(reducer=ee.Reducer.frequencyHistogram(), geometry=ee_geom_complex, scale=1000, bestEffort=True).getInfo().get('BurnDate', {})
                    if hist:
                        df_evolucao_modis = pd.DataFrame(list(hist.items()), columns=['Dia_do_Ano', 'Pixels']).sort_values('Dia_do_Ano')
                        df_evolucao_modis['KM2'] = df_evolucao_modis['Pixels'].astype(float) * 0.25 # Aprox 500x500m
            except Exception as e:
                st.warning("⚠️ Erro no processamento MODIS. Tente uma escala menor.")

        focos_em_areas = pd.DataFrame()
        areas_afetadas_final = gpd.GeoDataFrame()
        if area_protegida != "Nenhuma":
            st.write(f"🌳 Cruzando com {area_protegida}...")
            gdf_areas_br = carregar_areas_protegidas(area_protegida)
            gdf_areas = gpd.sjoin(gdf_areas_br, limite, predicate='intersects').drop(columns=['index_right'])
            
            if not gdf_areas.empty:
                # Focos INPE nas áreas protegidas
                if ativar_inpe and not df_inpe_rec.empty:
                    gdf_inpe_pt = gpd.GeoDataFrame(df_inpe_rec, geometry=gpd.points_from_xy(df_inpe_rec["longitude"], df_inpe_rec["latitude"]), crs="EPSG:4326")
                    gdf_inpe_risco = gpd.sjoin(gdf_inpe_pt, gdf_areas, predicate='within')
                    focos_em_areas = pd.DataFrame(gdf_inpe_risco.drop(columns="geometry"))
                    if not focos_em_areas.empty:
                        areas_afetadas_final = pd.concat([areas_afetadas_final, gdf_areas[gdf_areas['nome_area'].isin(focos_em_areas['nome_area'])]])

                # Hectares MODIS nas áreas protegidas
                if ativar_modis and total_km2_modis > 0:
                    try:
                        features_ee = [ee.Feature(ee.Geometry(row['geometry'].__geo_interface__), {'nome_area': row['nome_area']}) for _, row in gdf_areas.iterrows()]
                        fc_areas = ee.FeatureCollection(features_ee)
                        stats_ap = img_area_km2.reduceRegions(collection=fc_areas, reducer=ee.Reducer.sum(), scale=500).getInfo()
                        recs = [{'Área Protegida': f['properties']['nome_area'], 'KM2': round(f['properties'].get('sum', 0), 2)} for f in stats_ap['features'] if f['properties'].get('sum', 0) > 0]
                        df_modis_areas = pd.DataFrame(recs).sort_values(by='KM2', ascending=False)
                        if not df_modis_areas.empty:
                            areas_afetadas_final = pd.concat([areas_afetadas_final, gdf_areas[gdf_areas['nome_area'].isin(df_modis_areas['Área Protegida'])]])
                    except: pass
                
                if not areas_afetadas_final.empty:
                    areas_afetadas_final = areas_afetadas_final.drop_duplicates(subset=['nome_area'])

        status.update(label="✅ Dashboard pronto!", state="complete", expanded=False)

    # --- DISPLAYS ---
    res = []
    if ativar_inpe: res.append(f"🔥 {len(df_inpe_rec):,} Focos (INPE)")
    if ativar_modis: res.append(f"🗺️ {total_km2_modis:,.2f} km² Queimados (MODIS)")
    st.markdown(f"<div style='background:#f8f9fa;padding:15px;border-radius:8px;border-left:8px solid #ff4b4b;margin-bottom:15px;'><h3 style='color:#c0392b;margin:0;'>{' | '.join(res)}</h3></div>", unsafe_allow_html=True)

    # GRÁFICOS DE ÁREAS DE RISCO (TOP 10)
    if not focos_em_areas.empty or not df_modis_areas.empty:
        st.subheader(f"🚨 Áreas de {area_protegida} mais afetadas")
        c_alt1, c_alt2 = st.columns(2)
        with c_alt1:
            if not focos_em_areas.empty:
                st.plotly_chart(px.bar(focos_em_areas['nome_area'].value_counts().reset_index().head(10), x='count', y='nome_area', orientation='h', title="Top 10 Focos (INPE)", color_discrete_sequence=['#ff4b4b'], template='plotly_dark'), use_container_width=True)
        with c_alt2:
            if not df_modis_areas.empty:
                st.plotly_chart(px.bar(df_modis_areas.head(10), x='KM2', y='Área Protegida', orientation='h', title="Top 10 Área ($km^2$) (MODIS)", color_discrete_sequence=['#e67e22'], template='plotly_dark'), use_container_width=True)

    # MAPA E EVOLUÇÃO (PICOS)
    col1, col2 = st.columns([1.3, 1])
    with col1:
        centro = limite.geometry.union_all().centroid
        m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
        folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
        if area_queimada_img: m.add_ee_layer(area_queimada_img.updateMask(area_queimada_img.gt(0)), {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']}, 'MODIS')
        
        # Filtro Rigoroso: Desenha APENAS as áreas que tiveram registro de fogo
        if not areas_afetadas_final.empty: 
            folium.GeoJson(areas_afetadas_final.__geo_interface__, style_function=lambda x: {'fillColor': 'red', 'fillOpacity': 0.3, 'color': 'yellow', 'weight': 1.5}).add_to(m)
        
        if ativar_inpe and not df_inpe_rec.empty: HeatMap(df_inpe_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15).add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

    with col2:
        if ativar_inpe and not df_inpe_rec.empty:
            df_inpe_rec['data_hora_gmt'] = pd.to_datetime(df_inpe_rec['data_hora_gmt'])
            df_t = df_inpe_rec.set_index('data_hora_gmt').resample('D').size().reset_index(name='focos')
            st.plotly_chart(px.line(df_t, x='data_hora_gmt', y='focos', title="Evolução de Focos (INPE)", template='plotly_dark').update_traces(line_color='#ff4b4b'), use_container_width=True)
        
        if ativar_modis and not df_evolucao_modis.empty:
            st.plotly_chart(px.line(df_evolucao_modis, x='Dia_do_Ano', y='KM2', title="Picos de Área Queimada ($km^2$) (MODIS)", template='plotly_dark').update_traces(line_color='#e67e22'), use_container_width=True)

else:
    st.info("👈 Ajuste os filtros e clique em 'Gerar Dashboard'.")
