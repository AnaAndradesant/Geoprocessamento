import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
import ee
import json
import requests
from datetime import datetime, timedelta
from io import BytesIO

# =============================================================
# 1. CONFIGURAÇÕES INICIAIS E FUNÇÕES DE SUPORTE
# =============================================================

st.set_page_config(page_title="Monitor de Fogo Pro", layout="wide", page_icon="🔥")

# Injeção de CSS para melhorar a estética
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# Inicialização do Earth Engine
@st.cache_resource
def iniciar_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

iniciar_ee()

# --- FUNÇÃO CRUCIAL: Adiciona suporte ao EE no Folium ---
def add_ee_layer(self, ee_object, vis_params, name, opacity=1, show=True):
    """Método injetado no folium.Map para plotar objetos do Earth Engine."""
    try:
        if isinstance(ee_object, ee.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True,
                opacity=opacity,
                show=show
            ).add_to(self)
        elif isinstance(ee_object, ee.FeatureCollection):
            ee_object_new = ee.FeatureCollection(ee_object).draw(color='000000', strokeWidth=1)
            map_id_dict = ee_object_new.getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True,
                opacity=opacity,
                show=show
            ).add_to(self)
    except Exception as e:
        print(f"Erro ao adicionar camada EE: {e}")

# Injeta a função no Folium
folium.Map.add_ee_layer = add_ee_layer

# =============================================================
# 2. FUNÇÕES DE PROCESSAMENTO (API INPE, MODIS, NBR)
# =============================================================

@st.cache_data(ttl=3600)
def carregar_focos_inpe(municipio_id):
    url = f"https://queimadas.dgi.inpe.br/api/focos/?municipio_id={municipio_id}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df['properties'] = df['properties'].apply(lambda x: x if isinstance(x, dict) else {})
                props = pd.json_normalize(df['properties'])
                df = pd.concat([df.drop('properties', axis=1), props], axis=1)
                # Extrai lat/lon da coluna geometry
                df['longitude'] = df['geometry'].apply(lambda x: x['coordinates'][0])
                df['latitude'] = df['geometry'].apply(lambda x: x['coordinates'][1])
                return df
    except:
        return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def obter_limite_area(tipo, nome):
    if tipo == "Por Estado":
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/estados/{nome}?formato=application/vnd.geo+json"
    else:
        # Busca ID do município
        url_mun = f"https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{nome.replace(' ', '%20')}"
        res = requests.get(url_mun).json()
        id_mun = res['id'] if isinstance(res, dict) else res[0]['id']
        url = f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{id_mun}?formato=application/vnd.geo+json"
    
    gdf = gpd.read_file(url)
    return gdf

@st.cache_data
def carregar_areas_protegidas(geom_json):
    # Simplificação: Usando WDPA via Earth Engine para interseção
    poly = ee.Geometry(json.loads(geom_json)['features'][0]['geometry'])
    wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons")\
             .filterBounds(poly)\
             .filter(ee.Filter.neq('DESIG_ENG', 'UNESCO-MAB Biosphere Reserve'))
    
    features = wdpa.getInfo()['features']
    if not features: return gpd.GeoDataFrame()
    
    rows = []
    for f in features:
        rows.append({
            'nome_area': f['properties']['NAME'],
            'tipo': f['properties']['DESIG_PORT'],
            'geometry': ee.Geometry(f['geometry']).toGeoJSONString()
        })
    
    gdf = gpd.GeoDataFrame(rows)
    gdf['geometry'] = gdf['geometry'].apply(lambda x: gpd.GeoSeries.from_json(x).iloc[0])
    return gdf

@st.cache_data
def calcular_area_queimada_modis(geom_json, ano, mes=None):
    poly = ee.Geometry(json.loads(geom_json)['features'][0]['geometry'])
    dataset = ee.ImageCollection('MODIS/061/MCD64A1').filterBounds(poly)
    
    if mes:
        img = dataset.filter(ee.Filter.calendarRange(ano, ano, 'year'))\
                     .filter(ee.Filter.calendarRange(mes, mes, 'month')).max()
    else:
        img = dataset.filter(ee.Filter.calendarRange(ano, ano, 'year')).max()
    
    burned = img.select('BurnDate').clip(poly)
    area_img = ee.Image.pixelArea().updateMask(burned.gt(0))
    stats = area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=poly, scale=500, maxPixels=1e9)
    area_km2 = ee.Number(stats.get('area')).divide(1e6).getInfo()
    
    return burned, area_km2 or 0

@st.cache_data
def calcular_anomalia_modis(geom_json, ano_alvo):
    poly = ee.Geometry(json.loads(geom_json)['features'][0]['geometry'])
    dataset = ee.ImageCollection('MODIS/061/MCD64A1').filterBounds(poly)
    
    meses_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    dados = []
    
    for m in range(1, 13):
        # Média histórica (2001 até ano anterior)
        hist = dataset.filter(ee.Filter.calendarRange(2001, ano_alvo-1, 'year'))\
                      .filter(ee.Filter.calendarRange(m, m, 'month'))
        
        def calc_area(img):
            return ee.Image.pixelArea().updateMask(img.select('BurnDate').gt(0))\
                           .reduceRegion(ee.Reducer.sum(), poly, 500).get('area')

        media_m = hist.map(lambda i: ee.Feature(None, {'a': calc_area(i)}))\
                      .aggregate_mean('a')
        
        # Ano atual
        atual_img = dataset.filter(ee.Filter.calendarRange(ano_alvo, ano_alvo, 'year'))\
                           .filter(ee.Filter.calendarRange(m, m, 'month')).max()
        area_atual = calc_area(atual_img)
        
        m_hist = ee.Number(media_m).divide(1e6).getInfo() or 0
        a_atual = ee.Number(area_atual).divide(1e6).getInfo() or 0
        
        anomalia = ((a_atual - m_hist) / m_hist * 100) if m_hist > 0 else 0
        
        dados.append({
            'Mês': m, 'Mês Nome': meses_nomes[m-1],
            f'Área {ano_alvo} (km²)': round(a_atual, 2),
            'Média Histórica (km²)': round(m_hist, 2),
            'Anomalia (%)': round(anomalia, 1)
        })
    
    return pd.DataFrame(dados)

@st.cache_data
def calcular_nbr_sentinel(geom_json, ano, mes):
    poly = ee.Geometry(json.loads(geom_json)['features'][0]['geometry'])
    
    # Datas para o Sentinel (1 mês antes e o mês atual)
    data_fim = datetime(ano, mes, 28)
    data_ini = data_fim - timedelta(days=60)
    
    def get_nbr(img):
        return img.normalizedDifference(['B8', 'B12']).rename('nbr')

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
           .filterBounds(poly)\
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    pre_fire = s2.filterDate(data_ini.strftime('%Y-%m-%d'), (data_ini + timedelta(days=30)).strftime('%Y-%m-%d')).median()
    post_fire = s2.filterDate(data_fim.replace(day=1).strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d')).median()
    
    dnbr = get_nbr(pre_fire).subtract(get_nbr(post_fire)).multiply(1000).clip(poly)
    
    # Classificação
    sld_intervals = (dnbr.gt(-100).add(dnbr.gt(100)).add(dnbr.gt(270)).add(dnbr.gt(440)).add(dnbr.gt(660)))
    
    stats = sld_intervals.reduceRegion(ee.Reducer.frequencyHistogram(), poly, 20).getInfo()
    
    classes = {0: 'Regeneração', 1: 'Não afetado', 2: 'Baixa', 3: 'Moderada', 4: 'Moderada-Alta', 5: 'Alta'}
    res_stats = {}
    if 'groups' not in str(stats): # Simplificado
        for k, v in stats['groups' if 'groups' in stats else 'constant'].items():
            area = (v * 400) / 1e6 # 20m scale
            res_stats[classes.get(int(float(k)), 'Outros')] = round(area, 2)
            
    return dnbr, sld_intervals, res_stats

def gerar_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    return output.getvalue()

# =============================================================
# 3. SIDEBAR / FILTROS
# =============================================================

with st.sidebar:
    st.title("🔥 Filtros de Análise")
    fonte_escolhida = st.selectbox("📡 Fonte de Dados:", ["Focos Ativos (INPE)", "Área Queimada (NASA MODIS)"])
    
    tipo_analise = st.radio("📍 Abrangência:", ["Por Estado", "Por Município"])
    
    if tipo_analise == "Por Estado":
        val_sel = st.selectbox("UF:", ["AM", "MT", "PA", "RO", "TO", "MS", "MA"])
    else:
        val_sel = st.text_input("Digite o nome do Município (Ex: Corumbá):", "Corumbá")

    if "INPE" in fonte_escolhida:
        dt_ini = st.date_input("Início:", datetime.now() - timedelta(days=30))
        dt_fim = st.date_input("Fim:", datetime.now())
    else:
        ano_modis = st.number_input("Ano:", 2001, 2024, 2024)
        mes_modis = st.slider("Mês (Opcional - 0 para ano todo):", 0, 12, 0)

    btn_gerar = st.button("🚀 Gerar Dashboard", use_container_width=True)

# =============================================================
# 4. LÓGICA PRINCIPAL DO DASHBOARD
# =============================================================

if btn_gerar:
    with st.spinner("⏳ Carregando dados espaciais..."):
        limite = obter_limite_area(tipo_analise, val_sel)
        geom_json_str = limite.to_json()
        areas_afetadas = carregar_areas_protegidas(geom_json_str)

    # --- PROCESSAMENTO ESPECÍFICO ---
    df_rec = pd.DataFrame()
    area_queimada_img = None
    total_valor = 0

    if "INPE" in fonte_escolhida:
        # Busca ID IBGE para API INPE
        res_ibge = requests.get(f"https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{val_sel.replace(' ', '%20')}").json()
        id_ibge = res_ibge[0]['id'] if isinstance(res_ibge, list) else None
        
        if id_ibge or tipo_analise == "Por Estado":
            # Nota: O INPE por estado exigiria loop ou outra rota, aqui simulamos o município
            df_rec = carregar_focos_inpe(id_ibge)
            if not df_rec.empty:
                total_valor = len(df_rec)
                # Ranking Áreas Protegidas (Focos por área)
                if not areas_afetadas.empty:
                    focos_gdf = gpd.GeoDataFrame(df_rec, geometry=gpd.points_from_xy(df_rec.longitude, df_rec.latitude), crs="EPSG:4326")
                    joined = gpd.sjoin(focos_gdf, areas_afetadas, predicate='within')
                    df_ranking_areas = joined['nome_area'].value_counts().reset_index()
                    df_ranking_areas.columns = ['Área Protegida', 'Valor']
                else:
                    df_ranking_areas = pd.DataFrame()

    else: # MODIS
        area_queimada_img, total_valor = calcular_area_queimada_modis(geom_json_str, ano_modis, mes_modis if mes_modis > 0 else None)
        
        # Série Temporal MODIS
        df_modis_temporal = pd.DataFrame()
        if mes_modis == 0:
            df_modis_temporal = calcular_anomalia_modis(geom_json_str, ano_modis)
        
        # Ranking Municípios (se for estado)
        df_top_mun_modis = pd.DataFrame() # Simplificado para o exemplo
        
        # Ranking Áreas Protegidas MODIS
        df_ranking_areas = pd.DataFrame() # Aqui entraria uma redução por região do EE

    # --- RENDERIZAÇÃO ---
    st.title(f"📊 Dashboard: {val_sel}")
    
    c1, c2, c3 = st.columns(3)
    metrica_nome = "Focos Detectados" if "INPE" in fonte_escolhida else "Área Queimada Est."
    unidade = "focos" if "INPE" in fonte_escolhida else "km²"
    c1.metric(metrica_nome, f"{total_valor:,.1f} {unidade}")
    c2.metric("Áreas Protegidas em Risco", len(areas_afetadas))
    c3.metric("Fonte", "INPE" if "INPE" in fonte_escolhida else "NASA MODIS")

    # --- ALERTA DE ÁREAS PROTEGIDAS ---
    if not df_ranking_areas.empty:
        metrica = "focos detectados" if "INPE" in fonte_escolhida else "km² queimados"
        st.error(f"🚨 **ANÁLISE FOCADA:** {total_valor} {metrica} limitados dentro de áreas protegidas!")
        
        col_alerta1, col_alerta2 = st.columns([1.5, 1])
        with col_alerta1:
            qtd_areas = min(10, len(df_ranking_areas))
            fig_areas = px.bar(df_ranking_areas.head(10), x='Valor', y='Área Protegida', orientation='h',
                             title=f"🔥 Top {qtd_areas} Áreas Afetadas", color='Valor', color_continuous_scale='Reds')
            fig_areas.update_layout(template='plotly_dark', height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_areas, use_container_width=True)
        with col_alerta2:
            st.markdown("**Lista Completa**")
            st.dataframe(df_ranking_areas, hide_index=True, height=350, use_container_width=True)

    st.markdown("---")

    # --- ABAS ---
    aba_mapa, aba_graficos, aba_nbr, aba_export = st.tabs([
        "🗺️ Mapa de Focos", "📈 Gráficos & Anomalia", "🔬 Severidade (Sentinel-2)", "⬇️ Exportar"
    ])

    with aba_mapa:
        estilo_mapa = st.radio("Estilo:", ["🌑 Dark", "🛰️ Satélite"], horizontal=True)
        
        centro = limite.geometry.union_all().centroid
        tiles = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}' if "Satélite" in estilo_mapa else 'cartodbpositron'
        
        m = folium.Map(location=[centro.y, centro.x], zoom_start=9, tiles=tiles, attr='Map')
        
        folium.GeoJson(limite, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
        
        if "INPE" in fonte_escolhida and not df_rec.empty:
            HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=8).add_to(m)
        elif area_queimada_img:
            m.add_ee_layer(area_queimada_img.updateMask(area_queimada_img.gt(0)), 
                          {'min': 1, 'max': 366, 'palette': ['orange', 'red']}, 'MODIS')

        st_folium(m, width=None, height=600, returned_objects=[])

    with aba_graficos:
        if "INPE" in fonte_escolhida and not df_rec.empty:
            df_rec['data_hora_gmt'] = pd.to_datetime(df_rec['data_hora_gmt'])
            df_temp = df_rec.set_index('data_hora_gmt').resample('D').size().reset_index(name='focos')
            fig = px.line(df_temp, x='data_hora_gmt', y='focos', title="Evolução Diária", color_discrete_sequence=['#e64a19'])
            st.plotly_chart(fig, use_container_width=True)
        elif not df_modis_temporal.empty:
            # Mostra gráfico de anomalia (conforme código anterior)
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(x=df_modis_temporal['Mês Nome'], y=df_modis_temporal['Média Histórica (km²)'], name='Média Histórica', line=dict(dash='dash')))
            fig_anom.add_trace(go.Scatter(x=df_modis_temporal['Mês Nome'], y=df_modis_temporal[f'Área {ano_modis} (km²)'], name=str(ano_modis), line=dict(width=3)))
            fig_anom.update_layout(template='plotly_dark', title="Anomalia vs Média")
            st.plotly_chart(fig_anom, use_container_width=True)

    with aba_nbr:
        if "MODIS" in fonte_escolhida:
            st.subheader("🔬 Análise Pixel a Pixel (Sentinel-2)")
            with st.spinner("🛰️ Processando imagens..."):
                try:
                    dnbr, sev, stats_sev = calcular_nbr_sentinel(geom_json_str, ano_modis, mes_modis if mes_modis > 0 else 9)
                    
                    c_n1, c_n2 = st.columns([2, 1])
                    with c_n1:
                        m_nbr = folium.Map(location=[centro.y, centro.x], zoom_start=11, tiles=tiles, attr='Satellite')
                        # PALETA NBR
                        vis_dnbr = {'min': -100, 'max': 800, 'palette': ['#1a9850', '#91cf60', '#ffffbf', '#fd8d3c', '#d73027', '#7a0403']}
                        m_nbr.add_ee_layer(dnbr, vis_dnbr, 'Severidade')
                        st_folium(m_nbr, width=None, height=500)
                    with c_n2:
                        st.write("### Impacto")
                        df_s = pd.DataFrame(list(stats_sev.items()), columns=['Classe', 'km²'])
                        st.bar_chart(df_s.set_index('Classe'))
                        st.metric("Área Crítica", f"{df_s[df_s['Classe'].isin(['Alta', 'Moderada-Alta'])]['km²'].sum():.2f} km²")
                except Exception as e:
                    st.warning(f"Sem imagens disponíveis para o período ou erro: {e}")
        else:
            st.info("Mude para MODIS para ver a severidade Sentinel-2.")

    with aba_export:
        st.subheader("⬇️ Downloads")
        if "INPE" in fonte_escolhida and not df_rec.empty:
            st.download_button("Baixar CSV Focos", df_rec.to_csv().encode('utf-8'), "focos.csv", "text/csv")
        elif not df_modis_temporal.empty:
             st.download_button("Baixar CSV Temporal", df_modis_temporal.to_csv().encode('utf-8'), "temporal.csv", "text/csv")

else:
    st.info("👈 Selecione os filtros e clique em 'Gerar Dashboard' para iniciar a análise.")
