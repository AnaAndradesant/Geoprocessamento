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
        key_data=json.dumps(key_dict)  # FIX: garante que é string JSON consistente
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
    if tipo == "Por Estado":
        limite = read_state(code_state=estado, year=2020)
    elif tipo == "Por Bioma":
        limite = read_biomes(year=2019)
        limite = limite[limite['name_biome'] == bioma]
    elif tipo == "Por Município":
        limite = read_municipality(code_muni=estado, year=2020)
        busca = normalizar_texto(municipio.strip())
        limite['nome_norm'] = limite['name_muni'].apply(normalizar_texto)
        limite = limite[limite['nome_norm'].str.contains(busca)]
        
    limite = limite.to_crs("EPSG:4326")
    # FIX: usa unary_union para compatibilidade com GeoPandas < 0.14
    limite['geometry'] = limite['geometry'].simplify(tolerance=0.005, preserve_topology=True)
    return limite

@st.cache_data(show_spinner=False)
def carregar_areas_protegidas(tipo_area):
    if tipo_area == "Terras Indígenas":
        gdf_areas = read_indigenous_land()
        if 'terrai_nom' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'terrai_nom': 'nome_area'})
        elif 'name' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'name': 'nome_area'})
    else:
        gdf_areas = read_conservation_units()
        if 'name_conservation_unit' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'name_conservation_unit': 'nome_area'})
        elif 'name' in gdf_areas.columns:
            gdf_areas = gdf_areas.rename(columns={'name': 'nome_area'})
    
    # FIX: garante que coluna nome_area existe mesmo se nomes de colunas mudarem
    if 'nome_area' not in gdf_areas.columns:
        gdf_areas['nome_area'] = gdf_areas.iloc[:, 0].astype(str)

    gdf_areas['geometry'] = gdf_areas['geometry'].make_valid()
    gdf_areas = gdf_areas.to_crs("EPSG:4326")
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    return gdf_areas[['nome_area', 'geometry']]

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    dic_estados = {
        "AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS",
        "BA": "BAHIA", "CE": "CEAR%", "DF": "DISTRITO FEDERAL", "ES": "ESP%RITO SANTO",
        "GO": "GOI%S", "MA": "MARANH%O", "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL",
        "MG": "MINAS GERAIS", "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%",
        "PE": "PERNAMBUCO", "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE",
        "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA", "SC": "SANTA CATARINA",
        "SP": "S%O PAULO", "SE": "SERGIPE", "TO": "TOCANTINS"
    }

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
            r = requests.get(url, params={
                "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                "typeName": "bdqueimadas:focos", "outputFormat": "application/json",
                "CQL_FILTER": cql, "maxFeatures": 10000
            }, verify=False, timeout=60)
            if r.status_code == 200 and r.json().get("features"):
                registros = [{
                    "longitude": f["geometry"]["coordinates"][0],
                    "latitude": f["geometry"]["coordinates"][1],
                    **f["properties"]
                } for f in r.json()["features"]]
                all_dfs.append(pd.DataFrame(registros))
        except: pass
        dt_ini = dt_bloco_fim + timedelta(days=1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# --- INTERFACE (BARRA LATERAL) ---
st.sidebar.title("⚙️ Filtros da Análise")
tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)
estado_dd = st.sidebar.selectbox('Selecione o Estado:', [
    "AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
    "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"
], index=25, disabled=(tipo_analise == 'Por Bioma'))
bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', [
    "Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"
], disabled=(tipo_analise != 'Por Bioma'))
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', buscar_cidades(estado_dd), disabled=(tipo_analise != 'Por Município'))

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Fontes de Dados")

ativar_inpe = st.sidebar.checkbox("🔥 Focos de Calor (INPE)", value=True)
if ativar_inpe:
    with st.sidebar.expander("⏱️ Filtros de Tempo (INPE)", expanded=True):
        unidade_dd = st.selectbox("Analisar por:", ["Dias", "Meses", "Anos"], index=1)
        quantidade_sel = st.selectbox(
            f"Quantidade:",
            options=list(range(1, 91)) if unidade_dd == "Dias" else list(range(1, 61)) if unidade_dd == "Meses" else list(range(1, 11)),
            index=1
        )
        satelites_sel = st.multiselect(
            "Satélites:",
            ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03'],
            default=['AQUA_M-T', 'NPP-375', 'NPP-375D']
        )

ativar_modis = st.sidebar.checkbox("🗺️ Cicatrizes (NASA MODIS)", value=True)
if ativar_modis:
    with st.sidebar.expander("📅 Filtros de Data (MODIS)", expanded=True):
        ano_modis = st.selectbox("Ano:", list(range(2001, datetime.now().year + 1)), index=datetime.now().year - 2002)
        mes_modis = st.selectbox("Mês:", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
# FIX: label corrigida (sem "Nenhuma" como opção de análise de risco)
area_protegida = st.sidebar.selectbox(
    "🌳 Análise de Risco Espacial:",
    ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"]
)
gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

# --- EXECUÇÃO PRINCIPAL ---
st.title("🔥 Dashboard Integrado de Queimadas")

if gerar:
    if not ativar_inpe and not ativar_modis:
        st.error("⚠️ Selecione pelo menos uma fonte de dados.")
        st.stop()

    hoje = datetime.now()
    # FIX: inicializa dt_ini sempre, independente de ativar_inpe
    dt_ini = hoje - timedelta(
        days=quantidade_sel if unidade_dd == "Dias"
        else 30 * quantidade_sel if unidade_dd == "Meses"
        else 365 * quantidade_sel
    ) if ativar_inpe else hoje - timedelta(days=30)

    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.status(f"🛰️ Processando: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando limites geográficos...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        # FIX: usa unary_union para compatibilidade ampla
        geom_unida = limite.geometry.unary_union
        ee_geom_complex = ee.Geometry(geom_unida.__geo_interface__)

        df_inpe_rec = pd.DataFrame()
        gdf_inpe_pt = gpd.GeoDataFrame()
        if ativar_inpe:
            st.write("📡 Consultando INPE...")
            df = buscar_focos_inpe(
                tipo_analise, estado_dd, bioma_dd, municipio_dd,
                dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel
            )
            if not df.empty:
                gdf_inpe = gpd.GeoDataFrame(
                    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
                )
                df_inpe_rec = pd.DataFrame(gpd.sjoin(gdf_inpe, limite, predicate="within").drop(columns="geometry"))
                # FIX: mantém o GeoDataFrame dos focos para cruzamento posterior com áreas protegidas
                gdf_inpe_pt = gpd.GeoDataFrame(
                    df_inpe_rec,
                    geometry=gpd.points_from_xy(df_inpe_rec["longitude"], df_inpe_rec["latitude"]),
                    crs="EPSG:4326"
                )

        total_km2_modis = 0
        df_modis_areas = pd.DataFrame()
        df_evolucao_modis = pd.DataFrame()
        area_queimada_img = None
        img_area_km2 = None  # FIX: inicializa fora do bloco para evitar NameError
        if ativar_modis:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao = (ee.ImageCollection('MODIS/061/MCD64A1')
                           .filterDate(data_ini_ee, data_ini_ee.advance(1, 'month'))
                           .filterBounds(ee_geom_complex))
                if colecao.size().getInfo() > 0:
                    area_queimada_img = colecao.select('BurnDate').max().clip(ee_geom_complex)
                    img_area_km2 = (ee.Image.pixelArea()
                                    .divide(1_000_000)
                                    .updateMask(area_queimada_img.gt(0))
                                    .rename('area_km2'))
                    
                    stats_total = img_area_km2.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=ee_geom_complex,
                        scale=500,
                        maxPixels=1e13,
                        bestEffort=True
                    ).getInfo()
                    total_km2_modis = round(
                        stats_total.get('area_km2', 0) if stats_total.get('area_km2') else 0, 2
                    )

                    hist = (area_queimada_img
                            .reduceRegion(
                                reducer=ee.Reducer.frequencyHistogram(),
                                geometry=ee_geom_complex,
                                scale=1000,
                                bestEffort=True
                            ).getInfo().get('BurnDate', {}))
                    if hist:
                        df_evolucao_modis = (pd.DataFrame(list(hist.items()), columns=['Dia do Ano', 'Pixels'])
                                             .sort_values('Dia do Ano'))
                        df_evolucao_modis['km²'] = df_evolucao_modis['Pixels'].astype(float) * 0.25
            except Exception as e:
                st.warning(f"⚠️ Erro no processamento MODIS: {e}")

        # --- ANÁLISE DE ÁREAS PROTEGIDAS ---
        focos_em_areas = pd.DataFrame()
        areas_afetadas_final = gpd.GeoDataFrame({'nome_area': pd.Series(dtype=str), 'geometry': pd.Series(dtype=object)})
        areas_afetadas_final = areas_afetadas_final.set_geometry('geometry')
        areas_afetadas_final.crs = None  # será definido no primeiro concat

        if area_protegida != "Nenhuma":
            st.write(f"🌳 Cruzando com {area_protegida}...")
            gdf_areas_br = carregar_areas_protegidas(area_protegida)
            
            # FIX: garante mesmo CRS antes do sjoin
            gdf_areas_br = gdf_areas_br.to_crs("EPSG:4326")
            limite_reproj = limite.to_crs("EPSG:4326")
            
            # FIX: usa intersects para pegar todas as áreas que tocam o limite selecionado
            gdf_areas = gpd.sjoin(gdf_areas_br, limite_reproj[['geometry']], predicate='intersects', how='inner')
            gdf_areas = gdf_areas.drop(columns=[c for c in gdf_areas.columns if c.startswith('index_')], errors='ignore')
            gdf_areas = gdf_areas[['nome_area', 'geometry']].drop_duplicates(subset=['nome_area'])
            
            if not gdf_areas.empty:
                # Focos INPE nas áreas protegidas
                if ativar_inpe and not gdf_inpe_pt.empty:
                    # FIX: usa gdf_inpe_pt (GeoDataFrame com geometria) em vez de df_inpe_rec
                    gdf_inpe_risco = gpd.sjoin(gdf_inpe_pt, gdf_areas[['nome_area', 'geometry']], predicate='within', how='inner')
                    focos_em_areas = pd.DataFrame(gdf_inpe_risco.drop(columns="geometry", errors='ignore'))
                    
                    if not focos_em_areas.empty:
                        nomes_afetados = focos_em_areas['nome_area'].unique()
                        novas_areas = gdf_areas[gdf_areas['nome_area'].isin(nomes_afetados)].copy()
                        areas_afetadas_final = gpd.GeoDataFrame(
                            pd.concat([areas_afetadas_final, novas_areas], ignore_index=True),
                            geometry='geometry', crs="EPSG:4326"
                        )

                # Hectares MODIS nas áreas protegidas
                if ativar_modis and total_km2_modis > 0 and img_area_km2 is not None:
                    try:
                        features_ee = [
                            ee.Feature(ee.Geometry(row['geometry'].__geo_interface__), {'nome_area': row['nome_area']})
                            for _, row in gdf_areas.iterrows()
                        ]
                        fc_areas = ee.FeatureCollection(features_ee)
                        stats_ap = img_area_km2.reduceRegions(
                            collection=fc_areas, reducer=ee.Reducer.sum(), scale=500
                        ).getInfo()
                        recs = [
                            {'Área Protegida': f['properties']['nome_area'],
                             'km²': round(f['properties'].get('sum', 0), 2)}
                            for f in stats_ap['features']
                            if f['properties'].get('sum', 0) > 0
                        ]
                        df_modis_areas = pd.DataFrame(recs).sort_values(by='km²', ascending=False)
                        if not df_modis_areas.empty:
                            nomes_modis = df_modis_areas['Área Protegida'].unique()
                            novas_areas_modis = gdf_areas[gdf_areas['nome_area'].isin(nomes_modis)].copy()
                            areas_afetadas_final = gpd.GeoDataFrame(
                                pd.concat([areas_afetadas_final, novas_areas_modis], ignore_index=True),
                                geometry='geometry', crs="EPSG:4326"
                            )
                    except Exception as e:
                        st.warning(f"⚠️ Erro MODIS em áreas protegidas: {e}")

                if not areas_afetadas_final.empty:
                    areas_afetadas_final = areas_afetadas_final.drop_duplicates(subset=['nome_area'])

        status.update(label="✅ Dashboard pronto!", state="complete", expanded=False)

    # --- DISPLAYS ---
    res = []
    if ativar_inpe: res.append(f"🔥 {len(df_inpe_rec):,} Focos (INPE)")
    if ativar_modis: res.append(f"🗺️ {total_km2_modis:,.2f} km² Queimados (MODIS)")
    st.markdown(
        f"<div style='background:#f8f9fa;padding:15px;border-radius:8px;border-left:8px solid #ff4b4b;margin-bottom:15px;'>"
        f"<h3 style='color:#c0392b;margin:0;'>{' | '.join(res)}</h3></div>",
        unsafe_allow_html=True
    )

    # --- GRÁFICOS DE ÁREAS DE RISCO ---
    if area_protegida != "Nenhuma" and (not focos_em_areas.empty or not df_modis_areas.empty):
        st.subheader(f"🚨 {area_protegida} mais afetadas")
        c_alt1, c_alt2 = st.columns(2)

        with c_alt1:
            if not focos_em_areas.empty:
                df_top_focos = (focos_em_areas['nome_area']
                                .value_counts()
                                .reset_index()
                                .rename(columns={'nome_area': 'Área', 'count': 'Focos'})
                                .head(10))
                fig_focos = px.bar(
                    df_top_focos, x='Focos', y='Área', orientation='h',
                    # FIX: legenda sem LaTeX, clara e direta
                    title=f"Top 10 por Focos de Calor (INPE)",
                    labels={'Focos': 'Quantidade de focos', 'Área': area_protegida},
                    color_discrete_sequence=['#ff4b4b'],
                    template='plotly_dark',
                    height=400
                )
                fig_focos.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig_focos, use_container_width=True)

        with c_alt2:
            if not df_modis_areas.empty:
                fig_modis = px.bar(
                    df_modis_areas.head(10), x='km²', y='Área Protegida', orientation='h',
                    # FIX: legenda sem LaTeX
                    title="Top 10 por Área Queimada em km² (MODIS)",
                    labels={'km²': 'Área queimada (km²)', 'Área Protegida': area_protegida},
                    color_discrete_sequence=['#e67e22'],
                    template='plotly_dark',
                    height=400
                )
                fig_modis.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig_modis, use_container_width=True)

    # --- MAPA E GRÁFICOS DE EVOLUÇÃO ---
    # FIX: layout melhorado — mapa ocupa linha inteira, gráficos de evolução em linha separada
    st.subheader("🗺️ Mapa de Calor e Cicatrizes")
    centro = limite.geometry.unary_union.centroid
    m = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=10 if tipo_analise == "Por Município" else 6,
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google Satélite'
    )
    folium.GeoJson(
        limite.__geo_interface__,
        style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}
    ).add_to(m)

    if area_queimada_img:
        m.add_ee_layer(
            area_queimada_img.updateMask(area_queimada_img.gt(0)),
            {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']},
            'MODIS'
        )

    # FIX: apenas áreas COM registro de fogo são destacadas
    if not areas_afetadas_final.empty:
        folium.GeoJson(
            areas_afetadas_final.__geo_interface__,
            style_function=lambda x: {
                'fillColor': 'red', 'fillOpacity': 0.3,
                'color': 'yellow', 'weight': 1.5
            },
            tooltip=folium.GeoJsonTooltip(fields=['nome_area'], aliases=[area_protegida + ':'])
        ).add_to(m)

    if ativar_inpe and not df_inpe_rec.empty:
        HeatMap(df_inpe_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15).add_to(m)

    st_folium(m, width="100%", height=520, returned_objects=[])

    # FIX: gráficos de evolução em linha separada, com largura balanceada
    st.subheader("📈 Evolução Temporal")
    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        if ativar_inpe and not df_inpe_rec.empty:
            df_inpe_rec['data_hora_gmt'] = pd.to_datetime(df_inpe_rec['data_hora_gmt'])
            df_t = (df_inpe_rec
                    .set_index('data_hora_gmt')
                    .resample('D').size()
                    .reset_index(name='Focos'))
            df_t.rename(columns={'data_hora_gmt': 'Data'}, inplace=True)
            fig_evo_inpe = px.line(
                df_t, x='Data', y='Focos',
                title="Focos de calor por dia (INPE)",
                labels={'Data': 'Data', 'Focos': 'Quantidade de focos'},
                template='plotly_dark', height=350
            )
            fig_evo_inpe.update_traces(line_color='#ff4b4b')
            fig_evo_inpe.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_evo_inpe, use_container_width=True)
        elif ativar_inpe:
            st.info("Sem focos INPE para o período/região selecionados.")

    with col_ev2:
        if ativar_modis and not df_evolucao_modis.empty:
            fig_evo_modis = px.line(
                df_evolucao_modis, x='Dia do Ano', y='km²',
                title="Picos de área queimada por dia do ano (MODIS)",
                # FIX: legenda clara, sem LaTeX
                labels={'Dia do Ano': 'Dia do ano', 'km²': 'Área queimada (km²)'},
                template='plotly_dark', height=350
            )
            fig_evo_modis.update_traces(line_color='#e67e22')
            fig_evo_modis.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_evo_modis, use_container_width=True)
        elif ativar_modis:
            st.info("Sem dados MODIS para o período/região selecionados.")

else:
    st.info("👈 Ajuste os filtros e clique em 'Gerar Dashboard'.")
