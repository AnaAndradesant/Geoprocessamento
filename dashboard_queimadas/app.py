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
    page_title="Monitor de Queimadas Brasil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 🛠️ KEEP-ALIVE E ESTADO
if 'last_heartbeat' not in st.session_state:
    st.session_state.last_heartbeat = datetime.now()

if (datetime.now() - st.session_state.last_heartbeat).seconds > 60:
    st.session_state.last_heartbeat = datetime.now()

if 'gerar_dashboard' not in st.session_state:
    st.session_state.gerar_dashboard = False
# ==========================================

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

def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1.0):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = (
            map_id_dict.get('tile_fetcher', {}).url_format
            if 'tile_fetcher' in map_id_dict
            else map_id_dict.get('urlFormat', map_id_dict.get('url_format', ''))
        )
        folium.raster_layers.TileLayer(
            tiles=tiles_url, attr='Map Data © Google Earth Engine', name=name,
            overlay=True, control=True, show=show, opacity=opacity
        ).add_to(self)
    except Exception as e:
        st.error(f"🚨 Erro crítico ao desenhar a camada: {e}")

folium.Map.add_ee_layer = add_ee_layer

# =============================================================
# --- FUNÇÕES UTILITÁRIAS ---
# =============================================================

def gerar_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Dados')
    return buffer.getvalue()

def normalizar_texto(txt):
    if pd.isna(txt):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(txt))
        if unicodedata.category(c) != 'Mn'
    ).lower()

# =============================================================
# --- FUNÇÕES COM CACHE ---
# =============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def buscar_cidades(uf):
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return sorted([d['nome'] for d in resp.json()])
    except:
        pass
    return ["Erro ao carregar cidades"]

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
    dic_estados = {
        "AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS",
        "BA": "BAHIA", "CE": "CEAR%", "DF": "DISTRITO FEDERAL",
        "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O",
        "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS",
        "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%", "PE": "PERNAMBUCO",
        "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE",
        "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA",
        "SC": "SANTA CATARINA", "SP": "S%O PAULO", "SE": "SERGIPE",
        "TO": "TOCANTINS"
    }
    if tipo == "Por Estado":
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'"
    elif tipo == "Por Bioma":
        tradutor = {"Amazônia": "Amaz%nia", "Mata Atlântica": "Mata Atl%ntica"}
        filtro_base = f"bioma ILIKE '{tradutor.get(val_bioma, val_bioma)}'"
    elif tipo == "Por Município":
        muni_curinga = re.sub(
            r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni
        ).replace(' ', '%')
        filtro_base = (
            f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' "
            f"AND municipio ILIKE '{muni_curinga}%'"
        )

    dt_ini = datetime.strptime(d_ini, "%Y-%m-%d")
    dt_fim = datetime.strptime(d_fim, "%Y-%m-%d")
    all_dfs = []
    sat_str = "','".join(satelites)

    while dt_ini <= dt_fim:
        # AUMENTADO DE 5 PARA 30 DIAS PARA MELHORAR PERFORMANCE DE REQUISIÇÃO
        dt_bloco_fim = min(dt_ini + timedelta(days=30), dt_fim)
        cql = (
            f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' "
            f"AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' "
            f"AND satelite IN ('{sat_str}') AND pais_complete_id=33 AND {filtro_base}"
        )
        try:
            r = requests.get(
                url,
                params={
                    "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                    "typeName": "bdqueimadas:focos", "outputFormat": "application/json",
                    "CQL_FILTER": cql, "maxFeatures": 10000
                },
                verify=False, timeout=60
            )
            if r.status_code == 200 and r.json().get("features"):
                registros = [
                    {"longitude": f["geometry"]["coordinates"][0],
                     "latitude": f["geometry"]["coordinates"][1],
                     **f["properties"]}
                    for f in r.json()["features"]
                ]
                all_dfs.append(pd.DataFrame(registros))
        except:
            pass
        dt_ini = dt_bloco_fim + timedelta(days=1)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def calcular_anomalia_modis(geom_json_str, ano_ref):
    """
    Calcula a anomalia mensal do ano_ref versus a média histórica 2001–(ano_ref-1).
    Retorna DataFrame com: Mês, Área no ano_ref, Média histórica, Anomalia (%).
    """
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    anos_historico = list(range(2001, ano_ref))
    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    registros = []

    for mes in range(1, 13):
        ini_ref = ee.Date.fromYMD(ano_ref, mes, 1)
        img_ref = (
            ee.ImageCollection('MODIS/061/MCD64A1')
            .filterDate(ini_ref, ini_ref.advance(1, 'month'))
            .filterBounds(ee_geom)
            .select('BurnDate').max().clip(ee_geom)
        )
        area_ref = (
            ee.Image.pixelArea().divide(1000000)
            .updateMask(img_ref.gt(0))
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=ee_geom,
                scale=1000, maxPixels=1e10, bestEffort=True
            ).getInfo()
        )
        val_ref = round(area_ref.get('area', 0) or 0, 2)

        areas_hist = []
        for ano_h in anos_historico:
            ini_h = ee.Date.fromYMD(ano_h, mes, 1)
            img_h = (
                ee.ImageCollection('MODIS/061/MCD64A1')
                .filterDate(ini_h, ini_h.advance(1, 'month'))
                .filterBounds(ee_geom)
                .select('BurnDate').max().clip(ee_geom)
            )
            a = (
                ee.Image.pixelArea().divide(1000000)
                .updateMask(img_h.gt(0))
                .reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=ee_geom,
                    scale=1000, maxPixels=1e10, bestEffort=True
                ).getInfo()
            )
            areas_hist.append(a.get('area', 0) or 0)

        media_hist = round(sum(areas_hist) / len(areas_hist), 2) if areas_hist else 0
        anomalia_pct = (
            round(((val_ref - media_hist) / media_hist * 100), 1)
            if media_hist > 0 else 0
        )

        registros.append({
            'Mês': mes,
            'Mês Nome': meses_map[mes],
            f'Área {ano_ref} (km²)': val_ref,
            'Média Histórica (km²)': media_hist,
            'Anomalia (%)': anomalia_pct
        })

    return pd.DataFrame(registros)

@st.cache_data(ttl=86400, show_spinner=False)
def calcular_nbr_sentinel(geom_json_str, ano, mes):
    """
    Calcula dNBR (severidade de queimada) com Sentinel-2 SR.
    Janela pré: 3 meses antes. Janela pós: mês selecionado + 1 mês após.
    Retorna: (dnbr_img_serializado, sev_img_serializado, stats_por_classe)
    """
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    data_ref = ee.Date.fromYMD(ano, mes, 1)
    data_pre = data_ref.advance(-3, 'month')
    data_pos = data_ref.advance(2, 'month')

    def mascara_nuvem(img):
        qa = img.select('QA60')
        mascara = (
            qa.bitwiseAnd(1 << 10).eq(0)
            .And(qa.bitwiseAnd(1 << 11).eq(0))
        )
        return img.updateMask(mascara).divide(10000)

    colecao = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(ee_geom)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        .map(mascara_nuvem)
    )

    img_pre = colecao.filterDate(data_pre, data_ref).median().clip(ee_geom)
    img_pos = colecao.filterDate(data_ref, data_pos).median().clip(ee_geom)

    nbr_pre = img_pre.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
    nbr_pos = img_pos.normalizedDifference(['B8', 'B12']).rename('NBR_pos')
    dnbr = nbr_pre.subtract(nbr_pos).rename('dNBR')

    # Classificação padrão USGS
    severidade = (
        dnbr
        .where(dnbr.lt(-0.1), 0)                              # Regeneração
        .where(dnbr.gte(-0.1).And(dnbr.lt(0.1)), 1)          # Não afetado
        .where(dnbr.gte(0.1).And(dnbr.lt(0.27)), 2)          # Baixa
        .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 3)         # Moderada
        .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 4)         # Moderada-Alta
        .where(dnbr.gte(0.66), 5)                             # Alta
    ).rename('severidade')

    pixel_area = ee.Image.pixelArea().divide(1000000)
    labels = {
        0: 'Regeneração', 1: 'Não afetado', 2: 'Baixa',
        3: 'Moderada', 4: 'Moderada-Alta', 5: 'Alta'
    }
    stats = {}
    for classe, nome in labels.items():
        mascara_cls = severidade.eq(classe)
        area = (
            pixel_area.updateMask(mascara_cls)
            .reduceRegion(
                reducer=ee.Reducer.sum(), geometry=ee_geom,
                scale=20, maxPixels=1e11, bestEffort=True
            ).getInfo()
        )
        stats[nome] = round(area.get('area', 0) or 0, 2)

    return dnbr, severidade, stats

# =============================================================
# --- INTERFACE (BARRA LATERAL) ---
# =============================================================

st.sidebar.title("⚙️ Filtros da Análise")

tipo_analise = st.sidebar.radio(
    'Escala Geográfica:',
    ['Por Estado', 'Por Bioma', 'Por Município'],
    index=2
)

estado_dd = st.sidebar.selectbox(
    'Selecione o Estado:',
    ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
     "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"],
    index=25,
    disabled=(tipo_analise == 'Por Bioma')
)
bioma_dd = st.sidebar.selectbox(
    'Selecione o Bioma:',
    ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"],
    disabled=(tipo_analise != 'Por Bioma')
)
municipio_dd = st.sidebar.selectbox(
    'Selecione a Cidade:',
    buscar_cidades(estado_dd),
    disabled=(tipo_analise != 'Por Município')
)

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Fonte de Dados")
fonte_escolhida = st.sidebar.radio(
    "Escolha o que analisar:",
    ["🔥 Focos de Calor (INPE)", "🗺️ Área Queimada (NASA MODIS)"]
)

if "INPE" in fonte_escolhida:
    st.sidebar.markdown("**Filtros do INPE**")
    unidade_dd = st.sidebar.selectbox("Analisar por:", ["Dias", "Meses", "Anos"], index=1)
    if unidade_dd == "Dias":
        op_qtd = list(range(1, 91))
    elif unidade_dd == "Meses":
        op_qtd = list(range(1, 61))
    else:
        op_qtd = list(range(1, 11))
    quantidade_sel = st.sidebar.selectbox(f"Quantidade de {unidade_dd}:", options=op_qtd, index=1)

    satelites_lista = ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03']
    satelites_sel = st.sidebar.multiselect(
        "Satélites de Referência:",
        satelites_lista,
        default=['AQUA_M-T', 'NPP-375', 'NPP-375D']
    )
else:
    st.sidebar.markdown("**Filtros do MODIS**")
    ano_modis = st.sidebar.selectbox(
        "Ano de Referência:",
        list(range(2001, datetime.now().year + 1)),
        index=datetime.now().year - 2002
    )
    mes_modis = st.sidebar.selectbox("Mês do Mapa Principal:", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox(
    "🌳 Análise de Risco (Cruzamento Espacial):",
    ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"]
)

# ==========================================
st.sidebar.markdown("---")
modo_debug = False

if "qa" in st.query_params and st.query_params["qa"].lower() == "true":
    modo_debug = st.sidebar.toggle("🐛 Modo de Validação (QA)", value=True)

if st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True):
    st.session_state.gerar_dashboard = True

# --- SEÇÃO DE CONTATO ---
st.sidebar.markdown("---")
gmail_logo_url = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg"
linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"

html_contato_novo = f"""
<div style="text-align: center;">
<p style="font-size: 12px; color: #888; margin-bottom: 2px; margin-top: 10px;">Desenvolvido por</p>
<h2 style="font-family: serif; font-size: 20px; margin-top: 0px; margin-bottom: 2px; color: inherit;">ANA ANDRADE</h2>
<p style="font-size: 12px; color: #777; margin-top: 0px; margin-bottom: 15px;">Especialista em Geoprocessamento</p>
<hr style="border: 0; border-top: 1px solid #e0e0e0; margin-bottom: 15px;">
<div style="display: flex; justify-content: space-between; gap: 10px;">
<a href="https://mail.google.com/mail/?view=cm&fs=1&to=anacarolinasantos580@gmail.com" target="_blank"
   style="flex: 1; text-decoration: none; background-color: #ffffff; border: 1px solid #e0e0e0;
          border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
          display: flex; justify-content: center; align-items: center;">
  <img src="{gmail_logo_url}" alt="Gmail" style="width: 26px; height: auto; display: block; margin: 0 auto;">
</a>
<a href="https://www.linkedin.com/in/ana-carolina-santos-3920931b3" target="_blank"
   style="flex: 1; text-decoration: none; background-color: #0077B5; border-radius: 8px;
          padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
          display: flex; justify-content: center; align-items: center;">
  <img src="{linkedin_logo_url}" alt="LinkedIn" style="width: 22px; height: auto; display: block; margin: 0 auto;">
</a>
</div>
</div>
"""
st.sidebar.markdown(html_contato_novo, unsafe_allow_html=True)

# =============================================================
# --- TELA PRINCIPAL ---
# =============================================================

st.title("🔥 Dashboard de Queimadas 🔥")

total_valor = 0
dados_indisponiveis = False

if st.session_state.gerar_dashboard:
    hoje = datetime.now()
    val_sel = (
        bioma_dd if tipo_analise == "Por Bioma"
        else (estado_dd if tipo_analise == "Por Estado"
              else f"{municipio_dd} ({estado_dd})")
    )

    with st.status(f"🛰️ Processando dados para: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando fronteiras geográficas...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        geom_unida = limite.geometry.union_all()
        ee_geom_complex = ee.Geometry(geom_unida.__geo_interface__)
        geom_json_str = json.dumps(geom_unida.__geo_interface__, sort_keys=True)

        df_ranking_areas = pd.DataFrame()
        areas_afetadas = gpd.GeoDataFrame()
        df_rec = pd.DataFrame()
        area_queimada_img = None
        df_top_mun_modis = pd.DataFrame()
        df_modis_temporal = pd.DataFrame()
        ee_geom_afetadas = ee_geom_complex  # fallback

        # -------------------------------------------------------
        # FONTE: INPE
        # -------------------------------------------------------
        if "INPE" in fonte_escolhida:
            if unidade_dd == "Dias":
                dt_ini = hoje - timedelta(days=quantidade_sel)
            elif unidade_dd == "Meses":
                dt_ini = hoje - timedelta(days=30 * quantidade_sel)
            else:
                dt_ini = hoje - timedelta(days=365 * quantidade_sel)

            if not satelites_sel:
                st.error("⚠️ Você precisa selecionar pelo menos um satélite.")
                st.stop()

            st.write("📡 Consultando satélites do INPE...")
            df = buscar_focos_inpe(
                tipo_analise, estado_dd, bioma_dd, municipio_dd,
                dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel
            )

            if not df.empty:
                gdf = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                    crs="EPSG:4326"
                )
                gdf = gpd.sjoin(gdf, limite, predicate="within")
                df_rec = pd.DataFrame(gdf.drop(columns="geometry"))
                total_valor = len(df_rec)

                if area_protegida != "Nenhuma" and not df_rec.empty:
                    st.write(f"🌳 Isolando focos em {area_protegida}...")
                    gdf_areas = carregar_areas_protegidas(area_protegida)
                    gdf_areas = gdf_areas.to_crs(gdf.crs)

                    if 'index_right' in gdf.columns:
                        gdf = gdf.drop(columns=['index_right'])

                    gdf_focos_risco = gpd.sjoin(gdf, gdf_areas, predicate='within')

                    if not gdf_focos_risco.empty:
                        areas_afetadas = gdf_areas[
                            gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])
                        ]
                        focos_em_areas = pd.DataFrame(gdf_focos_risco.drop(columns="geometry"))
                        df_ranking_areas = (
                            focos_em_areas['nome_area']
                            .value_counts()
                            .reset_index()
                        )
                        df_ranking_areas.columns = ['Área Protegida', 'Valor']
                        df_rec = focos_em_areas
                        total_valor = len(df_rec)
                    else:
                        df_rec = pd.DataFrame()
                        total_valor = 0

        # -------------------------------------------------------
        # FONTE: MODIS
        # -------------------------------------------------------
        else:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao = (
                    ee.ImageCollection('MODIS/061/MCD64A1')
                    .filterDate(data_ini_ee, data_ini_ee.advance(1, 'month'))
                    .filterBounds(ee_geom_complex)
                )

                if colecao.size().getInfo() == 0:
                    dados_indisponiveis = True
                    total_valor = 0
                else:
                    area_queimada_img = (
                        colecao.select('BurnDate').max().clip(ee_geom_complex)
                    )
                    img_area_km2 = (
                        ee.Image.pixelArea().divide(1000000)
                        .updateMask(area_queimada_img.gt(0))
                        .rename('area_km2')
                    )

                    stats_total = img_area_km2.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=ee_geom_complex,
                        scale=500, maxPixels=1e13, bestEffort=True
                    ).getInfo()
                    total_valor = round(
                        stats_total.get('area_km2', 0)
                        if stats_total.get('area_km2') else 0, 2
                    )

                    if area_protegida != "Nenhuma" and total_valor > 0:
                        st.write(f"🌳 Isolando km² afetados em {area_protegida}...")
                        gdf_areas_br = carregar_areas_protegidas(tipo_area=area_protegida)
                        gdf_areas = gpd.sjoin(
                            gdf_areas_br, limite, predicate='intersects'
                        ).drop(columns=['index_right'])

                        if not gdf_areas.empty:
                            features_ee = [
                                ee.Feature(
                                    ee.Geometry(row['geometry'].__geo_interface__),
                                    {'nome_area': row['nome_area']}
                                )
                                for _, row in gdf_areas.iterrows()
                            ]
                            fc_areas = ee.FeatureCollection(features_ee)
                            stats = img_area_km2.reduceRegions(
                                collection=fc_areas, reducer=ee.Reducer.sum(), scale=500
                            ).getInfo()
                            recs = [
                                {
                                    'Área Protegida': f['properties']['nome_area'],
                                    'Valor': round(f['properties'].get('sum', 0), 2)
                                }
                                for f in stats['features']
                                if f['properties'].get('sum', 0) > 0
                            ]
                            df_ranking_areas = pd.DataFrame(recs).sort_values(
                                by='Valor', ascending=False
                            )
                            if not df_ranking_areas.empty:
                                areas_afetadas = gdf_areas[
                                    gdf_areas['nome_area'].isin(
                                        df_ranking_areas['Área Protegida']
                                    )
                                ]
                                total_valor = round(df_ranking_areas['Valor'].sum(), 2)
                                ee_geom_afetadas = ee.Geometry(
                                    areas_afetadas.geometry.union_all().__geo_interface__
                                )
                                area_queimada_img = area_queimada_img.clip(ee_geom_afetadas)
                                img_area_km2 = (
                                    ee.Image.pixelArea().divide(1000000)
                                    .updateMask(area_queimada_img.gt(0))
                                )
                            else:
                                total_valor = 0

                    if tipo_analise != "Por Município" and total_valor > 0:
                        st.write("🏙️ Calculando ranking de municípios (MODIS)...")
                        muns_ee = (
                            ee.FeatureCollection("FAO/GAUL/2015/level2")
                            .filterBounds(ee_geom_complex)
                        )
                        stats_mun = img_area_km2.reduceRegions(
                            collection=muns_ee, reducer=ee.Reducer.sum(), scale=1000
                        ).getInfo()
                        recs_mun = [
                            {
                                'Município': f['properties']['ADM2_NAME'],
                                'Valor': round(f['properties'].get('sum', 0), 2)
                            }
                            for f in stats_mun['features']
                            if f['properties'].get('sum', 0) > 0
                        ]
                        if recs_mun:
                            df_top_mun_modis = (
                                pd.DataFrame(recs_mun)
                                .sort_values(by='Valor', ascending=False)
                                .head(5)
                            )

                    if total_valor > 0:
                        st.write("📊 Calculando série temporal anual (MODIS)...")
                        geom_temporal = (
                            ee_geom_afetadas
                            if (area_protegida != "Nenhuma" and not areas_afetadas.empty)
                            else ee_geom_complex
                        )

                        def calc_mes(m):
                            m_num = ee.Number(m)
                            ini = ee.Date.fromYMD(ano_modis, m_num, 1)
                            fim = ini.advance(1, 'month')
                            img_mes = (
                                ee.ImageCollection('MODIS/061/MCD64A1')
                                .filterDate(ini, fim)
                                .select('BurnDate').max().clip(geom_temporal)
                            )
                            area_calc = (
                                ee.Image.pixelArea().divide(1000000)
                                .updateMask(img_mes.gt(0))
                            )
                            val = area_calc.reduceRegion(
                                reducer=ee.Reducer.sum(),
                                geometry=geom_temporal,
                                scale=1000, maxPixels=1e10
                            ).get('area')
                            return ee.Feature(None, {'mes': m_num, 'area': val})

                        meses_list = ee.List.sequence(1, 12)
                        fc_meses = ee.FeatureCollection(meses_list.map(calc_mes)).getInfo()
                        dados_temp = [
                            {
                                'Mês': f['properties']['mes'],
                                'Área (km²)': round(f['properties'].get('area') or 0, 2)
                            }
                            for f in fc_meses['features']
                        ]
                        df_modis_temporal = pd.DataFrame(dados_temp)
                        meses_map_label = {
                            1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                            7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
                        }
                        df_modis_temporal['Mês Nome'] = df_modis_temporal['Mês'].map(
                            meses_map_label
                        )

            except Exception as e:
                st.warning(f"⚠️ Erro ao processar MODIS: {e}")

        status.update(label="✅ Análise concluída!", state="complete", expanded=False)

    # =============================================================
    # --- RENDERIZAÇÃO DAS INFORMAÇÕES NA TELA ---
    # =============================================================

    if dados_indisponiveis:
        mes_nome = {
            1:'Jan',2:'Fev',3:'Mar',4:'Abr',5:'Mai',6:'Jun',
            7:'Jul',8:'Ago',9:'Set',10:'Out',11:'Nov',12:'Dez'
        }[mes_modis]
        st.warning(
            f"⏳ **Aviso de Processamento NASA:** Os dados do satélite MODIS para "
            f"**{mes_nome} de {ano_modis}** ainda não foram publicados. "
            f"(Geralmente há atraso de 1 a 2 meses). Por favor, tente um mês anterior."
        )

    elif total_valor == 0:
        st.error("⚠️ Nenhum registro detectado nos limites selecionados.")

    else:
        # --- CARD PRINCIPAL ---
        texto_titulo = (
            f"Total Confirmado: {total_valor:,} focos"
            if "INPE" in fonte_escolhida
            else f"Área Queimada Total: {total_valor:,.2f} km²"
        )
        if "INPE" in fonte_escolhida:
            texto_sub = (
                f"Período Analisado: {dt_ini.strftime('%d/%m/%Y')} "
                f"até {hoje.strftime('%d/%m/%Y')}"
            )
        else:
            texto_sub = (
                f"Período: Mês {mes_modis} de {ano_modis} (Mapa) "
                f"/ Ano {ano_modis} (Evolução)"
            )

        card_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px;
                    border-left: 8px solid #ff4b4b; margin-bottom: 15px;
                    box-shadow: 1px 1px 4px rgba(0,0,0,0.05);">
            <h3 style="color: #c0392b; margin: 0; font-size: 22px; font-weight: bold;">
                🔥 {texto_titulo}
            </h3>
            <p style="color: #636e72; margin: 4px 0 0 0; font-size: 15px;">
                Análise: {val_sel} | {texto_sub}
            </p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # --- ALERTA DE ÁREAS PROTEGIDAS E GRÁFICOS INICIAIS ---
        if not df_ranking_areas.empty:
            metrica = "focos detectados" if "INPE" in fonte_escolhida else "km² queimados"
            st.error(
                f"🚨 **ANÁLISE FOCADA:** {total_valor} {metrica} "
                f"limitados dentro de {area_protegida}!"
            )
            col_alerta1, col_alerta2 = st.columns([1.5, 1])
            with col_alerta1:
                qtd_areas = min(10, len(df_ranking_areas))
                titulo_dinamico = (
                    f"🔥 Top {qtd_areas} Áreas Mais Afetadas"
                    if qtd_areas > 1 else "🔥 Área Mais Afetada"
                )
                fig_areas = px.bar(
                    df_ranking_areas.head(10),
                    x='Valor', y='Área Protegida', orientation='h',
                    text='Valor', color='Valor',
                    color_continuous_scale='Reds', title=titulo_dinamico
                )
                fig_areas.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_areas, use_container_width=True)

        # =========================================================
        # --- ABAS DE ANÁLISE PROFUNDA (Aqui entram as novas funções) ---
        # =========================================================
        
        tab_resumo, tab_anomalia, tab_severidade = st.tabs([
            "🗺️ Resumo Espacial", 
            "📈 Anomalia Histórica (MODIS)", 
            "🔥 Severidade (Sentinel-2)"
        ])
        
        with tab_resumo:
            st.write("*(Insira aqui o seu mapa original do Folium, se desejar)*")
            if not df_top_mun_modis.empty:
                st.markdown("### Top 5 Municípios Afetados")
                st.dataframe(df_top_mun_modis, use_container_width=True)
                
        with tab_anomalia:
            if "MODIS" in fonte_escolhida:
                st.markdown(f"### Anomalia de Queimadas x Média Histórica (Desde 2001)")
                st.write("Comparando o ano selecionado com a média histórica da mesma região.")
                with st.spinner("Calculando série histórica no Earth Engine (isso pode levar um tempo)..."):
                    df_anomalia = calcular_anomalia_modis(geom_json_str, ano_modis)
                    if not df_anomalia.empty:
                        fig_anomalia = px.bar(
                            df_anomalia, x='Mês Nome', y='Anomalia (%)', 
                            color='Anomalia (%)', color_continuous_scale='RdYlGn_r',
                            text='Anomalia (%)'
                        )
                        fig_anomalia.update_traces(texttemplate='%{text}%', textposition='outside')
                        st.plotly_chart(fig_anomalia, use_container_width=True)
                        st.dataframe(df_anomalia, use_container_width=True)
            else:
                st.info("💡 **Dica:** Para ver o gráfico de Anomalia Histórica, mude a fonte de dados na barra lateral para **Área Queimada (NASA MODIS)**.")
                
        with tab_severidade:
            if "MODIS" in fonte_escolhida: # Você pode habilitar para o INPE também ajustando a lógica de ano/mês
                st.markdown(f"### Índice de Severidade de Queimada (dNBR) - Sentinel-2")
                st.write(f"Avaliando o impacto do fogo na vegetação para o mês {mes_modis}/{ano_modis}.")
                with st.spinner("Processando imagens Sentinel-2 (isso exige bastante do servidor)..."):
                    dnbr, severidade, stats_sev = calcular_nbr_sentinel(geom_json_str, ano_modis, mes_modis)
                    
                    df_sev = pd.DataFrame(list(stats_sev.items()), columns=['Classe de Severidade', 'Área (km²)'])
                    fig_sev = px.pie(df_sev, values='Área (km²)', names='Classe de Severidade', 
                                     title='Distribuição da Severidade', hole=0.4,
                                     color='Classe de Severidade',
                                     color_discrete_map={
                                         'Regeneração': 'green', 'Não afetado': 'lightgreen',
                                         'Baixa': 'yellow', 'Moderada': 'orange',
                                         'Moderada-Alta': 'darkorange', 'Alta': 'red'
                                     })
                    st.plotly_chart(fig_sev, use_container_width=True)
                    st.dataframe(df_sev, use_container_width=True)
            else:
                st.info("💡 **Dica:** O cálculo de severidade dNBR consome os dados de Ano e Mês do filtro MODIS.")
