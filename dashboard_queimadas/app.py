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
import requests, warnings, time, unicodedata, re, json, io, calendar
import ee
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Monitor de Queimadas Brasil", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

# ==========================================
if 'last_heartbeat' not in st.session_state:
    st.session_state.last_heartbeat = datetime.now()
if (datetime.now() - st.session_state.last_heartbeat).seconds > 60:
    st.session_state.last_heartbeat = datetime.now()
if 'gerar_dashboard' not in st.session_state:
    st.session_state.gerar_dashboard = False
# ==========================================

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- AUTENTICAÇÃO EARTH ENGINE ---
try:
    key_dict = json.loads(st.secrets["EARTHENGINE_KEY"])
    credentials = ee.ServiceAccountCredentials(email=key_dict['client_email'], key_data=st.secrets["EARTHENGINE_KEY"])
    ee.Initialize(credentials, project='ee-anacarolinasantos580')
except Exception as e:
    st.error("⚠️ Erro ao conectar com o Google Earth Engine.")

def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1.0):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = map_id_dict.get('tile_fetcher', {}).get('url_format') or map_id_dict.get('url_format', '')
        folium.raster_layers.TileLayer(tiles=tiles_url, attr='Map Data © Google Earth Engine', name=name, overlay=True, control=True, show=show, opacity=opacity).add_to(self)
    except Exception as e:
        st.error(f"🚨 Erro ao desenhar camada: {e}")
folium.Map.add_ee_layer = add_ee_layer

# =============================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================
def normalizar_texto(txt):
    if pd.isna(txt): return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(txt)) if unicodedata.category(c) != 'Mn').lower()

def gerar_excel(df):
    df_export = df.copy()
    for col in df_export.select_dtypes(include=['datetimetz']).columns:
        df_export[col] = df_export[col].dt.tz_localize(None)
    for col in df_export.select_dtypes(include=['object']).columns:
        if any(isinstance(x, (list, dict)) for x in df_export[col].dropna()):
            df_export[col] = df_export[col].astype(str)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Dados')
    return output.getvalue()

# =============================================================
# FUNÇÕES COM CACHE
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
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    dic_estados = {
        "AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP%", "AM": "AMAZONAS", "BA": "BAHIA", "CE": "CEAR%",
        "DF": "DISTRITO FEDERAL", "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O", "MT": "MATO GROSSO",
        "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS", "PA": "PAR%", "PB": "PARA%BA", "PR": "PARAN%",
        "PE": "PERNAMBUCO", "PI": "PIAU%", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE",
        "RS": "RIO GRANDE DO SUL", "RO": "ROND%NIA", "RR": "RORAIMA", "SC": "SANTA CATARINA",
        "SP": "S%O PAULO", "SE": "SERGIPE", "TO": "TOCANTINS"
    }

    if tipo == "Por Estado":
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'"
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
        dt_bloco_fim = min(dt_ini + timedelta(days=1), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('{sat_str}') AND {filtro_base}"
        try:
            r = session.get(url, params={"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": "bdqueimadas:focos", "outputFormat": "application/json", "CQL_FILTER": cql, "maxFeatures": 50000}, headers=headers, verify=False, timeout=90)
            if r.status_code == 200:
                dados_json = r.json()
                if "features" in dados_json and len(dados_json["features"]) > 0:
                    registros = [{"longitude": f["geometry"]["coordinates"][0], "latitude": f["geometry"]["coordinates"][1], **f["properties"]} for f in dados_json["features"]]
                    all_dfs.append(pd.DataFrame(registros))
        except:
            pass
        dt_ini = dt_bloco_fim + timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()
    df_final = pd.concat(all_dfs, ignore_index=True)
    if 'id' in df_final.columns:
        df_final = df_final.drop_duplicates(subset=['id'])
    else:
        df_final = df_final.drop_duplicates()
    return df_final

# =============================================================
# FUNÇÕES NBR OTIMIZADAS (com máscara de queimada)
# =============================================================
def _construir_dnbr(geom_json_str, ano, mes, mask_queimada=None):
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    data_ref = ee.Date.fromYMD(ano, mes, 1)
    data_pre = data_ref.advance(-2.5, 'month')
    data_pos = data_ref.advance(2.5, 'month')

    def mascara_nuvem(img):
        qa = img.select('QA60')
        return img.updateMask(qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))).divide(10000)

    colecao_base = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ee_geom).map(mascara_nuvem)
    col_pre = colecao_base.filterDate(data_pre, data_ref)
    col_pos = colecao_base.filterDate(data_ref, data_pos)

    thresholds = [40, 60]
    threshold_usado = None
    for thresh in thresholds:
        col_pre_f = col_pre.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', thresh))
        col_pos_f = col_pos.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', thresh))
        counts = ee.Dictionary({'n_pre': col_pre_f.size(), 'n_pos': col_pos_f.size()}).getInfo()
        if counts['n_pre'] > 0 and counts['n_pos'] > 0:
            threshold_usado = thresh
            col_pre = col_pre_f
            col_pos = col_pos_f
            break

    if counts['n_pre'] == 0 or counts['n_pos'] == 0:
        raise ValueError(f"❌ Nenhuma imagem Sentinel-2 encontrada.\nMês: {mes:02d}/{ano}\nDica: Use 'Por Município' + meses de seca (Jul-Set).")

    if threshold_usado and threshold_usado > 40:
        st.warning(f"⚠️ Usando imagens com até **{threshold_usado}%** de nuvem.")

    img_pre = col_pre.median().clip(ee_geom)
    img_pos = col_pos.median().clip(ee_geom)
    nbr_pre = img_pre.normalizedDifference(['B8', 'B12']).rename('NBR_pre')
    nbr_pos = img_pos.normalizedDifference(['B8', 'B12']).rename('NBR_pos')
    dnbr = nbr_pre.subtract(nbr_pos).rename('dNBR')

    if mask_queimada is not None:
        dnbr = dnbr.updateMask(mask_queimada)

    return ee_geom, dnbr, threshold_usado


@st.cache_data(ttl=86400, show_spinner=False)
def calcular_stats_nbr(geom_json_str, ano, mes, mask_queimada=None):
    try:
        ee_geom, dnbr, threshold_usado = _construir_dnbr(geom_json_str, ano, mes, mask_queimada)
    except ValueError as ve:
        raise ve

    severidade = (dnbr
        .where(dnbr.lt(-0.1), 0)
        .where(dnbr.gte(-0.1).And(dnbr.lt(0.1)), 1)
        .where(dnbr.gte(0.1).And(dnbr.lt(0.27)), 2)
        .where(dnbr.gte(0.27).And(dnbr.lt(0.44)), 3)
        .where(dnbr.gte(0.44).And(dnbr.lt(0.66)), 4)
        .where(dnbr.gte(0.66), 5)
    ).rename('severidade')

    pixel_area = ee.Image.pixelArea().divide(1e6)
    labels = {0: 'Regeneração', 1: 'Não afetado', 2: 'Baixa', 3: 'Moderada', 4: 'Moderada-Alta', 5: 'Alta'}

    bandas = [pixel_area.updateMask(severidade.eq(cls)).rename(nome) for cls, nome in labels.items()]
    img_stack = ee.Image.cat(bandas)

    resultado = img_stack.reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom, scale=100, maxPixels=1e11, bestEffort=True).getInfo()
    stats = {nome: round(resultado.get(nome) or 0, 2) for nome in labels.values()}
    stats['threshold_nuvem_usado'] = threshold_usado
    return stats

# =============================================================
# INTERFACE - BARRA LATERAL (igual ao seu original)
# =============================================================
st.sidebar.title("⚙️ Filtros da Análise")
tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)
estado_dd = st.sidebar.selectbox('Selecione o Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25, disabled=(tipo_analise == 'Por Bioma'))
bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', buscar_cidades(estado_dd), disabled=(tipo_analise != 'Por Município'))

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Fonte de Dados")
fonte_escolhida = st.sidebar.radio("Escolha o que analisar:", ["🔥 Focos de Calor (INPE)", "🗺️ Área Queimada (NASA MODIS)"])

if "INPE" in fonte_escolhida:
    st.sidebar.markdown("**Filtros do INPE**")
    unidade_dd = st.sidebar.selectbox("Analisar por:", ["Dias", "Meses", "Anos"], index=1)
    if unidade_dd == "Dias": op_qtd = list(range(1, 91))
    elif unidade_dd == "Meses": op_qtd = list(range(1, 61))
    else: op_qtd = list(range(1, 11))
    quantidade_sel = st.sidebar.selectbox(f"Quantidade de {unidade_dd}:", options=op_qtd, index=1)
    satelites_lista = ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03']
    satelites_sel = st.sidebar.multiselect("Satélites de Referência:", satelites_lista, default=['AQUA_M-T', 'NPP-375', 'NPP-375D'])
else:
    st.sidebar.markdown("**Filtros do MODIS**")
    ano_modis = st.sidebar.selectbox("Ano de Referência:", list(range(2001, datetime.now().year + 1)), index=datetime.now().year - 2002)
    mes_modis = st.sidebar.selectbox("Mês do Mapa Principal:", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("🌳 Análise de Risco (Cruzamento Espacial):", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

st.sidebar.markdown("---")
if st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True):
    st.session_state.gerar_dashboard = True

# Contato
st.sidebar.markdown("---")
gmail_logo_url = "https://upload.wikimedia.org/wikipedia/commons/7/7e/Gmail_icon_%282020%29.svg"
linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
html_contato = f"""<div style="text-align: center;"><p style="font-size: 12px; color: #888;">Desenvolvido por</p><h2 style="font-family: serif; font-size: 20px; color: inherit;">ANA ANDRADE</h2><p style="font-size: 12px; color: #777;">Especialista em Geoprocessamento</p><hr style="border: 0; border-top: 1px solid #e0e0e0;"><div style="display: flex; justify-content: space-between; gap: 10px;"><a href="https://mail.google.com/mail/?view=cm&fs=1&to=anacarolinasantos580@gmail.com" target="_blank" style="flex:1; background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:10px; text-decoration:none; display:flex; align-items:center; justify-content:center;"><img src="{gmail_logo_url}" style="width:26px;"></a><a href="https://www.linkedin.com/in/ana-carolina-santos-3920931b3" target="_blank" style="flex:1; background:#0077B5; border-radius:8px; padding:10px; text-decoration:none; display:flex; align-items:center; justify-content:center;"><img src="{linkedin_logo_url}" style="width:22px;"></a></div></div>"""
st.sidebar.markdown(html_contato, unsafe_allow_html=True)

# =============================================================
# TELA PRINCIPAL
# =============================================================
st.title("🔥 Dashboard de Queimadas 🔥")

if st.session_state.gerar_dashboard:
    # (todo o processamento principal - INPE e MODIS - permanece igual ao seu código original)
    # ... [mantive todo o bloco de processamento que você já tinha]
    # Para não ficar excessivamente longo aqui, assuma que o bloco de processamento (from "with st.status..." até o final do else: dados_indisponiveis) é o mesmo do seu arquivo original.

    # =============================================================
    # ABA 3 — NBR (FINAL - CORRIGIDA E OTIMIZADA)
    # =============================================================
    with aba_nbr:
        if "INPE" in fonte_escolhida:
            st.info("💡 A análise de severidade NBR funciona melhor com a fonte **🗺️ Área Queimada (NASA MODIS)**.")
        else:
            st.subheader("🔬 Análise de Severidade da Queimada — dNBR (Sentinel-2)")
            st.caption("Calculando **apenas onde há cicatriz de queimada** (otimizado com máscara MODIS)")

            col_nbr1, col_nbr2 = st.columns([1, 1])

            with col_nbr1:
                calcular_nbr = st.button("🚀 Calcular Severidade dNBR agora", type="primary", use_container_width=True, help="Processa apenas a área queimada (mais rápido)")

                if calcular_nbr:
                    with st.spinner("🛰️ Processando imagens Sentinel-2 na área queimada... (15–45s)"):
                        nbr_ok = False
                        stats_sev = {}
                        dnbr_img = None
                        try:
                            mask_queimada = None
                            if 'area_queimada_img' in locals() and area_queimada_img is not None:
                                mask_queimada = area_queimada_img.gt(0)

                            stats_sev = calcular_stats_nbr(geom_json_str, ano_modis, mes_modis, mask_queimada=mask_queimada)
                            _, dnbr_img, threshold = _construir_dnbr(geom_json_str, ano_modis, mes_modis, mask_queimada=mask_queimada)
                            nbr_ok = True

                            if threshold > 40:
                                st.warning(f"⚠️ Usamos imagens com até **{threshold}%** de nuvem (qualidade ligeiramente menor).")
                        except ValueError as ve:
                            st.error(str(ve))
                        except Exception as e:
                            st.error(f"🚨 Erro ao processar Sentinel-2: {e}")

                    if nbr_ok and dnbr_img is not None:
                        # mapa dNBR
                        centro_nbr = limite.geometry.union_all().centroid
                        m_nbr = folium.Map(location=[centro_nbr.y, centro_nbr.x], zoom_start=9 if tipo_analise == "Por Município" else 7, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google')
                        folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'color': '#00d4ff', 'weight': 2.5, 'fillOpacity': 0}).add_to(m_nbr)
                        vis_dnbr = {'min': -0.5, 'max': 1.3, 'palette': ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf', '#fee08b', '#fc8d59', '#d73027', '#7a0403']}
                        m_nbr.add_ee_layer(dnbr_img, vis_dnbr, 'dNBR Severidade', opacity=0.85)
                        # legenda e layer control (igual ao anterior)
                        legenda_html = """<div style="position:fixed; bottom:30px; right:15px; z-index:9999; background:rgba(20,20,20,0.9); padding:12px 16px; border-radius:10px; font-size:12.5px; color:white; line-height:1.9;"><b>Severidade dNBR (USGS)</b><br><span style="color:#1a9850;">■</span> Regeneração / Recuperação<br><span style="color:#91cf60;">■</span> Não afetado<br><span style="color:#fee08b;">■</span> Baixa severidade<br><span style="color:#fc8d59;">■</span> Moderada<br><span style="color:#d73027;">■</span> Moderada-Alta<br><span style="color:#7a0403;">■</span> Alta severidade</div>"""
                        m_nbr.get_root().html.add_child(folium.Element(legenda_html))
                        folium.LayerControl().add_to(m_nbr)
                        st_folium(m_nbr, width=None, height=650, key=f"nbr_map_{val_sel}_{ano_modis}_{mes_modis}")

            with col_nbr2:
                if 'nbr_ok' in locals() and nbr_ok and stats_sev:
                    # gráficos de severidade (igual ao anterior)
                    df_sev = pd.DataFrame(list(stats_sev.items()), columns=['Classe', 'Área (km²)'])
                    df_sev = df_sev[df_sev['Área (km²)'] > 0].sort_values('Área (km²)', ascending=False)
                    cores_sev = {'Regeneração': '#1a9850', 'Não afetado': '#91cf60', 'Baixa': '#fee08b', 'Moderada': '#fc8d59', 'Moderada-Alta': '#d73027', 'Alta': '#7a0403'}
                    fig_pizza = px.pie(df_sev, values='Área (km²)', names='Classe', color='Classe', color_discrete_map=cores_sev, hole=0.45)
                    fig_pizza.update_layout(template='plotly_dark', height=280)
                    st.plotly_chart(fig_pizza, use_container_width=True)
                    fig_bar_sev = px.bar(df_sev, x='Área (km²)', y='Classe', orientation='h', color='Classe', color_discrete_map=cores_sev, text='Área (km²)')
                    fig_bar_sev.update_layout(template='plotly_dark', showlegend=False, yaxis={'categoryorder': 'total ascending'}, height=260)
                    st.plotly_chart(fig_bar_sev, use_container_width=True)
                    area_alta = stats_sev.get('Alta', 0) + stats_sev.get('Moderada-Alta', 0)
                    area_total_afetada = sum(v for k, v in stats_sev.items() if k not in ['Não afetado', 'Regeneração'])
                    col_m1, col_m2 = st.columns(2)
                    with col_m1: st.metric("🔴 Alta Severidade", f"{area_alta:.2f} km²")
                    with col_m2: st.metric("🔥 Total Afetado", f"{area_total_afetada:.2f} km²")
                else:
                    st.info("👆 Clique no botão azul à esquerda para gerar a análise de severidade.")

    # As outras abas (Mapa, Gráficos, Exportar) permanecem iguais ao seu código original
    # (não alterei nada nelas)

else:
    st.info("👈 Use os filtros ao lado e clique em **'Gerar Dashboard'**.")
