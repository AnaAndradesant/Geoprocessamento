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
import traceback
import numpy as np
import concurrent.futures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score





# Define o método para adicionar a camada do Earth Engine ao Folium
def add_ee_layer(self, ee_image_object, vis_params, name, opacity=1):
    # Gera as informações do tile a partir da imagem do EE
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    
    # Cria uma camada de mapa com os tiles obtidos do EE
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True,
        opacity=opacity
    ).add_to(self)

# Adiciona o método à classe folium.Map para que ele possa ser chamado como m.add_ee_layer
folium.Map.add_ee_layer = add_ee_layer

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


# =====================================================================
# FUNÇÕES DO SENTINEL-2 (NBR) - OTIMIZADAS COM A MÁSCARA DO MODIS
# =====================================================================

@st.cache_data(ttl=86400, show_spinner=False)
def calcular_stats_nbr(geom_json_str, ano, mes, _mascara_modis=None, area_km2_hint=0):
    # === LEITURA INTELIGENTE DE GEOJSON ===
    geom_dict = json.loads(geom_json_str)
    if 'features' in geom_dict:
        poly = ee.Geometry(geom_dict['features'][0]['geometry'])
    else:
        poly = ee.Geometry(geom_dict)
    # ======================================

    # === ESCALA E SIMPLIFICAÇÃO DINÂMICAS baseadas em area_km2_hint ===
    # area_km2_hint é calculado localmente via Shapely — zero chamadas GEE bloqueantes.
    # Amazônia (~5,5M km²): scale=1000 + simplify 10km evita timeout garantido.
    area_km2 = area_km2_hint
    if area_km2 > 1_500_000:       # Biomas gigantes (ex: Amazônia ~5,5M km²)
        scale = 1000
        max_error_simplify = 10000
        tile_scale = 16
    elif area_km2 > 500_000:       # Estados grandes (AM, PA, MT) / Cerrado
        scale = 500
        max_error_simplify = 5000
        tile_scale = 16
    elif area_km2 > 100_000:       # Estados médios
        scale = 200
        max_error_simplify = 1000
        tile_scale = 8
    elif area_km2 > 10_000:        # Estados pequenos / regiões
        scale = 100
        max_error_simplify = 500
        tile_scale = 4
    else:                          # Municípios e áreas pequenas
        scale = 50
        max_error_simplify = 100
        tile_scale = 2
    # ==================================================================

    # Simplifica a geometria no GEE — crítico para Amazônia (milhares de vértices)
    if area_km2 > 10_000:
        poly = poly.simplify(maxError=max_error_simplify)

    # Datas para o Sentinel (1 mês antes e o mês atual)
    data_fim = datetime(ano, mes, 28)
    data_ini = data_fim - timedelta(days=60)

    def get_nbr(img):
        return img.normalizedDifference(['B8', 'B12']).rename('nbr')

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")\
           .filterBounds(poly)\
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))

    pre_fire  = s2.filterDate(data_ini.strftime('%Y-%m-%d'),
                              (data_ini + timedelta(days=30)).strftime('%Y-%m-%d')).median()
    post_fire = s2.filterDate(data_fim.replace(day=1).strftime('%Y-%m-%d'),
                              data_fim.strftime('%Y-%m-%d')).median()

    dnbr = get_nbr(pre_fire).subtract(get_nbr(post_fire)).multiply(1000).clip(poly)

    if _mascara_modis is not None:
        dnbr = dnbr.updateMask(_mascara_modis.gt(0))

    sld_intervals = (
        dnbr.gt(-100).add(dnbr.gt(100)).add(dnbr.gt(270))
            .add(dnbr.gt(440)).add(dnbr.gt(660))
    )

    stats = sld_intervals.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=poly,
        scale=scale,
        maxPixels=1e13,
        tileScale=tile_scale,
        bestEffort=True
    ).getInfo()

    classes = {0: 'Regeneração', 1: 'Não afetado', 2: 'Baixa',
               3: 'Moderada', 4: 'Moderada-Alta', 5: 'Alta'}
    res_stats = {}

    if stats:
        hist = list(stats.values())[0]
        if isinstance(hist, dict):
            pixel_area_km2 = (scale * scale) / 1e6
            for k, v in hist.items():
                area = v * pixel_area_km2
                res_stats[classes.get(int(float(k)), 'Outros')] = round(area, 2)
        elif not hist:
            pass  # histograma vazio = sem pixels queimados no período

    return res_stats


# =============================================================
# --- FUNÇÕES UTILITÁRIAS ---
# =============================================================

def normalizar_texto(txt):
    """Remove acentos e converte para minúsculo para buscas tolerantes."""
    if pd.isna(txt):
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(txt))
        if unicodedata.category(c) != 'Mn'
    ).lower()

def gerar_excel(df):
    # 1. Cria uma cópia para não estragar os dados originais do dashboard
    df_export = df.copy()
    
    # 2. Varre as colunas procurando datas com Fuso Horário e remove o fuso
    for col in df_export.select_dtypes(include=['datetimetz']).columns:
        df_export[col] = df_export[col].dt.tz_localize(None)
        
    # 3. Varre as colunas procurando listas/dicionários (que o Excel também odeia) e vira texto
    for col in df_export.select_dtypes(include=['object']).columns:
        if any(isinstance(x, (list, dict)) for x in df_export[col].dropna()):
            df_export[col] = df_export[col].astype(str)

    # 4. Gera o arquivo Excel em memória
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Dados')
    
    return output.getvalue()

# =============================================================
# --- FUNÇÕES COM CACHE ---
# =============================================================

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def buscar_cotacao_dolar():
    """Busca a cotação PTAX do dólar no Banco Central. Cache de 1h."""
    from datetime import datetime, timedelta
    try:
        # Tenta os últimos 5 dias úteis para garantir que ache uma cotação
        for delta in range(5):
            data = (datetime.now() - timedelta(days=delta)).strftime("%m-%d-%Y")
            url = (
                f"https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
                f"CotacaoDolarDia(dataCotacao=@d)?@d='{data}'&$top=1&$format=json&$select=cotacaoVenda"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                valores = resp.json().get("value", [])
                if valores:
                    return round(float(valores[0]["cotacaoVenda"]), 2)
    except Exception:
        pass
    return 5.04  # fallback caso a API esteja indisponível

@st.cache_data(ttl=86400, show_spinner=False, persist="disk")
def buscar_cidades(uf):
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return sorted([d['nome'] for d in resp.json()])
    except:
        pass
    return ["Erro ao carregar cidades"]

@st.cache_data(show_spinner=False, persist="disk")
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

@st.cache_data(show_spinner=False, persist="disk")
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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_maxsize=20))
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

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
        muni_curinga = re.sub(r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni).replace(' ', '%')
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{muni_curinga}%'"

    dt_ini = datetime.strptime(d_ini, "%Y-%m-%d")
    dt_fim = datetime.strptime(d_fim, "%Y-%m-%d")
    sat_str = "','".join(satelites)

    # Monta a lista de blocos (dia a dia) ANTES de disparar as chamadas
    blocos = []
    cursor = dt_ini
    while cursor <= dt_fim:
        bloco_fim = min(cursor + timedelta(days=1), dt_fim)
        blocos.append((cursor, bloco_fim))
        cursor = bloco_fim + timedelta(days=1)

    def _buscar_bloco(bloco):
        b_ini, b_fim = bloco
        cql = (
            f"data_hora_gmt >= '{b_ini.strftime('%Y-%m-%d')}T00:00:00' "
            f"AND data_hora_gmt <= '{b_fim.strftime('%Y-%m-%d')}T23:59:59' "
            f"AND satelite IN ('{sat_str}') AND {filtro_base}"
        )
        try:
            r = session.get(
                url,
                params={
                    "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                    "typeName": "bdqueimadas:focos", "outputFormat": "application/json",
                    "CQL_FILTER": cql, "maxFeatures": 50000
                },
                headers=headers, verify=False, timeout=90
            )
            if r.status_code == 200:
                dados_json = r.json()
                if "features" in dados_json and len(dados_json["features"]) > 0:
                    return [
                        {"longitude": f["geometry"]["coordinates"][0],
                         "latitude": f["geometry"]["coordinates"][1],
                         **f["properties"]}
                        for f in dados_json["features"]
                    ]
        except Exception:
            pass  # Ignora erros de conexão para não parar as outras chamadas
        return []

    # Chamadas em paralelo (são independentes — não precisa esperar uma terminar
    # pra começar a próxima). 12 workers é um bom equilíbrio entre velocidade e
    # não sobrecarregar o servidor do INPE.
    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        for registros in executor.map(_buscar_bloco, blocos):
            if registros:
                all_dfs.append(pd.DataFrame(registros))

    if not all_dfs:
        return pd.DataFrame()

    df_final = pd.concat(all_dfs, ignore_index=True)
    if 'id' in df_final.columns:
        df_final = df_final.drop_duplicates(subset=['id'])
    else:
        df_final = df_final.drop_duplicates()
    return df_final


# =====================================================================
# 🎯 MODELO PREDITIVO DE RISCO DE QUEIMADA (ML)
# =====================================================================
# Ideia: usar o histórico de focos (INPE) + clima diário (NASA POWER, API
# pública e gratuita) para treinar, sob demanda, um modelo simples e
# interpretável (Regressão Logística) que estima a probabilidade de
# ocorrência de foco de calor na região selecionada nos próximos dias.
# O modelo é treinado on-the-fly para a região escolhida nos filtros e
# fica em cache por 24h (st.cache_data) para não retreinar a cada clique.
# =====================================================================

FEATURES_CLIMA = [
    "T2M", "RH2M", "PRECTOTCORR", "WS10M", "dias_secos_consecutivos",
    "temp_media_3d", "umidade_media_3d", "chuva_acumulada_3d",
    "temp_media_7d", "umidade_media_7d", "chuva_acumulada_7d",
    "mes_sin", "mes_cos",
]

# Feature adicional de "efeito fixo por município": a taxa histórica de dias-com-
# ocorrência daquele município específico, calculada só com o período de TREINO
# (sem vazamento de dados). Captura risco basal que não depende do clima do dia
# (ex: uso do solo, fronteira agrícola, proximidade de estradas) — dois municípios
# com o mesmo clima podem ter risco de base muito diferente por essas razões.
FEATURES_RISCO = FEATURES_CLIMA + ["taxa_hist_municipio"]


@st.cache_data(ttl=86400, show_spinner=False, persist="disk")
def buscar_clima_nasa_power(lat, lon, d_ini, d_fim):
    """
    Clima diário (temp, umidade, chuva, vento) via NASA POWER — API pública, sem chave.
    NUNCA levanta exceção: se a chamada falhar (rede instável, timeout, erro do servidor),
    retorna um DataFrame vazio. Isso é proposital — uma falha isolada em 1 município não
    pode derrubar o treino/mapa inteiro, que depende de várias chamadas independentes.
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,RH2M,PRECTOTCORR,WS10M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": d_ini.replace("-", ""),
        "end": d_fim.replace("-", ""),
        "format": "JSON",
    }
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=20))

    try:
        r = session.get(url, params=params, timeout=60, verify=False)
        r.raise_for_status()
        payload = r.json()["properties"]["parameter"]
        df = pd.DataFrame(payload)
    except Exception:
        return pd.DataFrame(columns=["data", "T2M", "RH2M", "PRECTOTCORR", "WS10M"])

    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.rename_axis("data").reset_index()
    df[["T2M", "RH2M", "PRECTOTCORR", "WS10M"]] = df[
        ["T2M", "RH2M", "PRECTOTCORR", "WS10M"]
    ].replace(-999, np.nan)
    return df


@st.cache_data(ttl=86400, show_spinner=False, persist="disk")
def buscar_historico_focos_diario(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, dias_bloco=7):
    """
    Versão 'leve' de buscar_focos_inpe, otimizada para treino do modelo:
    busca em blocos de N dias (em vez de 1) e traz só a data de cada foco,
    reduzindo bastante o número de chamadas à API do INPE para períodos longos.
    Retorna a contagem diária de focos na região.
    """
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_maxsize=20))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

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
    else:
        muni_curinga = re.sub(r'[aeiouáéíóúãõâêîôûAEIOUÁÉÍÓÚÃÕÂÊÎÔÛ]', '%', val_muni).replace(' ', '%')
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{muni_curinga}%'"

    dt_ini = datetime.strptime(d_ini, "%Y-%m-%d")
    dt_fim = datetime.strptime(d_fim, "%Y-%m-%d")

    blocos = []
    cursor = dt_ini
    while cursor <= dt_fim:
        bloco_fim = min(cursor + timedelta(days=dias_bloco), dt_fim)
        blocos.append((cursor, bloco_fim))
        cursor = bloco_fim + timedelta(days=1)

    def _buscar_bloco(bloco):
        b_ini, b_fim = bloco
        cql = (
            f"data_hora_gmt >= '{b_ini.strftime('%Y-%m-%d')}T00:00:00' "
            f"AND data_hora_gmt <= '{b_fim.strftime('%Y-%m-%d')}T23:59:59' "
            f"AND {filtro_base}"
        )
        try:
            r = session.get(
                url,
                params={
                    "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                    "typeName": "bdqueimadas:focos", "outputFormat": "application/json",
                    "propertyName": "data_hora_gmt",
                    "CQL_FILTER": cql, "maxFeatures": 50000
                },
                headers=headers, verify=False, timeout=90
            )
            if r.status_code == 200:
                dados = r.json()
                if dados.get("features"):
                    return [f["properties"]["data_hora_gmt"][:10] for f in dados["features"]]
        except Exception:
            pass
        return []

    todas_datas = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for datas_bloco in executor.map(_buscar_bloco, blocos):
            todas_datas.extend(datas_bloco)

    if not todas_datas:
        return pd.DataFrame(columns=["data", "n_focos"])

    df = pd.Series(todas_datas, name="data").value_counts().rename_axis("data").reset_index(name="n_focos")
    df["data"] = pd.to_datetime(df["data"])
    return df


def _engenharia_features_risco(df_clima):
    """Cria features de sazonalidade, médias móveis e dias secos consecutivos.
    Usa shift(1) para garantir que só usamos informação disponível ATÉ ontem
    ao prever o risco de hoje/amanhã (evita vazamento de dados)."""
    df_clima = df_clima.sort_values("data").reset_index(drop=True)

    sem_chuva = df_clima["PRECTOTCORR"].fillna(0) < 1.0
    grupo = (~sem_chuva).cumsum()
    df_clima["dias_secos_consecutivos"] = sem_chuva.groupby(grupo).cumsum()

    for janela in [3, 7]:
        df_clima[f"temp_media_{janela}d"] = df_clima["T2M"].rolling(janela, min_periods=1).mean()
        df_clima[f"umidade_media_{janela}d"] = df_clima["RH2M"].rolling(janela, min_periods=1).mean()
        df_clima[f"chuva_acumulada_{janela}d"] = df_clima["PRECTOTCORR"].rolling(janela, min_periods=1).sum()

    df_clima["mes"] = df_clima["data"].dt.month
    df_clima["mes_sin"] = np.sin(2 * np.pi * df_clima["mes"] / 12)
    df_clima["mes_cos"] = np.cos(2 * np.pi * df_clima["mes"] / 12)

    cols_shift = [c for c in df_clima.columns if c not in ("data", "mes", "mes_sin", "mes_cos")]
    df_clima[cols_shift] = df_clima[cols_shift].shift(1)
    return df_clima


@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _carregar_todos_municipios_brasil():
    """
    Carrega os ~5.570 municípios do Brasil inteiro UMA VEZ (cache de 30 dias —
    fronteiras municipais praticamente não mudam). Compartilhado entre todos os
    biomas, em vez de recarregar o Brasil inteiro toda vez que o bioma muda.
    """
    return read_municipality(code_muni="all", year=2020)


@st.cache_data(ttl=604800, show_spinner=False, persist="disk")
def obter_municipios_regiao(tipo_analise, estado_dd, bioma_dd, municipio_dd, max_municipios=50, seed=42):
    """
    Retorna os municípios que compõem a região selecionada (estado inteiro,
    bioma inteiro, ou o próprio município), já com lat/lon do centro de cada um.
    Se a região tiver mais municípios que max_municipios, uma amostra aleatória
    (com seed fixa, reprodutível) é usada — necessário para biomas enormes como
    a Amazônia, que têm centenas de municípios.
    """
    if tipo_analise == "Por Município":
        gdf = read_municipality(code_muni=estado_dd, year=2020)
        busca = normalizar_texto(municipio_dd.strip())
        gdf['nome_norm'] = gdf['name_muni'].apply(normalizar_texto)
        gdf = gdf[gdf['nome_norm'].str.contains(busca)]
    elif tipo_analise == "Por Estado":
        gdf = read_municipality(code_muni=estado_dd, year=2020)
    else:  # Por Bioma
        gdf_bioma = read_biomes(year=2019)
        gdf_bioma = gdf_bioma[gdf_bioma['name_biome'] == bioma_dd]
        gdf_todos = _carregar_todos_municipios_brasil()
        gdf_bioma = gdf_bioma.to_crs(gdf_todos.crs)
        gdf = gpd.sjoin(gdf_todos, gdf_bioma[['geometry']], predicate='intersects', how='inner')
        gdf = gdf.drop(columns=['index_right'], errors='ignore').drop_duplicates(subset=['code_muni'])

    if gdf.empty:
        return gdf

    gdf = gdf.to_crs("EPSG:4326")

    # Amostra ANTES de simplificar geometria e calcular representative_point —
    # evita gastar tempo processando geometria de municípios que serão descartados.
    total_disponivel = len(gdf)
    amostrado = total_disponivel > max_municipios
    if amostrado:
        gdf = gdf.sample(n=max_municipios, random_state=seed)

    gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    pontos = gdf.geometry.representative_point()
    gdf['lat'] = pontos.y
    gdf['lon'] = pontos.x

    gdf['total_disponivel'] = total_disponivel
    gdf['amostrado_flag'] = amostrado
    cols = ['name_muni', 'abbrev_state', 'lat', 'lon', 'geometry', 'total_disponivel', 'amostrado_flag']
    return gdf[[c for c in cols if c in gdf.columns]].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False, persist="disk")
def buscar_burndate_diario_modis(geom_json_str, d_ini, d_fim):
    """
    Equivalente ao 'buscar_historico_focos_diario', mas usando ÁREA QUEIMADA
    (MODIS MCD64A1, banda BurnDate) em vez de focos de calor do INPE.
    A banda BurnDate traz o dia-do-ano em que cada pixel queimou dentro do mês,
    então dá pra reconstruir uma contagem diária de pixels queimados — o
    equivalente ao 'n_focos', mas baseado em área detectada, não em ponto de calor.

    NOTA: mais lento que a versão INPE (cada chamada ao Earth Engine custa mais
    que uma consulta WFS simples), por isso os parâmetros de amostra devem ser
    menores quando essa fonte é usada.
    """
    poly = ee.Geometry(json.loads(geom_json_str))
    dt_ini = datetime.strptime(d_ini, "%Y-%m-%d")
    dt_fim = datetime.strptime(d_fim, "%Y-%m-%d")

    meses = []
    cursor = dt_ini.replace(day=1)
    while cursor <= dt_fim:
        prox_mes = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        meses.append((cursor, prox_mes))
        cursor = prox_mes

    def _buscar_mes(bloco_mes):
        mes_ini, mes_fim = bloco_mes
        try:
            img = (
                ee.ImageCollection('MODIS/061/MCD64A1')
                .filterDate(ee.Date(mes_ini.strftime("%Y-%m-%d")), ee.Date(mes_fim.strftime("%Y-%m-%d")))
                .select('BurnDate')
                .max()
                .clip(poly)
            )
            stats = img.reduceRegion(
                reducer=ee.Reducer.frequencyHistogram(),
                geometry=poly, scale=500, maxPixels=1e13, bestEffort=True
            ).getInfo()
            hist = (stats or {}).get('BurnDate', {})
            registros_mes = []
            for dia_str, contagem_px in hist.items():
                try:
                    dia_ano = int(float(dia_str))
                    if dia_ano <= 0:
                        continue  # 0 = pixel não queimado
                    data_real = datetime(mes_ini.year, 1, 1) + timedelta(days=dia_ano - 1)
                    if dt_ini.date() <= data_real.date() <= dt_fim.date():
                        registros_mes.append((data_real.date(), int(contagem_px)))
                except Exception:
                    continue
            return registros_mes
        except Exception:
            return []

    todos_registros = []
    # Um mês é independente do outro no Earth Engine também — paraleliza,
    # mas com menos workers (a API do GEE é mais sensível a excesso de chamadas simultâneas)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for registros_mes in executor.map(_buscar_mes, meses):
            todos_registros.extend(registros_mes)

    if not todos_registros:
        return pd.DataFrame(columns=["data", "n_focos"])

    df = pd.DataFrame(todos_registros, columns=["data", "n_focos"])
    df["data"] = pd.to_datetime(df["data"])
    df = df.groupby("data", as_index=False)["n_focos"].sum()
    return df


@st.cache_data(ttl=86400, show_spinner=False, persist="disk")
def treinar_modelo_risco_regional(tipo_analise, estado_dd, bioma_dd, municipio_dd,
                                   anos_historico=2, n_amostra_treino=8, fonte_dados="INPE"):
    """
    Treina UM modelo (Regressão Logística) usando dados agrupados de vários
    municípios da região selecionada (não mais um único ponto central).
    Isso torna o modelo representativo do bioma/estado inteiro, e não só do
    centroide. Usa TODOS OS DIAS do período de histórico para cada município
    amostrado (clima + presença/ausência de ocorrência), não uma amostra de dias.

    fonte_dados:
      - "INPE"  -> alvo = houve foco de calor (detecção pontual) naquele dia
      - "MODIS" -> alvo = houve pixel de área queimada detectado naquele dia
    """
    gdf_treino = obter_municipios_regiao(tipo_analise, estado_dd, bioma_dd, municipio_dd,
                                          max_municipios=n_amostra_treino)
    if gdf_treino.empty:
        return {"erro": "Não foi possível localizar municípios para esta região."}

    hoje = datetime.now()
    d_fim = (hoje - timedelta(days=2)).strftime("%Y-%m-%d")
    d_ini = (hoje - timedelta(days=365 * anos_historico)).strftime("%Y-%m-%d")

    def _montar_dados_municipio(row):
        # Qualquer falha aqui (rede, dado malformado, etc.) descarta só ESTE
        # município — nunca deve derrubar o treino inteiro por causa de 1 falha.
        try:
            df_clima = buscar_clima_nasa_power(row['lat'], row['lon'], d_ini, d_fim)
            if df_clima.empty or df_clima["T2M"].isna().all():
                return None
            if fonte_dados == "MODIS":
                geom_muni_str = json.dumps(row['geometry'].__geo_interface__, sort_keys=True)
                df_ocorrencia = buscar_burndate_diario_modis(geom_muni_str, d_ini, d_fim)
            else:
                df_ocorrencia = buscar_historico_focos_diario(
                    "Por Município", row['abbrev_state'], bioma_dd, row['name_muni'], d_ini, d_fim, dias_bloco=14
                )
            df_m = _engenharia_features_risco(df_clima)
            df_m = df_m.merge(df_ocorrencia, on="data", how="left")
            df_m["n_focos"] = df_m["n_focos"].fillna(0)
            df_m["target_foco"] = (df_m["n_focos"] > 0).astype(int)
            df_m["municipio"] = row['name_muni']
            return df_m
        except Exception:
            return None

    # Um município é independente do outro — busca em paralelo. Com MODIS,
    # cada chamada ao Earth Engine é mais pesada, então usamos menos workers.
    # _montar_dados_municipio nunca levanta exceção (retorna None em falha),
    # então uma falha isolada não derruba o ThreadPoolExecutor inteiro.
    frames = []
    linhas_municipios = [row for _, row in gdf_treino.iterrows()]
    max_workers_muni = 3 if fonte_dados == "MODIS" else 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers_muni) as executor:
        for df_m in executor.map(_montar_dados_municipio, linhas_municipios):
            if df_m is not None:
                frames.append(df_m)

    n_municipios_falha = len(linhas_municipios) - len(frames)

    if not frames:
        return {"erro": "Não foi possível obter dados para os municípios amostrados. Tente novamente."}

    df = pd.concat(frames, ignore_index=True).dropna(subset=FEATURES_CLIMA)
    if len(df) < 100 or df["target_foco"].nunique() < 2:
        fonte_txt = "área queimada (MODIS)" if fonte_dados == "MODIS" else "focos (INPE)"
        return {"erro": f"Histórico insuficiente ou sem variação de {fonte_txt} para treinar um modelo confiável nesta região/período."}

    # Baseline ingênuo (sem ML, sem clima) pra comparação: "risco alto se o
    # município teve pelo menos 1 dia com ocorrência nos 7 dias ANTERIORES a
    # este" — pura persistência recente. shift(1) garante que não olha o
    # próprio dia (mesma regra de não-vazamento usada no resto do pipeline).
    df = df.sort_values(["municipio", "data"]).reset_index(drop=True)
    df["baseline_focos_7d"] = (
        df.groupby("municipio")["target_foco"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
    )
    df["pred_baseline"] = (df["baseline_focos_7d"].fillna(0) > 0).astype(int)

    # Split TEMPORAL pooled: mesmo corte de data para todos os municípios
    # (treina no passado de todos eles, testa no período mais recente de todos eles)
    datas_unicas = sorted(df["data"].unique())
    corte_data = datas_unicas[int(len(datas_unicas) * 0.8)]
    treino, teste = df[df["data"] < corte_data].copy(), df[df["data"] >= corte_data].copy()

    # Efeito fixo por município: taxa histórica de dias-com-ocorrência DAQUELE
    # município específico, calculada SÓ com o período de treino (para não vazar
    # informação do teste). Aplicada como valor constante por município tanto no
    # treino quanto no teste — captura risco de base (uso do solo, etc.) que o
    # clima sozinho não explica.
    taxa_por_municipio = treino.groupby("municipio")["target_foco"].mean().to_dict()
    taxa_geral_fallback = float(treino["target_foco"].mean())
    treino["taxa_hist_municipio"] = treino["municipio"].map(taxa_por_municipio)
    teste["taxa_hist_municipio"] = teste["municipio"].map(taxa_por_municipio).fillna(taxa_geral_fallback)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(treino[FEATURES_RISCO])
    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(X_train, treino["target_foco"])

    auc = None
    avaliacao_teste = None
    if teste["target_foco"].nunique() == 2 and len(teste) > 0:
        X_test = scaler.transform(teste[FEATURES_RISCO])
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_true = teste["target_foco"].to_numpy()
        auc = roc_auc_score(y_true, y_prob)
        # Guarda tudo que o painel de transparência do modelo precisa: rótulo
        # verdadeiro, probabilidade prevista pelo modelo e previsão do baseline
        # ingênuo — tudo do MESMO conjunto de teste (nunca visto no treino).
        avaliacao_teste = {
            "y_true": y_true,
            "y_prob": y_prob,
            "y_pred_baseline": teste["pred_baseline"].to_numpy(),
        }

    # Climatologia mensal: taxa histórica média de dias-com-ocorrência por mês,
    # calculada com TODOS os dias observados (usada para a estimativa "próximo mês")
    climatologia_mensal = df.groupby("mes")["target_foco"].mean().to_dict()

    return {
        "modelo": modelo, "scaler": scaler, "auc": auc,
        "avaliacao_teste": avaliacao_teste,
        "fonte_dados": fonte_dados,
        "climatologia_mensal": climatologia_mensal,
        "taxa_por_municipio": taxa_por_municipio,
        "taxa_geral_fallback": taxa_geral_fallback,
        "municipios_treino": sorted(df["municipio"].unique().tolist()),
        "n_municipios_treino": df["municipio"].nunique(),
        "n_dias_treino": len(treino), "n_dias_teste": len(teste),
        "n_linhas_total": len(df),
        "taxa_base": df["target_foco"].mean(),
        "amostrado_treino": bool(gdf_treino["amostrado_flag"].iloc[0]),
        "n_municipios_falha": n_municipios_falha,
        "total_disponivel": int(gdf_treino["total_disponivel"].iloc[0]),
    }


@st.cache_data(ttl=21600, show_spinner=False, persist="disk")
def buscar_condicoes_atuais(lat, lon):
    """Busca só os últimos ~35 dias de clima (leve, rápido) para calcular o
    risco ATUAL de um ponto específico — usado para colorir cada município no mapa.
    Nunca levanta exceção: falha isolada em 1 município não pode derrubar o mapa inteiro."""
    try:
        hoje = datetime.now()
        d_fim = (hoje - timedelta(days=2)).strftime("%Y-%m-%d")
        d_ini = (hoje - timedelta(days=35)).strftime("%Y-%m-%d")
        df_clima = buscar_clima_nasa_power(lat, lon, d_ini, d_fim)
        if df_clima.empty or df_clima["T2M"].isna().all():
            return None
        df_feat = _engenharia_features_risco(df_clima).dropna(subset=FEATURES_CLIMA)
        if df_feat.empty:
            return None
        return df_feat.iloc[[-1]]
    except Exception:
        return None


def _logit(p, eps=1e-4):
    """Converte probabilidade para log-odds. Evita explosão numérica perto de 0 ou 1."""
    p = min(max(p, eps), 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(x):
    """Inverso do logit: converte log-odds de volta para probabilidade (0-1)."""
    return 1 / (1 + np.exp(-x))


def classificar_risco(prob):
    if prob < 0.25:
        return "Baixo", "🟢", "#2ecc71"
    elif prob < 0.50:
        return "Moderado", "🟡", "#f1c40f"
    elif prob < 0.75:
        return "Alto", "🟠", "#e67e22"
    else:
        return "Crítico", "🔴", "#e74c3c"


@st.cache_data
def calcular_area_queimada_modis(geom_json, ano, mes=None):
    poly = ee.Geometry(json.loads(geom_json)['features'][0]['geometry'])
    dataset = ee.ImageCollection('MODIS/061/MCD64A1').filterBounds(poly)
    
    if mes:
        img = dataset.filter(ee.Filter.calendarRange(ano, ano, 'year'))\
                     .filter(ee.Filter.calendarRange(mes, mes, 'month')).max()
        burned = img.select('BurnDate').clip(poly)
        
        # Correção do Efeito Fantasma
        doy_inicio = datetime(ano, mes, 1).timetuple().tm_yday
        ultimo_dia = calendar.monthrange(ano, mes)[1]
        doy_fim = datetime(ano, mes, ultimo_dia).timetuple().tm_yday
        
        mask = burned.gte(doy_inicio).And(burned.lte(doy_fim))
        burned = burned.updateMask(mask)
    else:
        img = dataset.filter(ee.Filter.calendarRange(ano, ano, 'year')).max()
        burned = img.select('BurnDate').clip(poly)
    
    area_img = ee.Image.pixelArea().updateMask(burned.gt(0))
    stats = area_img.reduceRegion(reducer=ee.Reducer.sum(), geometry=poly, scale=500, maxPixels=1e9)
    area_km2 = ee.Number(stats.get('area')).divide(1e6).getInfo()
    
    return burned, area_km2 or 0


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_total_modis(geom_json_str, ano, mes):
    """Busca area queimada total MODIS. Cache de 1h por geometria+periodo."""
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    ee_geom_simple = ee_geom.simplify(maxError=10000)
    data_ini = ee.Date.fromYMD(ano, mes, 1)
    colecao = (
        ee.ImageCollection('MODIS/061/MCD64A1')
        .filterDate(data_ini, data_ini.advance(1, 'month'))
        .filterBounds(ee_geom_simple)
    )
    if colecao.size().getInfo() == 0:
        return None, 0.0  # dados indisponiveis
    img = colecao.select('BurnDate').max().clip(ee_geom_simple)
    stats = (
        ee.Image.pixelArea().divide(1e6)
        .updateMask(img.gt(0))
        .rename('area_km2')
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom_simple,
                      scale=1000, maxPixels=1e13, bestEffort=True)
        .getInfo()
    )
    area = round(stats.get('area_km2') or 0, 2)
    return img, area


@st.cache_data(ttl=86400, show_spinner=False)
def calcular_anomalia_modis(geom_json_str, ano_ref):
    ee_geom = ee.Geometry(json.loads(geom_json_str))

    # Simplifica a geometria UMA VEZ fora do loop — evita recriá-la a cada imagem
    geom_simplificada = ee_geom.simplify(maxError=5000)

    anos_historico = list(range(2001, ano_ref))
    anos_ee = ee.List(anos_historico)
    n_anos = len(anos_historico)

    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }

    def get_area_km2(img):
        raw = (
            ee.Image.pixelArea().divide(1e6)
            .updateMask(img.gt(0))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geom_simplificada,
                scale=10000,
                maxPixels=1e13,
                tileScale=16,
                bestEffort=True
            ).get('area')
        )
        return ee.Algorithms.If(raw, raw, 0)

    def calc_mes_feature(mes):
        mes_n = ee.Number(mes)
        ini_ref = ee.Date.fromYMD(ano_ref, mes_n, 1)
        img_ref = (
            ee.ImageCollection('MODIS/061/MCD64A1')
            .filterDate(ini_ref, ini_ref.advance(1, 'month'))
            .filterBounds(geom_simplificada)
            .select('BurnDate').max().clip(geom_simplificada)
        )
        area_ref = ee.Number(get_area_km2(img_ref))

        def area_ano_hist(ano):
            ano_n = ee.Number(ano)
            ini_h = ee.Date.fromYMD(ano_n, mes_n, 1)
            img_h = (
                ee.ImageCollection('MODIS/061/MCD64A1')
                .filterDate(ini_h, ini_h.advance(1, 'month'))
                .filterBounds(geom_simplificada)
                .select('BurnDate').max().clip(geom_simplificada)
            )
            return get_area_km2(img_h)

        areas_hist = anos_ee.map(area_ano_hist)
        soma = areas_hist.iterate(
            lambda cur, acc: ee.Number(acc).add(ee.Number(cur)),
            ee.Number(0)
        )
        media_hist = ee.Number(soma).divide(ee.Number(n_anos))

        return ee.Feature(None, {
            'mes': mes_n,
            'area_ref': area_ref,
            'media_hist': media_hist
        })

    # ── PROCESSAMENTO PARALELO (4 meses simultâneos) ───────────
    # Reduz de ~60s sequencial para ~15s com 4 workers paralelos
    from concurrent.futures import ThreadPoolExecutor, as_completed
    MAX_RETRIES = 4
    registros = []

    def processar_mes(mes):
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                feat_resultado = ee.Feature(calc_mes_feature(mes)).getInfo()
                p = feat_resultado['properties']
                val_ref  = round(float(p.get('area_ref')  or 0), 2)
                media    = round(float(p.get('media_hist') or 0), 2)
                anomalia = round(((val_ref - media) / media * 100), 1) if media > 0 else 0
                return {
                    'Mês': mes,
                    'Mês Nome': meses_map[mes],
                    f'Área {ano_ref} (km²)': val_ref,
                    'Média Histórica (km²)': media,
                    'Anomalia (%)': anomalia
                }
            except Exception as e:
                msg = str(e)
                if ('Too many concurrent' in msg or '429' in msg) and tentativa < MAX_RETRIES:
                    time.sleep(2 ** tentativa)
                    continue
                return {
                    'Mês': mes, 'Mês Nome': meses_map[mes],
                    f'Área {ano_ref} (km²)': 0,
                    'Média Histórica (km²)': 0, 'Anomalia (%)': 0
                }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(processar_mes, mes): mes for mes in range(1, 13)}
        for future in as_completed(futures):
            registros.append(future.result())

    return pd.DataFrame(sorted(registros, key=lambda x: x['Mês']))

def _construir_dnbr(geom_json_str, ano, mes, _mascara_modis=None, area_km2_hint=0):
    # 1. LEITURA CORRETA DA GEOMETRIA
    geom_dict = json.loads(geom_json_str)
    if 'features' in geom_dict:
        poly = ee.Geometry(geom_dict['features'][0]['geometry'])
    else:
        poly = ee.Geometry(geom_dict)

    # 1b. SIMPLIFICAÇÃO DA GEOMETRIA — crítico para Amazônia
    if area_km2_hint > 1_500_000:
        poly = poly.simplify(maxError=10000)
    elif area_km2_hint > 500_000:
        poly = poly.simplify(maxError=5000)
    elif area_km2_hint > 10_000:
        poly = poly.simplify(maxError=1000)

    # 2. JANELA TEMPORAL EXPANDIDA (Garante que sempre ache imagens limpas)
    data_ref = datetime(ano, mes, 1)
    data_ini_pre = data_ref - timedelta(days=90)  # 3 meses antes
    data_fim_pre = data_ref
    
    data_ini_pos = data_ref
    data_fim_pos = data_ref + timedelta(days=60)  # 2 meses depois

    def get_nbr(img):
        return img.normalizedDifference(['B8', 'B12']).rename('nbr')

    # 3. BUSCA DO SENTINEL (Sem filtro estrito de nuvens para não esvaziar a coleção)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poly)
    
    col_pre = s2.filterDate(data_ini_pre.strftime('%Y-%m-%d'), data_fim_pre.strftime('%Y-%m-%d'))
    col_pos = s2.filterDate(data_ini_pos.strftime('%Y-%m-%d'), data_fim_pos.strftime('%Y-%m-%d'))

    # 4. TRAVA DE SEGURANÇA SILENCIOSA
    # Removidas as chamadas .size().getInfo() — eram bloqueantes e lentas em regiões grandes.
    # Se não houver imagens, o .median() retornará uma imagem vazia e o try/except externo captura.
    try:
        pre_fire  = col_pre.median()
        post_fire = col_pos.median()
    except Exception:
        return poly, ee.Image().constant(0).updateMask(0)
    
    # Cálculo bruto do dNBR
    dnbr = get_nbr(pre_fire).subtract(get_nbr(post_fire)).multiply(1000).clip(poly)
    
    # Aplica a máscara do MODIS (Filtro de Fogo)
    if _mascara_modis is not None:
        dnbr = dnbr.updateMask(_mascara_modis.gt(0))
        
    # RETORNA O DNBR ORIGINAL!
    # Isso fará as cores do mapa combinarem perfeitamente com as do gráfico
    return poly, dnbr

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
    # Congela os filtros no momento do clique — mudar um widget depois (nesta ou em
    # outra aba) não deve regenerar o dashboard sozinho, só um novo clique aqui.
    st.session_state.filtros_ativos = {
        "tipo_analise": tipo_analise, "estado_dd": estado_dd, "bioma_dd": bioma_dd,
        "municipio_dd": municipio_dd, "fonte_escolhida": fonte_escolhida,
        "area_protegida": area_protegida,
    }
    if "INPE" in fonte_escolhida:
        st.session_state.filtros_ativos.update({
            "unidade_dd": unidade_dd, "quantidade_sel": quantidade_sel, "satelites_sel": satelites_sel,
        })
    else:
        st.session_state.filtros_ativos.update({
            "ano_modis": ano_modis, "mes_modis": mes_modis,
        })

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
    # Usa os filtros CONGELADOS no momento do último clique em "Gerar Dashboard" —
    # não os valores atuais dos widgets (que podem ter mudado nesse meio tempo).
    _f = st.session_state.filtros_ativos
    tipo_analise = _f["tipo_analise"]
    estado_dd = _f["estado_dd"]
    bioma_dd = _f["bioma_dd"]
    municipio_dd = _f["municipio_dd"]
    fonte_escolhida = _f["fonte_escolhida"]
    area_protegida = _f["area_protegida"]
    if "INPE" in fonte_escolhida:
        unidade_dd = _f["unidade_dd"]
        quantidade_sel = _f["quantidade_sel"]
        satelites_sel = _f["satelites_sel"]
    else:
        ano_modis = _f["ano_modis"]
        mes_modis = _f["mes_modis"]

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

        # Área calculada localmente via GeoPandas/Shapely — sem chamada GEE bloqueante
        _limite_proj = limite.to_crs("EPSG:6933")  # projeção equal-area
        area_km2_local = float(_limite_proj.geometry.union_all().area / 1e6)

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

                if area_protegida != "Nenhuma":
                    st.write(f"🌳 Carregando limites de {area_protegida}...")
                    gdf_areas = carregar_areas_protegidas(area_protegida)
                    gdf_areas = gdf_areas.to_crs(gdf.crs)

                    # Todas as áreas protegidas que TOCAM a região selecionada —
                    # usadas pra desenhar o limite no mapa, mesmo que não tenham
                    # nenhum foco dentro. Antes, o limite só aparecia quando
                    # havia foco, então uma área "limpa" nunca era desenhada.
                    gdf_areas_regiao = gpd.sjoin(gdf_areas, limite, predicate='intersects')
                    if 'index_right' in gdf_areas_regiao.columns:
                        gdf_areas_regiao = gdf_areas_regiao.drop(columns=['index_right'])
                    areas_afetadas = (
                        gdf_areas_regiao[['nome_area', 'geometry']]
                        .drop_duplicates(subset='nome_area')
                        .reset_index(drop=True)
                    )

                    if 'index_right' in gdf.columns:
                        gdf = gdf.drop(columns=['index_right'])

                    gdf_focos_risco = gpd.sjoin(gdf, gdf_areas, predicate='within')

                    if not gdf_focos_risco.empty:
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
                        # Sem focos dentro das áreas protegidas — zera os PONTOS,
                        # mas mantém 'areas_afetadas' com os limites já carregados
                        # acima, pra continuar desenhando as áreas no mapa.
                        df_rec = pd.DataFrame()
                        total_valor = 0

        # -------------------------------------------------------
        # FONTE: MODIS
        # -------------------------------------------------------
        else:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                # Usa funcao cacheada -- segunda consulta ao mesmo periodo e instantanea
                area_queimada_img, total_valor = buscar_total_modis(
                    geom_json_str, ano_modis, mes_modis
                )
                if area_queimada_img is None:
                    dados_indisponiveis = True
                    total_valor = 0
                else:
                    ee_geom_simple = ee_geom_complex.simplify(maxError=10000)
                    area_queimada_img = ee.Image(area_queimada_img)
                    img_area_km2 = (
                        ee.Image.pixelArea().divide(1000000)
                        .updateMask(area_queimada_img.gt(0))
                        .rename('area_km2')
                    )

                    if area_protegida != "Nenhuma" and total_valor > 0:
                        st.write(f"🌳 Isolando km² afetados em {area_protegida}...")
                        gdf_areas_br = carregar_areas_protegidas(tipo_area=area_protegida)
                        gdf_areas = gpd.sjoin(
                            gdf_areas_br, limite, predicate='intersects'
                        ).drop(columns=['index_right'])

                        # Desenha TODAS as áreas protegidas que tocam a região desde
                        # já — mesmo que acabem com 0 km² queimados dentro, elas
                        # continuam aparecendo no mapa como contexto.
                        areas_afetadas = (
                            gdf_areas[['nome_area', 'geometry']]
                            .drop_duplicates(subset='nome_area')
                            .reset_index(drop=True)
                        )

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
                                # Recorta o raster só nas áreas com queima de fato,
                                # mas mantém 'areas_afetadas' com TODAS as áreas da
                                # região (definido acima) pro desenho do limite.
                                areas_com_queima = gdf_areas[
                                    gdf_areas['nome_area'].isin(
                                        df_ranking_areas['Área Protegida']
                                    )
                                ]
                                total_valor = round(df_ranking_areas['Valor'].sum(), 2)
                                ee_geom_afetadas = ee.Geometry(
                                    areas_com_queima.geometry.union_all().__geo_interface__
                                )
                                area_queimada_img = area_queimada_img.clip(ee_geom_afetadas)
                                img_area_km2 = (
                                    ee.Image.pixelArea().divide(1000000)
                                    .updateMask(area_queimada_img.gt(0))
                                )
                            else:
                                total_valor = 0

                    # ranking de municipios e serie temporal: calculados dentro das abas

            except Exception as e:
                st.warning(f"⚠️ Erro ao processar MODIS: {e}")

        status.update(label="✅ Análise concluída!", state="complete", expanded=False)

    # =============================================================
    # --- RENDERIZAÇÃO ---
    # =============================================================

    # Mostra o card + mapa completo quando: há registros (total_valor > 0) OU
    # quando há áreas protegidas na região mesmo sem foco/queima dentro delas
    # (nesse caso o mapa ainda desenha os limites, só sem card de contagem).
    tem_areas_sem_ocorrencia = (
        total_valor == 0 and area_protegida != "Nenhuma" and not areas_afetadas.empty
    )

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

    elif total_valor == 0 and not tem_areas_sem_ocorrencia:
        st.error("⚠️ Nenhum registro detectado nos limites selecionados.")

    else:
        if tem_areas_sem_ocorrencia:
            st.warning(
                f"🌳 Nenhum {'foco' if 'INPE' in fonte_escolhida else 'km² queimado'} "
                f"encontrado dentro de {area_protegida} nesta região/período — mas há "
                f"{areas_afetadas['nome_area'].nunique()} área(s) desse tipo na região "
                "e os limites delas estão desenhados no mapa abaixo."
            )
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

        # --- ALERTA DE ÁREAS PROTEGIDAS ---
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
                    color_continuous_scale=px.colors.sequential.Reds,
                    title=titulo_dinamico
                )
                nome_eixo_x = (
                    "Nº de Focos" if "INPE" in fonte_escolhida else "Área Afetada (km²)"
                )
                fig_areas.update_layout(
                    template='plotly_dark',
                    xaxis_title=nome_eixo_x,
                    yaxis={'categoryorder': 'total ascending'},
                    height=350, margin=dict(t=40, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_areas, use_container_width=True)
            with col_alerta2:
                st.markdown("**Lista Completa de Áreas Afetadas**")
                st.dataframe(
                    df_ranking_areas, hide_index=True,
                    height=350, use_container_width=True
                )

        st.markdown("---")

        # Inicialização de variáveis NBR
        stats_sev = {}
        dnbr_img = None

        # =============================================================
        # --- ABAS PRINCIPAIS ---
        # =============================================================
        aba_mapa, aba_graficos, aba_nbr, aba_impacto, aba_export, aba_risco = st.tabs([
            "🗺️ Mapa de Focos",
            "📈 Gráficos & Anomalia",
            "🔬 Severidade (NBR Sentinel-2)",
            "💰 Impacto Econômico",
            "⬇️ Exportar Dados",
            "🎯 Risco Preditivo (ML)"
        ])

        # ----------------------------------------------------------
        # ABA 1 — MAPA
        # ----------------------------------------------------------
        with aba_mapa:
            col_controles1, col_controles2 = st.columns([1, 1.2])
            with col_controles1:
                estilo_mapa = st.radio(
                    "🎨 Estilo de Fundo:",
                    ["🌑 Mapa Dark", "🛰️ Satélite (Google)", "🗺️ Mapa Padrão"],
                    horizontal=False
                )
            focar_area = "Visão Geral"
            with col_controles2:
                if not areas_afetadas.empty:
                    focar_area = st.selectbox(
                        "🔍 Zoom direto para:",
                        ["Visão Geral"] + df_ranking_areas['Área Protegida'].tolist()
                    )

            if focar_area != "Visão Geral" and not areas_afetadas.empty:
                area_especifica = areas_afetadas[
                    areas_afetadas['nome_area'] == focar_area
                ]
                centro = area_especifica.geometry.union_all().centroid
                bounds = area_especifica.geometry.total_bounds
                zoom_inicio = 11
            else:
                centro = limite.geometry.union_all().centroid
                bounds = limite.geometry.total_bounds
                zoom_inicio = 10 if tipo_analise == "Por Município" else 6

            tiles_config = {
                "🌑 Mapa Dark": {
                    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                    "attr": "Esri",
                    "ref_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                },
                "🛰️ Satélite (Google)": {
                    "url": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                    "attr": "Google",
                    "ref_url": None,
                },
                "🗺️ Mapa Padrão": {
                    "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "attr": "OpenStreetMap",
                    "ref_url": None,
                },
            }
            cfg = tiles_config[estilo_mapa]
            m = folium.Map(
                location=[centro.y, centro.x],
                zoom_start=zoom_inicio,
                tiles=cfg["url"],
                attr=cfg["attr"],
                zoom_control=True,
                prefer_canvas=True,
            )
            if cfg["ref_url"]:
                folium.TileLayer(
                    tiles=cfg["ref_url"], attr="Esri Ref",
                    overlay=True, control=False
                ).add_to(m)

            if focar_area != "Visão Geral":
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

            # Borda da região selecionada
            folium.GeoJson(
                limite.__geo_interface__,
                name="Região selecionada",
                style_function=lambda x: {
                    'fillColor': '#00d4ff',
                    'fillOpacity': 0.04,
                    'color': '#00d4ff',
                    'weight': 2.5,
                    'dashArray': '6 3',
                },
            ).add_to(m)

            # Áreas protegidas afetadas
            if not areas_afetadas.empty:
                folium.GeoJson(
                    areas_afetadas.__geo_interface__,
                    name="Áreas protegidas afetadas",
                    style_function=lambda x: {
                        'fillColor': '#e74c3c',
                        'fillOpacity': 0.18,
                        'color': '#c0392b',
                        'weight': 1.5,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['nome_area'],
                        aliases=['📍 Área Protegida:'],
                        style=(
                            "font-size:12px; background:white; color:#2c3e50; "
                            "border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.25);"
                        ),
                    ),
                ).add_to(m)

            # Dados de queimada / focos
            if "INPE" in fonte_escolhida and not df_rec.empty:
                HeatMap(
                    df_rec[["latitude", "longitude"]].dropna().values.tolist(),
                    name="Densidade de focos",
                    radius=8,
                    blur=12,
                    max_zoom=14,
                    min_opacity=0.35,
                    gradient={0.2: '#ffffb2', 0.45: '#fecc5c',
                               0.65: '#fd8d3c', 0.85: '#f03b20', 1.0: '#bd0026'},
                ).add_to(m)

                # Legenda INPE
                legenda_inpe = """
                <div style="position:fixed; bottom:28px; left:12px; z-index:9999;
                            background:rgba(15,15,15,0.82); padding:10px 14px;
                            border-radius:10px; font-size:12px; color:#ecf0f1;
                            line-height:1.9; border:1px solid rgba(255,255,255,0.1);">
                  <b style="font-size:13px; letter-spacing:.5px;">🔥 Intensidade de Focos</b><br>
                  <span style="background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026);
                               display:inline-block;width:130px;height:10px;border-radius:4px;
                               vertical-align:middle;margin-top:4px;"></span><br>
                  <span style="color:#ffffb2;">Baixa</span>
                  <span style="float:right;color:#bd0026;">Alta</span>
                </div>"""
                m.get_root().html.add_child(folium.Element(legenda_inpe))

            elif "MODIS" in fonte_escolhida and area_queimada_img:
                vis_params_quente = {
                    'min': 1, 'max': 366,
                    'palette': ['#fff7bc', '#fec44f', '#fe9929',
                                '#ec7014', '#cc4c02', '#8c2d04'],
                }
                m.add_ee_layer(
                    area_queimada_img.updateMask(area_queimada_img.gt(0)),
                    vis_params_quente, 'Área Queimada (MODIS)', opacity=0.85
                )

                # Legenda MODIS
                legenda_modis = """
                <div style="position:fixed; bottom:28px; left:12px; z-index:9999;
                            background:rgba(15,15,15,0.82); padding:10px 14px;
                            border-radius:10px; font-size:12px; color:#ecf0f1;
                            line-height:1.9; border:1px solid rgba(255,255,255,0.1);">
                  <b style="font-size:13px; letter-spacing:.5px;">🗺️ Área Queimada (MODIS)</b><br>
                  <span style="background:linear-gradient(to right,#fff7bc,#fec44f,#fe9929,#ec7014,#cc4c02,#8c2d04);
                               display:inline-block;width:130px;height:10px;border-radius:4px;
                               vertical-align:middle;margin-top:4px;"></span><br>
                  <span style="color:#fff7bc;">Início do mês</span>
                  <span style="float:right;color:#8c2d04;">Fim do mês</span>
                </div>"""
                m.get_root().html.add_child(folium.Element(legenda_modis))

            folium.LayerControl(collapsed=False).add_to(m)

            # Key dinâmico: muda junto com os dados, forçando re-render automático
            # sem precisar trocar o estilo do mapa manualmente
            if "INPE" in fonte_escolhida:
                _periodo = f"{dt_ini.strftime('%Y%m%d')}_{hoje.strftime('%Y%m%d')}_{'_'.join(sorted(satelites_sel))}"
            else:
                _periodo = f"{ano_modis}_{mes_modis}"
            _map_key = f"mapa_{val_sel}_{_periodo}_{area_protegida}_{estilo_mapa}_{focar_area}"

            st_folium(m, width=None, height=700, returned_objects=[], key=_map_key)

        # ----------------------------------------------------------
        # ABA 2 — GRÁFICOS & ANOMALIA
        # ----------------------------------------------------------
        with aba_graficos:
            if "INPE" in fonte_escolhida and df_rec.empty:
                st.info(
                    "🌳 Nenhum foco dentro da área protegida selecionada nesta região/período "
                    "— sem dados pra série temporal ou ranking de municípios."
                )
            elif "INPE" in fonte_escolhida:
                st.subheader("📈 Evolução Temporal dos Focos")
                data_col = next(c for c in df_rec.columns if 'data' in c)
                df_rec[data_col] = pd.to_datetime(df_rec[data_col])
                freq = 'D' if (hoje - dt_ini).days <= 90 else 'MS'
                df_g = (
                    df_rec.set_index(data_col)
                    .resample(freq).size()
                    .reset_index(name='focos')
                )
                fig_line = px.line(df_g, x=data_col, y='focos', markers=True, height=350)
                fig_line.update_traces(line_color='#e64a19', line_width=3)
                fig_line.update_layout(
                    template='plotly_dark',
                    xaxis_title="Tempo", yaxis_title="Nº de Focos",
                    margin=dict(t=20, b=20)
                )
                st.plotly_chart(fig_line, use_container_width=True)

                if 'municipio' in df_rec.columns and tipo_analise != "Por Município":
                    df_top_mun = df_rec['municipio'].value_counts().reset_index()
                    qtd_mun = min(5, len(df_top_mun))
                    st.subheader(f"🏆 Top {qtd_mun} Municípios Afetados")
                    df_top_mun = df_top_mun.head(5)
                    df_top_mun.columns = ['Município', 'Focos']
                    fig_bar = px.bar(
                        df_top_mun, x='Focos', y='Município', orientation='h',
                        text='Focos', color='Focos',
                        color_continuous_scale=px.colors.sequential.Reds
                    )
                    fig_bar.update_layout(
                        template='plotly_dark',
                        yaxis={'categoryorder': 'total ascending'},
                        height=320, margin=dict(t=20, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

            else:
                # --- Calcula serie temporal e ranking de municipios aqui (sob demanda) ---
                if df_modis_temporal.empty and total_valor > 0:
                    with st.spinner("📊 Calculando evolução mensal..."):
                        try:
                            geom_temporal = (
                                ee_geom_afetadas
                                if (area_protegida != "Nenhuma" and not areas_afetadas.empty)
                                else ee_geom_complex
                            )
                            def calc_mes(m):
                                m_num = ee.Number(m)
                                ini = ee.Date.fromYMD(ano_modis, m_num, 1)
                                img_mes = (
                                    ee.ImageCollection('MODIS/061/MCD64A1')
                                    .filterDate(ini, ini.advance(1, 'month'))
                                    .select('BurnDate').max().clip(geom_temporal)
                                )
                                val = (ee.Image.pixelArea().divide(1000000)
                                    .updateMask(img_mes.gt(0))
                                    .reduceRegion(reducer=ee.Reducer.sum(),
                                        geometry=geom_temporal, scale=1000,
                                        maxPixels=1e13, tileScale=4, bestEffort=True
                                    ).get('area'))
                                return ee.Feature(None, {'mes': m_num, 'area': val})
                            fc_meses = ee.FeatureCollection(ee.List.sequence(1,12).map(calc_mes)).getInfo()
                            meses_map_label = {1:'Jan',2:'Fev',3:'Mar',4:'Abr',5:'Mai',6:'Jun',
                                               7:'Jul',8:'Ago',9:'Set',10:'Out',11:'Nov',12:'Dez'}
                            df_modis_temporal = pd.DataFrame([
                                {'Mês': f['properties']['mes'],
                                 'Área (km²)': round(f['properties'].get('area') or 0, 2)}
                                for f in fc_meses['features']
                            ])
                            df_modis_temporal['Mês Nome'] = df_modis_temporal['Mês'].map(meses_map_label)
                        except Exception as _e_temp:
                            st.warning(f"⚠️ Não foi possível calcular a série temporal: {_e_temp}")

                if df_top_mun_modis.empty and tipo_analise != "Por Município" and total_valor > 0:
                    with st.spinner("🏙️ Calculando ranking de municípios..."):
                        try:
                            muns_ee = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(ee_geom_complex)
                            img_area_km2_mun = (ee.Image.pixelArea().divide(1000000)
                                .updateMask(area_queimada_img.gt(0)))
                            stats_mun = img_area_km2_mun.reduceRegions(
                                collection=muns_ee, reducer=ee.Reducer.sum(), scale=1000
                            ).getInfo()
                            recs_mun = [
                                {'Município': f['properties']['ADM2_NAME'],
                                 'Valor': round(f['properties'].get('sum', 0), 2)}
                                for f in stats_mun['features']
                                if f['properties'].get('sum', 0) > 0
                            ]
                            if recs_mun:
                                df_top_mun_modis = (pd.DataFrame(recs_mun)
                                    .sort_values(by='Valor', ascending=False).head(5))
                        except Exception as _e_mun:
                            st.warning(f"⚠️ Não foi possível calcular ranking de municípios: {_e_mun}")

                # Série temporal MODIS
                if not df_modis_temporal.empty:
                    st.subheader(f"📈 Evolução Mensal — {ano_modis}")
                    fig_line = px.line(
                        df_modis_temporal, x='Mês Nome', y='Área (km²)',
                        markers=True, height=350
                    )
                    fig_line.update_traces(line_color='#e64a19', line_width=3)
                    fig_line.update_layout(
                        template='plotly_dark',
                        xaxis_title="Mês", yaxis_title="Área Afetada (km²)",
                        margin=dict(t=20, b=20)
                    )
                    st.plotly_chart(fig_line, use_container_width=True)

                if tipo_analise != "Por Município" and not df_top_mun_modis.empty:
                    st.subheader("🏆 Top 5 Municípios Afetados")
                    fig_bar = px.bar(
                        df_top_mun_modis, x='Valor', y='Município', orientation='h',
                        text='Valor', color='Valor',
                        color_continuous_scale=px.colors.sequential.Reds
                    )
                    fig_bar.update_layout(
                        template='plotly_dark',
                        xaxis_title="Área Afetada (km²)",
                        yaxis={'categoryorder': 'total ascending'},
                        height=350, margin=dict(t=20, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # --- ANOMALIA HISTÓRICA ---
                if not df_modis_temporal.empty and ano_modis > 2001:
                    st.markdown("---")
                    st.subheader(f"📊 {ano_modis} foi um ano normal para queimadas?")
                    st.caption(
                        f"Comparamos cada mês de **{ano_modis}** com a média histórica "
                        f"do mesmo mês entre **2001 e {ano_modis - 1}**. "
                        "Assim você vê rapidamente se o ano foi mais ou menos crítico que o habitual."
                    )

                    with st.spinner("Calculando comparação histórica..."):
                        df_anomalia = calcular_anomalia_modis(geom_json_str, ano_modis)

                    if not df_anomalia.empty:
                        col_ano = f'Área {ano_modis} (km²)'

                        # --- Resumo geral em linguagem simples ---
                        meses_acima = df_anomalia[df_anomalia['Anomalia (%)'] > 20]
                        meses_abaixo = df_anomalia[df_anomalia['Anomalia (%)'] < -20]
                        mes_pior = df_anomalia.loc[df_anomalia['Anomalia (%)'].idxmax()]
                        mes_melhor = df_anomalia.loc[df_anomalia['Anomalia (%)'].idxmin()]

                        # Card de veredicto geral
                        if len(meses_acima) >= 6:
                            veredicto_cor = "#c0392b"
                            veredicto_icon = "🔴"
                            veredicto_texto = f"{ano_modis} foi um ano <b>acima do normal</b> em queimadas"
                            veredicto_detalhe = f"{len(meses_acima)} de 12 meses ficaram acima da média histórica."
                        elif len(meses_abaixo) >= 6:
                            veredicto_cor = "#27ae60"
                            veredicto_icon = "🟢"
                            veredicto_texto = f"{ano_modis} foi um ano <b>abaixo do normal</b> em queimadas"
                            veredicto_detalhe = f"{len(meses_abaixo)} de 12 meses ficaram abaixo da média histórica."
                        else:
                            veredicto_cor = "#e67e22"
                            veredicto_icon = "🟡"
                            veredicto_texto = f"{ano_modis} foi um ano <b>dentro do padrão histórico</b>"
                            veredicto_detalhe = "Os meses se distribuíram equilibradamente acima e abaixo da média."

                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.04); border-left:6px solid {veredicto_cor};
                                    border-radius:8px; padding:14px 18px; margin-bottom:16px;">
                            <span style="font-size:20px;">{veredicto_icon}</span>
                            <span style="font-size:17px; font-weight:600;"> {veredicto_texto}</span><br>
                            <span style="color:#aaa; font-size:13px;">{veredicto_detalhe}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        # Cards de destaques (mês mais crítico e mais tranquilo)
                        col_dest1, col_dest2 = st.columns(2)
                        with col_dest1:
                            pct = mes_pior['Anomalia (%)']
                            sinal = "+" if pct > 0 else ""
                            st.markdown(f"""
                            <div style="background:rgba(192,57,43,0.12); border:1px solid #c0392b;
                                        border-radius:8px; padding:12px 16px; text-align:center;">
                                <div style="font-size:12px; color:#e74c3c; text-transform:uppercase;
                                            letter-spacing:1px; margin-bottom:4px;">📛 Mês mais crítico</div>
                                <div style="font-size:28px; font-weight:700; color:#e74c3c;">
                                    {mes_pior['Mês Nome']}
                                </div>
                                <div style="font-size:20px; font-weight:600; color:#e74c3c;">
                                    {sinal}{pct:.0f}% vs. média
                                </div>
                                <div style="font-size:12px; color:#aaa; margin-top:4px;">
                                    {mes_pior[col_ano]:,.1f} km² queimados<br>
                                    Média histórica: {mes_pior['Média Histórica (km²)']:,.1f} km²
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col_dest2:
                            pct2 = mes_melhor['Anomalia (%)']
                            sinal2 = "+" if pct2 > 0 else ""
                            st.markdown(f"""
                            <div style="background:rgba(39,174,96,0.12); border:1px solid #27ae60;
                                        border-radius:8px; padding:12px 16px; text-align:center;">
                                <div style="font-size:12px; color:#2ecc71; text-transform:uppercase;
                                            letter-spacing:1px; margin-bottom:4px;">✅ Mês mais tranquilo</div>
                                <div style="font-size:28px; font-weight:700; color:#2ecc71;">
                                    {mes_melhor['Mês Nome']}
                                </div>
                                <div style="font-size:20px; font-weight:600; color:#2ecc71;">
                                    {sinal2}{pct2:.0f}% vs. média
                                </div>
                                <div style="font-size:12px; color:#aaa; margin-top:4px;">
                                    {mes_melhor[col_ano]:,.1f} km² queimados<br>
                                    Média histórica: {mes_melhor['Média Histórica (km²)']:,.1f} km²
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

                        # --- Gráfico de barras de anomalia (mais intuitivo que linhas) ---
                        cores_barras = [
                            '#c0392b' if v > 50 else
                            '#e67e22' if v > 20 else
                            '#27ae60' if v < -20 else
                            '#636e72'
                            for v in df_anomalia['Anomalia (%)']
                        ]

                        fig_anom = go.Figure()
                        fig_anom.add_bar(
                            x=df_anomalia['Mês Nome'],
                            y=df_anomalia['Anomalia (%)'],
                            marker_color=cores_barras,
                            text=[
                                f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"
                                for v in df_anomalia['Anomalia (%)']
                            ],
                            textposition='outside',
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Desvio da média: %{y:.1f}%<br>"
                                "<extra></extra>"
                            ),
                        )
                        fig_anom.add_hline(
                            y=0, line_color='rgba(255,255,255,0.4)',
                            line_width=1.5, line_dash='dot'
                        )
                        fig_anom.add_hrect(
                            y0=-20, y1=20,
                            fillcolor='rgba(255,255,255,0.03)',
                            line_width=0,
                            annotation_text="Faixa normal (±20%)",
                            annotation_position="top right",
                            annotation_font_size=11,
                            annotation_font_color='rgba(255,255,255,0.4)',
                        )
                        fig_anom.update_layout(
                            template='plotly_dark',
                            title=dict(
                                text=f"Desvio de {ano_modis} em relação à média histórica (%) — por mês",
                                font_size=14
                            ),
                            xaxis_title="",
                            yaxis_title="Desvio da média histórica (%)",
                            height=380,
                            margin=dict(t=50, b=20),
                            showlegend=False,
                            yaxis=dict(zeroline=False),
                        )
                        st.plotly_chart(fig_anom, use_container_width=True)

                        # Legenda de cores simples
                        st.markdown("""
                        <div style="display:flex; gap:16px; font-size:12px;
                                    color:#aaa; margin-top:-8px; margin-bottom:8px;
                                    flex-wrap:wrap;">
                            <span><span style="color:#c0392b;">■</span> Muito acima (+50%)</span>
                            <span><span style="color:#e67e22;">■</span> Acima (+20% a +50%)</span>
                            <span><span style="color:#636e72;">■</span> Dentro do normal (±20%)</span>
                            <span><span style="color:#27ae60;">■</span> Abaixo (menos de −20%)</span>
                        </div>
                        """, unsafe_allow_html=True)

                elif ano_modis == 2001:
                    st.info(
                        "💡 A comparação histórica requer pelo menos 2 anos de dados. "
                        "Selecione um ano a partir de 2002."
                    )

        # ----------------------------------------------------------
        # ABA 3 — NBR SENTINEL-2
        # ----------------------------------------------------------
        with aba_nbr:
            if "INPE" in fonte_escolhida:
                st.info(
                    "💡 A análise de severidade NBR usa imagens Sentinel-2 e avalia "
                    "a cicatriz da queimada pixel a pixel. "
                    "Para ativá-la, selecione **🗺️ Área Queimada (NASA MODIS)** "
                    "na barra lateral e escolha o mês do evento."
                )
            else:
                st.subheader("🔬 Análise de Severidade da Queimada — dNBR (Sentinel-2)")
                st.markdown(
                    "O índice **dNBR** (delta Normalized Burn Ratio) compara a reflectância "
                    "da vegetação **antes e depois** do fogo usando infravermelho próximo (B8) "
                    "e SWIR (B12). Valores altos indicam maior destruição da cobertura vegetal. "
                    "Classificação seguindo o padrão **USGS**."
                )

                with st.expander("📖 Como funciona o dNBR? Clique para entender as classes de severidade", expanded=False):
                    st.markdown("""
                    <style>
                    .dnbr-ruler { display:flex; width:100%; height:38px; border-radius:6px; overflow:hidden; margin-bottom:4px; }
                    .dnbr-ruler div { display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:600; color:#fff; }
                    .dnbr-ticks { display:flex; justify-content:space-between; font-size:11px; color:#888; margin-bottom:20px; padding:0 2px; }
                    .dnbr-cards { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:20px; }
                    .dnbr-card { display:flex; border:1px solid rgba(255,255,255,0.08); border-radius:8px; overflow:hidden; background:rgba(255,255,255,0.03); }
                    .dnbr-stripe { width:8px; flex-shrink:0; }
                    .dnbr-body { padding:9px 11px; }
                    .dnbr-name { font-size:13px; font-weight:700; margin:0 0 2px; }
                    .dnbr-range { font-size:11px; color:#999; margin:0 0 4px; font-family:monospace; }
                    .dnbr-desc { font-size:12px; color:#bbb; margin:0; line-height:1.4; }
                    .dnbr-flow { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:12px; }
                    .dnbr-box { border-radius:8px; padding:10px 14px; font-size:12px; line-height:1.6; flex:1; min-width:140px; }
                    .dnbr-arrow { font-size:20px; color:#888; flex-shrink:0; }
                    .dnbr-formula { background:rgba(255,255,255,0.05); border-radius:8px; padding:10px 14px;
                                    font-size:12px; color:#ccc; line-height:1.9; margin-bottom:0; }
                    .dnbr-formula code { background:rgba(255,255,255,0.1); border-radius:4px; padding:1px 6px; font-family:monospace; }
                    </style>

                    <p style="font-size:13px; color:#aaa; margin-bottom:10px;">
                        O <b style="color:#eee;">dNBR</b> compara imagens Sentinel-2 antes e depois do fogo usando as bandas
                        B8 (infravermelho próximo) e B12 (SWIR). Vegetação sã reflete muito em B8 e pouco em B12 —
                        o inverso ocorre em área queimada.
                    </p>

                    <div class="dnbr-flow">
                        <div class="dnbr-box" style="background:rgba(22,101,85,0.35); border:1px solid rgba(22,160,133,0.3);">
                            <b style="color:#1abc9c;">Imagem pré-fogo</b><br>
                            Sentinel-2 · 60–90 dias antes<br>
                            <span style="color:#888; font-size:11px;">NBR_pré = (B8−B12)÷(B8+B12)</span><br>
                            <span style="color:#999; font-size:11px;">valor típico: +0.4 a +0.8</span>
                        </div>
                        <div class="dnbr-arrow">⟶</div>
                        <div class="dnbr-box" style="background:rgba(120,60,20,0.35); border:1px solid rgba(180,80,20,0.3);">
                            <b style="color:#e67e22;">Imagem pós-fogo</b><br>
                            Sentinel-2 · mês do evento<br>
                            <span style="color:#888; font-size:11px;">NBR_pós = (B8−B12)÷(B8+B12)</span><br>
                            <span style="color:#999; font-size:11px;">valor típico: −0.1 a +0.2</span>
                        </div>
                        <div class="dnbr-arrow">⟶</div>
                        <div class="dnbr-box" style="background:rgba(150,100,0,0.35); border:1px solid rgba(200,150,0,0.3); text-align:center;">
                            <b style="color:#f1c40f; font-size:14px;">dNBR × 1000</b><br>
                            <span style="color:#ddd; font-size:12px;">NBR_pré − NBR_pós</span><br>
                            <span style="color:#aaa; font-size:11px;">= severidade da queima</span>
                        </div>
                    </div>

                    <div style="margin:16px 0 8px; font-size:12px; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
                        Régua de severidade (padrão USGS)
                    </div>
                    <div class="dnbr-ruler">
                        <div style="width:13%; background:#1a9850;">Reg.</div>
                        <div style="width:22%; background:#91cf60; color:#2d5010;">Não afetado</div>
                        <div style="width:18%; background:#fee08b; color:#7a5800;">Baixa</div>
                        <div style="width:18%; background:#fc8d59; color:#5c1a00;">Moderada</div>
                        <div style="width:15%; background:#d73027;">Mod-Alta</div>
                        <div style="width:14%; background:#8c0505;">Alta</div>
                    </div>
                    <div class="dnbr-ticks">
                        <span>≪ 0</span><span>−100</span><span>+100</span>
                        <span>+270</span><span>+440</span><span>+660</span><span>≫ 1000</span>
                    </div>

                    <div class="dnbr-cards">
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#1a9850;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#1a9850;">🌱 Regeneração</p>
                                <p class="dnbr-range">dNBR &lt; −100</p>
                                <p class="dnbr-desc">Vegetação cresceu após incêndio anterior (broto)</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#91cf60;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#6a9e30;">🌿 Não afetado</p>
                                <p class="dnbr-range">−100 a +100</p>
                                <p class="dnbr-desc">Vegetação intacta ou variação sazonal normal</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#fee08b;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#c9a000;">🟡 Baixa severidade</p>
                                <p class="dnbr-range">+100 a +270</p>
                                <p class="dnbr-desc">Queima superficial; dossel parcialmente afetado</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#fc8d59;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#e05010;">🟠 Moderada</p>
                                <p class="dnbr-range">+270 a +440</p>
                                <p class="dnbr-desc">Danos significativos; dossel destruído em parte</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#d73027;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#d73027;">🔴 Moderada-Alta</p>
                                <p class="dnbr-range">+440 a +660</p>
                                <p class="dnbr-desc">Destruição extensa do dossel; solo exposto</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#8c0505;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#c0392b;">⬛ Alta severidade</p>
                                <p class="dnbr-range">dNBR &gt; +660</p>
                                <p class="dnbr-desc">Destruição total; solo nu, cinzas, carvão</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # --- CHAVE DE CACHE PARA O BOTÃO ---
                _nbr_cache_key = f"nbr_resultado_{val_sel}_{ano_modis}_{mes_modis}"
                if _nbr_cache_key not in st.session_state:
                    st.session_state[_nbr_cache_key] = None

                # --- BOTÃO SOB DEMANDA ---
                if st.session_state[_nbr_cache_key] is None:
                    if area_km2_local > 500_000:
                        st.warning(
                            f"⚠️ A área selecionada é muito grande (~{area_km2_local:,.0f} km²). "
                            "O cálculo NBR pode levar **3–8 minutos** ou falhar por limite do Google Earth Engine. "
                            "Para melhores resultados, prefira análises **Por Município** ou estados menores."
                        )
                    st.info(
                        "⚡ A análise de severidade usa imagens Sentinel-2 e pode levar "
                        "**1–3 minutos** dependendo do tamanho da área. "
                        "Clique quando estiver pronto."
                    )
                    if st.button(
                        "🛰️ Calcular Severidade NBR (Sentinel-2)",
                        type="primary",
                        use_container_width=True,
                        key=f"btn_nbr_{val_sel}_{ano_modis}_{mes_modis}"
                    ):
                        with st.spinner("🛰️ Buscando imagens Sentinel-2 e calculando dNBR… aguarde."):
                            try:
                                # Constrói o dNBR UMA VEZ e reutiliza para stats + mapa
                                # evitando duas pipelines separadas no GEE
                                poly_nbr, dnbr_img = _construir_dnbr(
                                    geom_json_str, ano_modis, mes_modis,
                                    area_queimada_img,
                                    area_km2_hint=area_km2_local
                                )
                                # Calcula as estatísticas a partir do dNBR já construído
                                stats_sev = calcular_stats_nbr(
                                    geom_json_str, ano_modis, mes_modis,
                                    area_queimada_img,
                                    area_km2_hint=area_km2_local
                                )
                                st.session_state[_nbr_cache_key] = {
                                    "stats_sev": stats_sev,
                                    "dnbr_img": dnbr_img,
                                    "ok": True
                                }
                                st.success("✅ Severidade calculada com sucesso! Role a tela para ver os resultados.")
                                st.rerun()
                            except ValueError as ve:
                                st.warning(f"⚠️ {ve}")
                            except Exception as e:
                                st.error(
                                    f"⚠️ Erro ao processar Sentinel-2: {e}\n\n"
                                    "Tente uma região menor ou um mês diferente."
                                )

                # --- RECUPERA RESULTADO DO CACHE DE SESSÃO ---
                nbr_ok = False
                stats_sev = {}
                dnbr_img = None
                if st.session_state[_nbr_cache_key] and st.session_state[_nbr_cache_key].get("ok"):
                    stats_sev = st.session_state[_nbr_cache_key]["stats_sev"]
                    dnbr_img  = st.session_state[_nbr_cache_key]["dnbr_img"]
                    nbr_ok = True

                col_nbr1, col_nbr2 = st.columns([1.4, 1])

                with col_nbr1:

                    if nbr_ok and dnbr_img is not None:
                        try:
                            centro_nbr = limite.geometry.unary_union.centroid
                            m_nbr = folium.Map(
                                location=[centro_nbr.y, centro_nbr.x],
                                zoom_start=8,
                                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                                attr='Google'
                            )
                            
                            folium.GeoJson(
                                limite.__geo_interface__,
                                style_function=lambda x: {
                                    'fillColor': 'transparent',
                                    'color': '#00d4ff', 'weight': 2
                                }
                            ).add_to(m_nbr)
                            
                            vis_dnbr = {
                                'min': -100, 'max': 1000,
                                'palette': [
                                    '#1a9850', '#91cf60', '#d9ef8b', '#ffffbf',
                                    '#fee08b', '#fc8d59', '#d73027', '#7a0403'
                                ]
                            }
                            
                            m_nbr.add_ee_layer(
                                dnbr_img, vis_dnbr,
                                'dNBR (severidade contínua)', opacity=0.85
                            )
                            
                            legenda_html = """
                            <div style="position:fixed; bottom:28px; right:10px; z-index:9999;
                                        background:rgba(20,20,20,0.88); padding:12px 16px;
                                        border-radius:10px; font-size:12px; color:white; line-height:2;
                                        border:1px solid rgba(255,255,255,0.1);">
                                <b style="font-size:13px;">Severidade dNBR</b><br>
                                <span style="color:#1a9850;">■</span> Regeneração (dNBR &lt; -0.1)<br>
                                <span style="color:#91cf60;">■</span> Não afetado (-0.1 a 0.1)<br>
                                <span style="color:#fee08b;">■</span> Baixa (0.1 a 0.27)<br>
                                <span style="color:#fc8d59;">■</span> Moderada (0.27 a 0.44)<br>
                                <span style="color:#d73027;">■</span> Moderada-Alta (0.44 a 0.66)<br>
                                <span style="color:#7a0403;">■</span> Alta (&gt; 0.66)
                            </div>"""
                            m_nbr.get_root().html.add_child(folium.Element(legenda_html))
                            
                            folium.LayerControl().add_to(m_nbr)
                            
                            _nbr_key = f"nbr_{val_sel}_{ano_modis}_{mes_modis}"
                            st_folium(m_nbr, width=None, height=620, returned_objects=[], key=_nbr_key)

                            st.markdown("")
                            if st.button(
                                "🔄 Recalcular Severidade",
                                key=f"btn_nbr_reset_{val_sel}_{ano_modis}_{mes_modis}",
                                use_container_width=True
                            ):
                                st.session_state[_nbr_cache_key] = None
                                st.rerun()
                            
                        except Exception as erro_mapa:
                            st.error(f"⚠️ Os dados foram calculados, mas ocorreu um erro ao desenhar o mapa Folium: {erro_mapa}")

                with col_nbr2:
                    if stats_sev:       
                        st.markdown("---")
                        st.markdown("**Distribuição de Severidade NBR (Sentinel-2):**")
                        df_sev_exp = pd.DataFrame(
                            list(stats_sev.items()), columns=['Classe', 'Área (km²)']
                        )

                        cores_sev = {
                            'Regeneração':   '#1a9850',
                            'Não afetado':   '#91cf60',
                            'Baixa':         '#fee08b',
                            'Moderada':      '#fc8d59',
                            'Moderada-Alta': '#d73027',
                            'Alta':          '#7a0403'
                        }

                        # Gráfico de pizza
                        fig_pizza = px.pie(
                            df_sev_exp, values='Área (km²)', names='Classe',
                            color='Classe', color_discrete_map=cores_sev, hole=0.45
                        )
                        fig_pizza.update_layout(
                            template='plotly_dark',
                            height=280, margin=dict(t=10, b=10)
                        )
                        st.plotly_chart(fig_pizza, use_container_width=True)

                        # Gráfico de barras
                        fig_bar_sev = px.bar(
                            df_sev_exp, x='Área (km²)', y='Classe', orientation='h',
                            color='Classe', color_discrete_map=cores_sev,
                            text='Área (km²)'
                        )
                        fig_bar_sev.update_layout(
                            template='plotly_dark', showlegend=False,
                            yaxis={'categoryorder': 'total ascending'},
                            height=260, margin=dict(t=10, b=10)
                        )
                        st.plotly_chart(fig_bar_sev, use_container_width=True)

                        # Métricas de destaque
                        area_alta = (
                            stats_sev.get('Alta', 0)
                            + stats_sev.get('Moderada-Alta', 0)
                        )
                        area_total_afetada = sum(
                            v for k, v in stats_sev.items()
                            if k not in ['Não afetado', 'Regeneração']
                        )

                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            st.metric("🔴 Alta Severidade", f"{area_alta:.2f} km²")
                        with col_m2:
                            st.metric("🔥 Total Afetado", f"{area_total_afetada:.2f} km²")

                        st.markdown("---")
                        st.markdown(
                            "**Interpretação:**\n\n"
                            "- **Baixa:** vegetação parcialmente afetada, recuperação rápida\n"
                            "- **Moderada:** danos significativos ao dossel\n"
                            "- **Alta:** destruição quase total da cobertura vegetal"
                        )



        # ----------------------------------------------------------
        # ----------------------------------------------------------
        # ABA 4 — IMPACTO ECONÔMICO
        # ----------------------------------------------------------
        with aba_impacto:
          try:
            # --- TABELAS DE REFERÊNCIA ---
            VALORES_ECOSSIS = {
                "Amazônia":      {"conservador": 2000,  "moderado": 4000,  "otimista": 6000},
                "Cerrado":       {"conservador": 800,   "moderado": 1650,  "otimista": 2500},
                "Mata Atlântica":{"conservador": 3000,  "moderado": 5500,  "otimista": 8000},
                "Pantanal":      {"conservador": 1500,  "moderado": 2750,  "otimista": 4000},
                "Caatinga":      {"conservador": 400,   "moderado": 800,   "otimista": 1200},
                "Pampa":         {"conservador": 600,   "moderado": 1050,  "otimista": 1500},
            }
            EMISSOES_CO2_HA = {
                "Amazônia": 150, "Cerrado": 60, "Mata Atlântica": 120,
                "Pantanal": 80, "Caatinga": 30, "Pampa": 25,
            }
            PRECO_CARBONO_USD = 15
            CAMBIO_FIXO = buscar_cotacao_dolar()

            # --- DETECTAR BIOMA ---
            ESTADO_BIOMA = {
                "AM": "Amazônia", "PA": "Amazônia", "AC": "Amazônia",
                "RO": "Amazônia", "RR": "Amazônia", "AP": "Amazônia",
                "MT": "Cerrado",  "GO": "Cerrado",  "TO": "Cerrado",
                "MA": "Cerrado",  "PI": "Caatinga", "BA": "Caatinga",
                "CE": "Caatinga", "RN": "Caatinga", "PB": "Caatinga",
                "PE": "Caatinga", "AL": "Caatinga", "SE": "Caatinga",
                "MS": "Pantanal", "PR": "Mata Atlântica",
                "SC": "Mata Atlântica", "RS": "Pampa",
                "SP": "Mata Atlântica", "RJ": "Mata Atlântica",
                "ES": "Mata Atlântica", "MG": "Mata Atlântica",
                "DF": "Cerrado",
            }
            bioma_detectado = (
                bioma_dd if tipo_analise == "Por Bioma"
                else ESTADO_BIOMA.get(estado_dd, "Cerrado")
            )

            # --- ÁREA QUEIMADA ---
            # IMPORTANTE: só é possível estimar ÁREA (e, portanto, prejuízo em R$/ha) com
            # dados de ÁREA QUEIMADA (MODIS). Focos de calor (INPE) são detecções pontuais —
            # não representam uma área queimada, então não entram nesse cálculo.
            area_km2_calc = 0.0
            fonte_area = "Sem dados de área disponíveis"
            if "MODIS" in fonte_escolhida and total_valor > 0:
                area_km2_calc = float(total_valor)
                fonte_area = f"Satélite MODIS — {area_km2_calc:,.1f} km² queimados detectados"

            area_ha = area_km2_calc * 100

            st.markdown("## 💰 Impacto Econômico das Queimadas")
            st.markdown(
                f"Estimativa do prejuízo causado pelas queimadas em **{val_sel}**, "
                f"com base na área afetada e no valor dos serviços que o ecossistema presta à sociedade."
            )

            if "INPE" in fonte_escolhida:
                st.warning(
                    "⚠️ **Esta aba precisa de dados de Área Queimada (MODIS).** Você selecionou "
                    "'Focos de Calor (INPE)' na barra lateral — focos são **detecções pontuais** "
                    "(um pixel de calor), não medem a área efetivamente queimada, então não é "
                    "correto estimar hectares ou prejuízo em R$ a partir da contagem de focos. "
                    "Troque a fonte de dados para **'🗺️ Área Queimada (NASA MODIS)'** na barra "
                    "lateral e clique em 'Gerar Dashboard' novamente para ver o impacto econômico."
                )
            elif area_km2_calc == 0:
                st.warning(
                    "⚠️ Nenhuma área queimada detectada para o período selecionado. "
                    "Selecione outro mês ou ano."
                )
            else:
                # Leitura dos parâmetros do session_state (definidos no expander abaixo)
                bioma_calc = st.session_state.get("bioma_impacto",
                    bioma_detectado if bioma_detectado in VALORES_ECOSSIS
                    else list(VALORES_ECOSSIS.keys())[0])
                cenario    = st.session_state.get("cenario_impacto", "moderado")
                cambio_usd = st.session_state.get("cambio_impacto", CAMBIO_FIXO)

                # --- CÁLCULOS ---
                valor_usd_ha        = VALORES_ECOSSIS[bioma_calc][cenario]
                valor_brl_ha        = valor_usd_ha * cambio_usd
                fator_co2_ha        = EMISSOES_CO2_HA.get(bioma_calc, 60)
                perda_ecossis_brl   = area_ha * valor_brl_ha
                co2_emitido_t       = area_ha * fator_co2_ha
                credito_carbono_brl = co2_emitido_t * PRECO_CARBONO_USD * cambio_usd
                total_impacto_brl   = perda_ecossis_brl + credito_carbono_brl

                # --- CARD PRINCIPAL ---
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #c0392b22, #e74c3c11);
                            border-left: 6px solid #e74c3c; border-radius: 10px;
                            padding: 22px 26px; margin: 16px 0;">
                    <p style="margin: 0 0 4px 0; color: #aaa; font-size: 13px;
                              text-transform: uppercase; letter-spacing: 1px;">
                        Prejuízo econômico estimado — {bioma_calc} · Cenário {cenario.capitalize()}
                    </p>
                    <p style="font-size: 42px; font-weight: 800; margin: 0; color: #e74c3c;">
                        R$ {total_impacto_brl/1e9:,.2f} bilhões
                    </p>
                    <p style="margin: 6px 0 0 0; color: #bbb; font-size: 13px;">
                        📡 Base: {fonte_area}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # --- 3 CARDS SECUNDÁRIOS ---
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("🌍 Área Queimada",
                              f"{area_km2_calc:,.1f} km²",
                              f"{area_ha:,.0f} hectares")
                with m2:
                    st.metric("🌳 Perda de Serviços Ambientais",
                              f"R$ {perda_ecossis_brl/1e9:,.2f} bi",
                              f"R$ {valor_brl_ha:,.0f}/ha · {bioma_calc}")
                with m3:
                    st.metric("☁️ Carbono Emitido",
                              f"{co2_emitido_t/1e6:,.2f} Mt CO₂e",
                              f"≈ R$ {credito_carbono_brl/1e9:,.2f} bi em créditos perdidos")

                st.markdown("---")

                # --- GRÁFICO: COMPARAÇÃO DOS 3 CENÁRIOS ---
                st.markdown("### Como o valor muda conforme o cenário?")
                st.caption(
                    "Cada cenário reflete um intervalo da literatura científica — "
                    "do mais conservador (menor impacto estimado) ao otimista (maior)."
                )
                dados_cen = []
                for cen_k in ["conservador", "moderado", "otimista"]:
                    v_brl_ha_cen = VALORES_ECOSSIS[bioma_calc][cen_k] * cambio_usd
                    perda_cen    = area_ha * v_brl_ha_cen
                    co2_cen      = area_ha * fator_co2_ha * PRECO_CARBONO_USD * cambio_usd
                    total_cen    = perda_cen + co2_cen
                    dados_cen.append({
                        "Cenário": {"conservador": "Conservador", "moderado": "Moderado",
                                  "otimista": "Otimista"}[cen_k],
                        "Serviços Ambientais (R$ bi)": round(perda_cen / 1e9, 2),
                        "Créditos de Carbono (R$ bi)":  round(co2_cen / 1e9, 2),
                        "Total (R$ bi)":                   round(total_cen / 1e9, 2),
                    })
                df_cen = pd.DataFrame(dados_cen)
                fig_cen = go.Figure()
                fig_cen.add_bar(
                    name="Serviços Ambientais",
                    x=df_cen["Cenário"],
                    y=df_cen["Serviços Ambientais (R$ bi)"],
                    marker_color="#e74c3c",
                    text=df_cen["Serviços Ambientais (R$ bi)"].apply(lambda v: f"R$ {v:.2f} bi"),
                    textposition="inside",
                )
                fig_cen.add_bar(
                    name="Créditos de Carbono",
                    x=df_cen["Cenário"],
                    y=df_cen["Créditos de Carbono (R$ bi)"],
                    marker_color="#f39c12",
                    text=df_cen["Créditos de Carbono (R$ bi)"].apply(lambda v: f"R$ {v:.2f} bi"),
                    textposition="inside",
                )
                fig_cen.update_layout(
                    barmode="stack", template="plotly_dark", height=360,
                    yaxis_title="R$ Bilhões",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(fig_cen, use_container_width=True)
                st.dataframe(
                    df_cen.rename(columns={"Total (R$ bi)": "💰 Total (R$ bi)"}),
                    hide_index=True, use_container_width=True
                )

                st.markdown("---")

                # --- CONTEXTO EXPLICATIVO ---
                st.markdown("### O que significa esse prejuízo?")
                col_ctx1, col_ctx2 = st.columns(2)
                with col_ctx1:
                    st.markdown(f"""
**Serviços ambientais** são os benefícios que a floresta entrega à sociedade sem custo:
regulação das chuvas, filtragem da água, controle da temperatura e manutenção
do solo e da biodiversidade. Quando uma área queima, esses benefícios deixam
de existir por anos.

Para o **{bioma_calc}**, cada hectare vale em média **R$ {valor_brl_ha:,.0f}**
no cenário {cenario} — valor que a sociedade perde enquanto a vegetação não se recupera.
                    """)
                with col_ctx2:
                    st.markdown(f"""
**Carbono emitido:** as queimadas lançaram aproximadamente
**{co2_emitido_t/1e6:,.2f} milhões de toneladas de CO₂** na atmosfera.

No mercado voluntário de carbono, cada tonelada vale em torno de
**R$ {PRECO_CARBONO_USD * cambio_usd:,.0f}** — o equivalente a
**R$ {credito_carbono_brl/1e9:,.2f} bilhões** em créditos que não poderão mais
ser gerados por essa floresta perdida.
                    """)

                st.info(
                    "Estimativa indicativa com base em literatura científica. "
                    "Não substitui laudo técnico para fins legais ou regulatórios."
                )

                with st.expander("⚙️ Ajustar parâmetros do cálculo"):
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        st.selectbox(
                            "Bioma:", list(VALORES_ECOSSIS.keys()),
                            index=list(VALORES_ECOSSIS.keys()).index(bioma_calc),
                            key="bioma_impacto"
                        )
                    with pc2:
                        st.selectbox(
                            "Cenário:",
                            ["conservador", "moderado", "otimista"],
                            index=["conservador","moderado","otimista"].index(cenario),
                            format_func=lambda x: {"conservador": "Conservador (mínimo)",
                                                   "moderado": "Moderado (referência)",
                                                   "otimista": "Otimista (máximo)"}[x],
                            key="cenario_impacto"
                        )
                    with pc3:
                        st.number_input(
                            f"Câmbio R$/US$ (PTAX hoje: R$ {CAMBIO_FIXO:.2f}):",
                            min_value=1.0, max_value=20.0,
                            value=float(cambio_usd), step=0.10, key="cambio_impacto"
                        )
                    st.caption(
                        "Altere os parâmetros e clique em **Gerar Dashboard** novamente para recalcular."
                    )

                with st.expander("📚 Fontes e referências"):
                    st.markdown("""
| Fonte | O que fornece | Site |
|---|---|---|
| Costanza et al. (2014) — *Global Policy* | Valor de serviços ecossistêmicos por bioma | [acessar](https://www.sciencedirect.com/science/article/pii/S0959378014000685) |
| IPAM — Inst. de Pesquisa Ambiental da Amazônia | Custo de restauração e carbono na Amazônia | [ipam.org.br](https://ipam.org.br) |
| TNC Brasil — The Nature Conservancy | Custo-benefício de conservação por bioma | [tnc.org.br](https://www.tnc.org.br) |
| SEEG / Observatório do Clima | Fator de emissão de CO₂ por bioma (tCO₂e/ha) | [seeg.eco.br](https://seeg.eco.br) |
| Ecosystem Marketplace (2023) | Preço médio de carbono no mercado voluntário | [ecosystemmarketplace.com](https://www.ecosystemmarketplace.com) |
| CEPEA/USP | Custo de oportunidade agrícola e perdas em cadeias produtivas | [cepea.esalq.usp.br](https://www.cepea.esalq.usp.br) |
                    """)

          except Exception as _err_impacto:
            st.error(f"⚠️ Erro na aba de Impacto Econômico: {_err_impacto}")
            import traceback
            st.code(traceback.format_exc(), language="python")


        # ----------------------------------------------------------
        # ABA 5 — EXPORTAR DADOS
        # ----------------------------------------------------------
        with aba_export:
            st.subheader("⬇️ Exportar Dados da Análise")

            if "INPE" in fonte_escolhida:
                if not df_rec.empty:
                    st.markdown(
                        f"**{len(df_rec)} registros** disponíveis para exportação."
                    )
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv = df_rec.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📄 Baixar CSV — Focos INPE",
                            data=csv,
                            file_name=f"focos_{val_sel}_{hoje.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl2:
                        excel_data = gerar_excel(df_rec)
                        st.download_button(
                            label="📊 Baixar Excel — Focos INPE",
                            data=excel_data,
                            file_name=f"focos_{val_sel}_{hoje.strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                else:
                    st.info("Nenhum dado de focos disponível para exportação.")

                if not df_ranking_areas.empty:
                    st.markdown("---")
                    st.markdown("**Ranking de Áreas Protegidas:**")
                    col_dl3, col_dl4 = st.columns(2)
                    with col_dl3:
                        csv_areas = df_ranking_areas.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📄 Baixar CSV — Áreas Protegidas",
                            data=csv_areas,
                            file_name=f"areas_protegidas_{val_sel}_{hoje.strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl4:
                        excel_areas = gerar_excel(df_ranking_areas)
                        st.download_button(
                            label="📊 Baixar Excel — Áreas Protegidas",
                            data=excel_areas,
                            file_name=f"areas_protegidas_{val_sel}_{hoje.strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

            else:
                # MODIS — Série temporal
                if not df_modis_temporal.empty:
                    st.markdown("**Série Temporal Mensal (MODIS):**")
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        csv_mod = df_modis_temporal.to_csv(
                            index=False
                        ).encode('utf-8-sig')
                        st.download_button(
                            label="📄 Baixar CSV — Série Temporal",
                            data=csv_mod,
                            file_name=f"modis_temporal_{val_sel}_{ano_modis}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl2:
                        excel_mod = gerar_excel(df_modis_temporal)
                        st.download_button(
                            label="📊 Baixar Excel — Série Temporal",
                            data=excel_mod,
                            file_name=f"modis_temporal_{val_sel}_{ano_modis}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                # MODIS — Áreas protegidas
                if not df_ranking_areas.empty:
                    st.markdown("---")
                    st.markdown("**Ranking de Áreas Protegidas (MODIS):**")
                    col_dl3, col_dl4 = st.columns(2)
                    with col_dl3:
                        csv_areas = df_ranking_areas.to_csv(
                            index=False
                        ).encode('utf-8-sig')
                        st.download_button(
                            label="📄 Baixar CSV — Áreas Protegidas",
                            data=csv_areas,
                            file_name=f"areas_afetadas_{val_sel}_{ano_modis}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col_dl4:
                        excel_areas = gerar_excel(df_ranking_areas)
                        st.download_button(
                            label="📊 Baixar Excel — Áreas Protegidas",
                            data=excel_areas,
                            file_name=f"areas_afetadas_{val_sel}_{ano_modis}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                # MODIS — Anomalia histórica (se calculada)
                if not df_modis_temporal.empty and ano_modis > 2001:
                    try:
                        df_anomalia_exp = calcular_anomalia_modis(
                            geom_json_str, ano_modis
                        )
                        if not df_anomalia_exp.empty:
                            st.markdown("---")
                            st.markdown("**Dados de Anomalia Histórica:**")
                            csv_anom = df_anomalia_exp.to_csv(
                                index=False
                            ).encode('utf-8-sig')
                            st.download_button(
                                label="📄 Baixar CSV — Anomalia Histórica",
                                data=csv_anom,
                                file_name=f"anomalia_{val_sel}_{ano_modis}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    except:
                        pass

                # NBR Severidade
                if stats_sev:  # <-- DEIXE ASSIM, SIMPLES
                    st.markdown("---")
                    st.markdown("**Distribuição de Severidade NBR (Sentinel-2):**")
                    df_sev_exp = pd.DataFrame(
                        list(stats_sev.items()), columns=['Classe', 'Área (km²)']
                    )
                    csv_sev = df_sev_exp.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📄 Baixar CSV — Severidade NBR",
                        data=csv_sev,
                        file_name=f"nbr_{val_sel}_{ano_modis}_{mes_modis:02d}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        # ----------------------------------------------------------
        # ABA 6 — RISCO PREDITIVO (MACHINE LEARNING)
        # ----------------------------------------------------------
        with aba_risco:
            st.subheader("🎯 Risco Preditivo de Queimada — Mapa por Município")

            with st.expander("ℹ️ Como esse modelo funciona (leia antes de treinar)", expanded=False):
                st.markdown(
                    "- Você escolhe treinar o modelo com **Focos de Calor (INPE)** ou **Área Queimada "
                    "(MODIS)** — o alvo que o modelo aprende a prever muda conforme essa escolha (foco "
                    "pontual de calor vs. pixel de área efetivamente queimada).\n"
                    "- O modelo é treinado com uma **amostra de municípios** da região selecionada "
                    "(bioma ou estado inteiro — não é mais um único ponto central).\n"
                    "- Para **cada** município da amostra, o modelo usa **todos os dias** do período "
                    "de histórico escolhido (clima diário + se houve ou não ocorrência naquele dia) "
                    "— não é uma amostra de dias, é o histórico diário completo.\n"
                    "- Além do clima, o modelo usa a **taxa histórica de cada município** (calculada só "
                    "com o período de treino) como uma feature extra — assim, dois municípios com o "
                    "mesmo clima podem receber risco diferente se um deles historicamente pega fogo "
                    "muito mais que o outro (uso do solo, fronteira agrícola, etc.).\n"
                    "- Depois de treinado, o mesmo modelo é aplicado às condições climáticas **atuais** "
                    "de cada município da região (podendo ser um conjunto maior que o usado no treino) "
                    "para colorir o mapa.\n"
                    "- **Amanhã** usa o clima real mais recente disponível. **Próximo mês** é uma "
                    "**estimativa sazonal** (desloca o risco atual pela diferença histórica entre o mês "
                    "atual e o mês seguinte, em espaço log-odds — para não saturar bruscamente em 100%) "
                    "— não existe fonte gratuita de previsão climática de 30 dias, então isso não é uma "
                    "previsão dia-a-dia, e sim uma tendência baseada em climatologia."
                )

            try:
                fonte_sugerida = "MODIS" if "MODIS" in fonte_escolhida else "INPE"
                fonte_risco = st.radio(
                    "Treinar o modelo com:",
                    ["🔥 Focos de Calor (INPE)", "🗺️ Área Queimada (MODIS)"],
                    index=0 if fonte_sugerida == "INPE" else 1,
                    key="fonte_dados_risco",
                    horizontal=True,
                    help="Focos (INPE) = detecções pontuais de calor, mais rápido de treinar. "
                         "Área Queimada (MODIS) = baseado em pixels queimados detectados por satélite, "
                         "mais coerente com a fonte escolhida na barra lateral, porém mais lento."
                )
                fonte_dados_risco = "MODIS" if "MODIS" in fonte_risco else "INPE"
                if fonte_dados_risco == "MODIS":
                    st.caption(
                        "⏱️ Treinar com Área Queimada (MODIS) é mais lento — cada município consulta "
                        "o Earth Engine mês a mês (em vez de uma busca simples do INPE). Considere usar "
                        "menos municípios e menos anos de histórico."
                    )

                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    anos_hist = st.select_slider(
                        "Histórico para treinar o modelo:",
                        options=[1, 2, 3], value=1 if fonte_dados_risco == "MODIS" else 2,
                        format_func=lambda x: f"{x} ano(s)",
                        key="anos_hist_risco"
                    )
                with col_p2:
                    if tipo_analise == "Por Município":
                        n_amostra_treino = 1
                        st.select_slider(
                            "Municípios usados no treino:",
                            options=[1], value=1,
                            key="n_treino_risco_muni",
                            disabled=True,
                            help="Análise 'Por Município' treina só com o município selecionado."
                        )
                    else:
                        n_amostra_treino = st.select_slider(
                            "Municípios usados no treino:",
                            options=[3, 5, 8, 12, 15, 20],
                            value=5 if fonte_dados_risco == "MODIS" else 8,
                            key="n_treino_risco",
                            help="Mais municípios = modelo mais representativo do bioma/estado, "
                                 "porém mais lento para treinar."
                        )
                with col_p3:
                    horizonte = st.radio(
                        "Horizonte da previsão:",
                        ["Amanhã", "Próximo mês"],
                        key="horizonte_risco",
                        help="'Próximo mês' é uma estimativa sazonal (climatologia), não uma previsão exata."
                    )

                max_mapa = st.select_slider(
                    "Municípios exibidos no mapa (região inteira):",
                    options=[15, 30, 50, 80], value=30,
                    key="max_mapa_risco",
                    help="Quantos municípios da região terão o risco calculado e coloridos no mapa. "
                         "(O mapa sempre usa clima real recente, independente da fonte escolhida acima "
                         "— a fonte só muda o que o modelo aprendeu a reconhecer como risco.)"
                )

                if st.button("🧠 Treinar Modelo e Gerar Mapa de Risco", use_container_width=True):
                    status = st.status("Treinando modelo regional...", expanded=True)

                    status.write(f"🔎 Selecionando municípios de '{val_sel}' para treino...")
                    resultado = treinar_modelo_risco_regional(
                        tipo_analise, estado_dd, bioma_dd, municipio_dd,
                        anos_historico=anos_hist, n_amostra_treino=n_amostra_treino,
                        fonte_dados=fonte_dados_risco
                    )

                    if "erro" in resultado:
                        status.update(label="Falha no treino", state="error")
                        st.warning(f"⚠️ {resultado['erro']}")
                    else:
                        fonte_txt = "Área Queimada (MODIS)" if resultado["fonte_dados"] == "MODIS" else "Focos de Calor (INPE)"
                        msg_treino = (
                            f"✅ Modelo treinado com **{resultado['n_municipios_treino']} município(s)** "
                            f"usando **{fonte_txt}** ({resultado['n_linhas_total']} dias no total, "
                            f"todos os dias do período)."
                        )
                        if resultado["auc"] is not None:
                            msg_treino += f" AUC-ROC no teste: **{resultado['auc']:.2f}**"
                        status.write(msg_treino)

                        if resultado["amostrado_treino"]:
                            status.write(
                                f"ℹ️ A região tem {resultado['total_disponivel']} municípios — "
                                f"uma amostra foi usada para o treino ser mais rápido."
                            )
                        if resultado["n_municipios_falha"] > 0:
                            status.write(
                                f"⚠️ {resultado['n_municipios_falha']} município(s) da amostra não "
                                f"retornaram dados e foram descartados do treino (isso é normal em "
                                f"instabilidades pontuais de rede — não impede o treino de continuar)."
                            )

                        status.write("🗺️ Buscando lista de municípios para o mapa...")
                        gdf_mapa = obter_municipios_regiao(
                            tipo_analise, estado_dd, bioma_dd, municipio_dd, max_municipios=max_mapa
                        )

                        status.write(f"🌡️ Calculando risco atual em {len(gdf_mapa)} município(s)...")
                        modelo, scaler = resultado["modelo"], resultado["scaler"]
                        hoje = datetime.now()
                        mes_atual = hoje.month
                        mes_alvo = (mes_atual % 12) + 1
                        clima_mensal = resultado["climatologia_mensal"]
                        clim_atual = clima_mensal.get(mes_atual, resultado["taxa_base"])
                        clim_alvo = clima_mensal.get(mes_alvo, resultado["taxa_base"])
                        # Deslocamento sazonal em ESPAÇO LOGIT (log-odds), não multiplicativo direto.
                        # Isso evita que o ajuste "exploda" para 100% quando a taxa histórica do mês
                        # atual é muito baixa e a do mês-alvo é muito mais alta — o deslocamento em
                        # log-odds satura suavemente perto dos extremos, igual o próprio modelo logístico.
                        deslocamento_sazonal = _logit(clim_alvo) - _logit(clim_atual)

                        linhas_municipios_mapa = [row for _, row in gdf_mapa.iterrows()]

                        # Busca as condições atuais de todos os municípios em paralelo
                        # (até 15 de cada vez) — a previsão do modelo em si é instantânea,
                        # o gargalo é só a chamada de rede.
                        condicoes = {}
                        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                            futuros = {
                                executor.submit(buscar_condicoes_atuais, r["lat"], r["lon"]): r["name_muni"]
                                for r in linhas_municipios_mapa
                            }
                            for futuro in concurrent.futures.as_completed(futuros):
                                nome_muni = futuros[futuro]
                                try:
                                    condicoes[nome_muni] = futuro.result()
                                except Exception:
                                    condicoes[nome_muni] = None  # falha isolada não derruba o mapa

                        n_falhas_mapa = sum(1 for v in condicoes.values() if v is None)
                        if n_falhas_mapa > 0:
                            status.write(
                                f"⚠️ {n_falhas_mapa} de {len(linhas_municipios_mapa)} município(s) "
                                f"não retornaram dados climáticos e ficarão de fora do mapa."
                            )

                        linhas_resultado = []
                        cond_por_municipio = {}
                        taxa_por_municipio = resultado["taxa_por_municipio"]
                        taxa_fallback = resultado["taxa_geral_fallback"]
                        for row in linhas_municipios_mapa:
                            cond = condicoes.get(row["name_muni"])
                            if cond is None:
                                continue
                            cond = cond.copy()
                            # Município que participou do treino usa sua própria taxa histórica
                            # conhecida; município novo (só apareceu no mapa) usa a média geral
                            # da amostra de treino como aproximação razoável.
                            cond["taxa_hist_municipio"] = taxa_por_municipio.get(row["name_muni"], taxa_fallback)
                            X = scaler.transform(cond[FEATURES_RISCO])
                            prob_amanha = float(modelo.predict_proba(X)[0, 1])
                            if horizonte == "Próximo mês":
                                prob_final = float(_sigmoid(_logit(prob_amanha) + deslocamento_sazonal))
                            else:
                                prob_final = prob_amanha
                            nivel, emoji, cor_hex = classificar_risco(prob_final)
                            linhas_resultado.append({
                                "name_muni": row["name_muni"], "lat": row["lat"], "lon": row["lon"],
                                "probabilidade": prob_final, "nivel": nivel, "emoji": emoji, "cor": cor_hex,
                            })
                            cond_por_municipio[row["name_muni"]] = (cond, X)

                        status.update(label="✅ Mapa de risco pronto", state="complete")

                    if "erro" not in resultado:
                        if not linhas_resultado:
                            st.error("⚠️ Não foi possível calcular o risco atual para os municípios desta região.")
                        else:
                            df_res = pd.DataFrame(linhas_resultado)
                            df_res["probabilidade_fmt"] = df_res["probabilidade"].apply(lambda p: f"{p:.0%}")

                            st.markdown("---")
                            titulo_horiz = "Amanhã" if horizonte == "Amanhã" else f"Próximo mês (estimativa sazonal)"
                            st.markdown(f"### Risco por município — {titulo_horiz}")

                            contagem = df_res["nivel"].value_counts()
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            col_m1.metric("🟢 Baixo", int(contagem.get("Baixo", 0)))
                            col_m2.metric("🟡 Moderado", int(contagem.get("Moderado", 0)))
                            col_m3.metric("🟠 Alto", int(contagem.get("Alto", 0)))
                            col_m4.metric("🔴 Crítico", int(contagem.get("Crítico", 0)))

                            # ---- Mapa ----
                            gdf_render = gdf_mapa.merge(
                                df_res[["name_muni", "probabilidade", "probabilidade_fmt", "nivel", "cor"]],
                                on="name_muni", how="inner"
                            )
                            centro_mapa = geom_unida.centroid
                            m_risco = folium.Map(
                                location=[centro_mapa.y, centro_mapa.x],
                                zoom_start=6 if tipo_analise != "Por Município" else 10,
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri", prefer_canvas=True,
                            )

                            # --- Limites de referência (para o usuário se localizar) ---
                            # Contorno da região selecionada (bioma/estado/município) — igual ao mapa principal
                            folium.GeoJson(
                                limite.__geo_interface__,
                                name=f"Limite — {val_sel}",
                                style_function=lambda x: {
                                    'fillColor': '#00d4ff', 'fillOpacity': 0.02,
                                    'color': '#00d4ff', 'weight': 3, 'dashArray': '6 3',
                                },
                            ).add_to(m_risco)

                            # Limites dos municípios (contorno fino, sem preenchimento)
                            folium.GeoJson(
                                gdf_mapa.__geo_interface__,
                                name="Limites municipais",
                                style_function=lambda x: {
                                    'fillOpacity': 0, 'color': '#95a5a6', 'weight': 0.8, 'dashArray': '2 2',
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=["name_muni"], aliases=["Município:"],
                                    style="font-size:11px; background:white; color:#2c3e50; border-radius:6px;",
                                ),
                            ).add_to(m_risco)

                            # Limites estaduais (útil quando a região é um bioma, que cruza vários estados)
                            ufs_envolvidas = sorted(gdf_mapa["abbrev_state"].dropna().unique().tolist())
                            if tipo_analise == "Por Bioma" and 0 < len(ufs_envolvidas) <= 12:
                                try:
                                    frames_uf = [read_state(code_state=uf, year=2020) for uf in ufs_envolvidas]
                                    gdf_estados = pd.concat(frames_uf, ignore_index=True).to_crs("EPSG:4326")
                                    folium.GeoJson(
                                        gdf_estados.__geo_interface__,
                                        name="Limites estaduais",
                                        style_function=lambda x: {
                                            'fillOpacity': 0, 'color': '#ecf0f1', 'weight': 1.8,
                                        },
                                        tooltip=folium.GeoJsonTooltip(
                                            fields=["abbrev_state"], aliases=["Estado:"],
                                            style="font-size:11px; background:white; color:#2c3e50; border-radius:6px;",
                                        ),
                                    ).add_to(m_risco)
                                except Exception:
                                    pass  # limites estaduais são só um extra visual — não interrompe o mapa se falhar

                            # --- Risco por município, em formato de FOCO (ponto), não polígono preenchido ---
                            fg_focos_risco = folium.FeatureGroup(name="Risco de queimada (focos)")
                            for _, row_r in gdf_render.iterrows():
                                raio = 6 + (row_r["probabilidade"] * 14)  # 6px a 20px conforme a probabilidade
                                folium.CircleMarker(
                                    location=[row_r["lat"], row_r["lon"]],
                                    radius=raio,
                                    color="#1c1c1c",
                                    weight=1,
                                    fill=True,
                                    fill_color=row_r["cor"],
                                    fill_opacity=0.85,
                                    tooltip=folium.Tooltip(
                                        f"<b>{row_r['name_muni']}</b><br>"
                                        f"Risco: {row_r['nivel']}<br>"
                                        f"Probabilidade: {row_r['probabilidade_fmt']}",
                                        style=(
                                            "font-size:12px; background:white; color:#2c3e50; "
                                            "border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.25);"
                                        ),
                                    ),
                                ).add_to(fg_focos_risco)
                            fg_focos_risco.add_to(m_risco)

                            legenda_risco = """
                            <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999;
                                        background: rgba(30,30,30,0.9); padding: 12px 16px; border-radius: 10px;
                                        color: white; font-size: 13px; box-shadow: 2px 2px 8px rgba(0,0,0,0.4);">
                                <b>🎯 Nível de Risco (por foco/município)</b><br>
                                <span style="color:#2ecc71;">●</span> Baixo (&lt;25%)<br>
                                <span style="color:#f1c40f;">●</span> Moderado (25-50%)<br>
                                <span style="color:#e67e22;">●</span> Alto (50-75%)<br>
                                <span style="color:#e74c3c;">●</span> Crítico (≥75%)<br>
                                <span style="color:#95a5a6;">- - -</span> Limite municipal<br>
                                <span style="color:#00d4ff;">- - -</span> Região selecionada
                            </div>
                            """
                            m_risco.get_root().html.add_child(folium.Element(legenda_risco))
                            folium.LayerControl(collapsed=False).add_to(m_risco)
                            st_folium(m_risco, width=None, height=650, returned_objects=[], key="mapa_risco_ml")

                            # ---- Ranking dos municípios em maior risco ----
                            st.markdown("**🔥 Municípios com maior risco na região:**")
                            st.dataframe(
                                df_res.sort_values("probabilidade", ascending=False)
                                      .head(10)[["name_muni", "probabilidade_fmt", "nivel"]]
                                      .rename(columns={"name_muni": "Município", "probabilidade_fmt": "Probabilidade", "nivel": "Nível"}),
                                use_container_width=True, hide_index=True
                            )

                            # ---- Explicabilidade do município mais crítico ----
                            municipio_top = df_res.sort_values("probabilidade", ascending=False).iloc[0]["name_muni"]
                            cond_top, X_top = cond_por_municipio[municipio_top]
                            contrib = pd.DataFrame({
                                "feature": FEATURES_RISCO,
                                "contribuicao": (modelo.coef_[0] * X_top[0]),
                            }).sort_values("contribuicao", key=abs, ascending=False)

                            st.markdown(f"**🔍 Por que {municipio_top} está com o maior risco:**")
                            fig_contrib = px.bar(
                                contrib, x="contribuicao", y="feature",
                                orientation="h", color="contribuicao",
                                color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"],
                            )
                            fig_contrib.update_layout(
                                template="plotly_dark", height=380,
                                yaxis={'categoryorder': 'total ascending'},
                                coloraxis_showscale=False,
                                xaxis_title="Contribuição (+ aumenta risco / - reduz risco)",
                                yaxis_title=""
                            )
                            st.plotly_chart(fig_contrib, use_container_width=True)

                            # ---- Painel de Transparência do Modelo ----
                            st.markdown("---")
                            st.markdown("### 🔬 Transparência do Modelo")
                            st.caption(
                                "Todas as métricas abaixo são calculadas SÓ no conjunto de teste "
                                "(período mais recente, nunca visto no treino) — igual seria em produção."
                            )
                            aval = resultado.get("avaliacao_teste")
                            if aval is None:
                                st.info(
                                    "Não há dados de teste suficientes (ou só uma classe presente) "
                                    "para gerar as métricas de avaliação nesta região/período."
                                )
                            else:
                                from sklearn.metrics import (
                                    confusion_matrix, precision_score, recall_score,
                                    f1_score, accuracy_score
                                )

                                y_true = aval["y_true"]
                                y_prob = aval["y_prob"]
                                y_pred = (y_prob >= 0.5).astype(int)
                                y_pred_base = aval["y_pred_baseline"]

                                tab_metricas, tab_matriz, tab_calib, tab_baseline = st.tabs(
                                    ["📈 Métricas", "🧩 Matriz de Confusão", "🎯 Calibração", "⚖️ Vs. Baseline"]
                                )

                                # --- Métricas resumo ---
                                with tab_metricas:
                                    prec = precision_score(y_true, y_pred, zero_division=0)
                                    rec = recall_score(y_true, y_pred, zero_division=0)
                                    f1 = f1_score(y_true, y_pred, zero_division=0)
                                    acc = accuracy_score(y_true, y_pred)
                                    cme1, cme2, cme3, cme4, cme5 = st.columns(5)
                                    cme1.metric("AUC-ROC", f"{resultado['auc']:.2f}" if resultado["auc"] else "—")
                                    cme2.metric("Acurácia", f"{acc:.0%}")
                                    cme3.metric("Precisão", f"{prec:.0%}")
                                    cme4.metric("Recall", f"{rec:.0%}")
                                    cme5.metric("F1-score", f"{f1:.0%}")
                                    st.caption(
                                        "Corte de decisão em 50% de probabilidade. **Precisão** = dos dias que "
                                        "o modelo marcou como risco, quantos realmente tiveram ocorrência. "
                                        "**Recall** = das ocorrências reais, quantas o modelo conseguiu antecipar. "
                                        f"Avaliado em **{len(y_true)}** dias de teste."
                                    )

                                # --- Matriz de confusão ---
                                with tab_matriz:
                                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                                    fig_cm = px.imshow(
                                        cm, text_auto=True,
                                        x=["Previu: Sem risco", "Previu: Risco"],
                                        y=["Real: Sem ocorrência", "Real: Com ocorrência"],
                                        color_continuous_scale="Reds",
                                    )
                                    fig_cm.update_layout(template="plotly_dark", height=380, coloraxis_showscale=False)
                                    st.plotly_chart(fig_cm, use_container_width=True)
                                    st.caption(
                                        "Diagonal principal = acertos. Fora da diagonal = erros do modelo "
                                        "(falso positivo: alarme sem ocorrência real / falso negativo: "
                                        "ocorrência que o modelo não previu)."
                                    )

                                # --- Curva de calibração ---
                                with tab_calib:
                                    try:
                                        from sklearn.calibration import calibration_curve
                                        n_bins = 5 if len(y_true) < 200 else 10
                                        frac_pos, prob_media = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
                                        fig_calib = go.Figure()
                                        fig_calib.add_trace(go.Scatter(
                                            x=[0, 1], y=[0, 1], mode="lines", name="Calibração perfeita",
                                            line=dict(dash="dash", color="#7f8c8d")
                                        ))
                                        fig_calib.add_trace(go.Scatter(
                                            x=prob_media, y=frac_pos, mode="lines+markers", name="Modelo",
                                            line=dict(color="#e74c3c", width=3), marker=dict(size=9)
                                        ))
                                        fig_calib.update_layout(
                                            template="plotly_dark", height=380,
                                            xaxis_title="Probabilidade prevista (média por faixa)",
                                            yaxis_title="Frequência real de ocorrência",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                        )
                                        st.plotly_chart(fig_calib, use_container_width=True)
                                        st.caption(
                                            "Quanto mais perto da linha tracejada, mais confiável é a "
                                            "probabilidade — ou seja, quando o modelo diz '70% de risco', "
                                            "isso realmente acontece em ~70% desses dias."
                                        )
                                    except Exception:
                                        st.info("Dados de teste insuficientes para calcular a curva de calibração.")

                                # --- Comparação com baseline ingênuo ---
                                with tab_baseline:
                                    prec_b = precision_score(y_true, y_pred_base, zero_division=0)
                                    rec_b = recall_score(y_true, y_pred_base, zero_division=0)
                                    f1_b = f1_score(y_true, y_pred_base, zero_division=0)
                                    acc_b = accuracy_score(y_true, y_pred_base)
                                    df_comp = pd.DataFrame({
                                        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-score"],
                                        "Modelo (clima + ML)": [acc, prec, rec, f1],
                                        "Baseline ingênuo (focos nos últimos 7 dias)": [acc_b, prec_b, rec_b, f1_b],
                                    })
                                    fig_comp = go.Figure()
                                    fig_comp.add_bar(name="Modelo (clima + ML)", x=df_comp["Métrica"],
                                                      y=df_comp["Modelo (clima + ML)"], marker_color="#e74c3c")
                                    fig_comp.add_bar(name="Baseline ingênuo", x=df_comp["Métrica"],
                                                      y=df_comp["Baseline ingênuo (focos nos últimos 7 dias)"],
                                                      marker_color="#7f8c8d")
                                    fig_comp.update_layout(
                                        barmode="group", template="plotly_dark", height=360,
                                        yaxis_tickformat=".0%",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                    )
                                    st.plotly_chart(fig_comp, use_container_width=True)
                                    st.caption(
                                        "**Baseline ingênuo:** marca risco alto se o município teve pelo menos "
                                        "1 dia com ocorrência nos 7 dias anteriores — sem clima, sem ML, só "
                                        "persistência recente. É a régua mínima que o modelo precisa superar "
                                        "pra justificar a complexidade extra."
                                    )
                                    ganho_f1 = f1 - f1_b
                                    if ganho_f1 > 0:
                                        st.success(
                                            f"✅ O modelo supera o baseline em F1-score por **+{ganho_f1:.0%}** "
                                            "— o clima e o histórico por município agregam valor real."
                                        )
                                    else:
                                        st.warning(
                                            f"⚠️ O modelo não superou o baseline ingênuo nesta região/período "
                                            f"(F1 {ganho_f1:.0%}). Considere mais dados de treino ou revisar features."
                                        )

                            with st.expander("📊 Detalhes do treinamento do modelo"):
                                st.write(f"- Municípios usados no treino: **{', '.join(resultado['municipios_treino'])}**")
                                st.write(
                                    f"- Dias de treino: **{resultado['n_dias_treino']}** | "
                                    f"Dias de teste (mais recentes): **{resultado['n_dias_teste']}**"
                                )
                                st.write(f"- Taxa histórica média de dias com foco: **{resultado['taxa_base']:.1%}**")
                                if horizonte == "Próximo mês":
                                    sinal_desloc = "+" if deslocamento_sazonal >= 0 else ""
                                    st.write(
                                        f"- Climatologia do mês atual (mês {mes_atual}): **{clim_atual:.1%}** "
                                        f"→ do mês-alvo (mês {mes_alvo}): **{clim_alvo:.1%}**"
                                    )
                                    st.write(
                                        f"- Deslocamento sazonal aplicado (em log-odds): "
                                        f"**{sinal_desloc}{deslocamento_sazonal:.2f}**"
                                    )
                                st.caption(
                                    "Split temporal: o modelo treina no passado e é testado no período mais "
                                    "recente, simulando o uso real (prever o que ainda não aconteceu)."
                                )

                            st.info(
                                "💡 **Nota metodológica:** prova de conceito educacional (MBA). O horizonte "
                                "'Próximo mês' é uma estimativa sazonal baseada em climatologia histórica, "
                                "não uma previsão climática real (não há fonte gratuita de previsão de 30 "
                                "dias). Para uso operacional, recomenda-se incorporar NDVI, uso do solo e "
                                "comparar com modelos mais robustos (Random Forest, XGBoost)."
                            )
                else:
                    st.info(
                        "👆 Ajuste os parâmetros acima e clique no botão para treinar o modelo "
                        "e gerar o mapa de risco da região selecionada."
                    )
            except Exception as _err_risco:
                st.error(f"⚠️ Erro na aba de Risco Preditivo: {_err_risco}")
                st.code(traceback.format_exc(), language="python")

else:
    st.info(
        "👈 Use os filtros ao lado para selecionar a Fonte de Dados, "
        "o local e o período de análise. Depois clique em **'Gerar Dashboard'**."
    )
