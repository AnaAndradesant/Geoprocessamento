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


# =====================================================================
# HELPER: retry com backoff exponencial para erros 429 do GEE
# "Too many concurrent aggregations" acontece quando várias chamadas
# .getInfo() pesadas são disparadas em sequência rápida.
# A solução padrão do GEE é esperar e tentar novamente.
# =====================================================================
def _gee_retry(fn, max_tentativas=4, espera_inicial=6):
    """
    Executa fn() (que deve chamar .getInfo() internamente) com retry
    exponencial para erros HTTP 429 / 'Too many concurrent aggregations'.
    Tentativas: 6s → 12s → 24s → desiste.
    """
    for tentativa in range(max_tentativas):
        try:
            return fn()
        except Exception as e:
            eh_ultimo = tentativa == max_tentativas - 1
            msg = str(e).lower()
            eh_429 = any(t in msg for t in [
                "too many concurrent", "429", "quota", "rate limit", "resource exhausted"
            ])
            if eh_ultimo or not eh_429:
                raise   # erro diferente de throttling → propaga imediatamente
            espera = espera_inicial * (2 ** tentativa)  # 6 → 12 → 24s
            time.sleep(espera)





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
def calcular_stats_nbr(geom_json_str, ano, mes, area_km2_hint=0, _mascara_modis=None):
    # === LEITURA INTELIGENTE DE GEOJSON ===
    geom_dict = json.loads(geom_json_str)
    if 'features' in geom_dict:
        poly = ee.Geometry(geom_dict['features'][0]['geometry'])
    else:
        poly = ee.Geometry(geom_dict)
    # ======================================

    # === ESCALA DINÂMICA baseada em area_km2_hint (calculada localmente, sem getInfo()) ===
    # Amazônia (~5,5M km²): scale=500 ainda causava timeout porque a geometria complexa
    # mais o composito Sentinel-2 ultrapassavam o limite de 5min do getInfo().
    # Solução: escala 1000m para biomas gigantes + simplificação da geometria no GEE.
    area_km2 = area_km2_hint  # recebido de fora, calculado via Shapely (sem chamada GEE)
    if area_km2 > 1_500_000:       # Biomas gigantes (ex: Amazônia ~5,5M km²)
        scale = 1000
        max_error_simplify = 10000  # ~10 km de tolerância — sem impacto visual em 1000m
        tile_scale = 32
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
    # ========================================================================================

    # === SIMPLIFICAÇÃO DA GEOMETRIA NO GEE ===
    # Amazônia do geobr tem milhares de vértices. Enviar geometria complexa ao GEE
    # lentifica filterBounds, clip e reduceRegion. Simplificar com max_error adequado
    # não afeta o resultado a escalas de centenas de metros.
    if area_km2 > 10_000:
        poly = poly.simplify(maxError=max_error_simplify)
    # ==========================================

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
        scale=scale,           # ✅ escala dinâmica conforme tamanho da região
        maxPixels=1e13,
        tileScale=tile_scale,  # ✅ também dinâmico: 32 para Amazônia evita OOM no GEE
        bestEffort=True
    ).getInfo()

    classes = {0: 'Regeneração', 1: 'Não afetado', 2: 'Baixa',
               3: 'Moderada', 4: 'Moderada-Alta', 5: 'Alta'}
    res_stats = {}

    if stats:
        hist = list(stats.values())[0]
        if isinstance(hist, dict):
            # ✅ Área correta: usa scale² m²/pixel
            pixel_area_km2 = (scale * scale) / 1e6
            for k, v in hist.items():
                area = v * pixel_area_km2
                res_stats[classes.get(int(float(k)), 'Outros')] = round(area, 2)
        elif not hist:
            # Histograma vazio = nenhum pixel queimado no período — retorna vazio graciosamente
            pass

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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@st.cache_data(ttl=3600, show_spinner=False)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, satelites):
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
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
    all_dfs = []
    sat_str = "','".join(satelites)

    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=1), dt_fim)
        cql = (
            f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' "
            f"AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' "
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
                headers=headers,
                verify=False,
                timeout=90
            )
            if r.status_code == 200:
                dados_json = r.json()
                if "features" in dados_json and len(dados_json["features"]) > 0:
                    registros = [
                        {"longitude": f["geometry"]["coordinates"][0],
                         "latitude": f["geometry"]["coordinates"][1],
                         **f["properties"]}
                        for f in dados_json["features"]
                    ]
                    all_dfs.append(pd.DataFrame(registros))
        except Exception:
            pass  # Ignora erros de conexão para não parar o loop
            
        dt_ini = dt_bloco_fim + timedelta(days=1)

    if not all_dfs:
        return pd.DataFrame()

    df_final = pd.concat(all_dfs, ignore_index=True)
    if 'id' in df_final.columns:
        df_final = df_final.drop_duplicates(subset=['id'])
    else:
        df_final = df_final.drop_duplicates()
    return df_final



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


# =====================================================================
# SICAR — Consulta pública de imóveis rurais cadastrados
# Fonte: Sistema Nacional de Cadastro Ambiental Rural (car.gov.br)
# =====================================================================

# Estatísticas SICAR 2023 por UF (fallback se API indisponível)
# Fonte: Relatório Anual SICAR 2023 — MAPA/SENAR
_SICAR_FALLBACK = {
    "AC": {"imoveis": 37_800,  "area_mha": 8.1},
    "AL": {"imoveis": 62_100,  "area_mha": 2.0},
    "AM": {"imoveis": 169_000, "area_mha": 57.2},
    "AP": {"imoveis": 17_200,  "area_mha": 4.4},
    "BA": {"imoveis": 456_000, "area_mha": 34.0},
    "CE": {"imoveis": 212_000, "area_mha": 8.4},
    "DF": {"imoveis": 4_100,   "area_mha": 0.3},
    "ES": {"imoveis": 98_000,  "area_mha": 2.9},
    "GO": {"imoveis": 212_000, "area_mha": 25.8},
    "MA": {"imoveis": 262_000, "area_mha": 23.4},
    "MG": {"imoveis": 595_000, "area_mha": 44.6},
    "MS": {"imoveis": 93_000,  "area_mha": 27.1},
    "MT": {"imoveis": 271_000, "area_mha": 74.2},
    "PA": {"imoveis": 593_000, "area_mha": 81.3},
    "PB": {"imoveis": 118_000, "area_mha": 4.3},
    "PE": {"imoveis": 164_000, "area_mha": 6.8},
    "PI": {"imoveis": 155_000, "area_mha": 14.1},
    "PR": {"imoveis": 288_000, "area_mha": 13.7},
    "RJ": {"imoveis": 75_000,  "area_mha": 3.2},
    "RN": {"imoveis": 106_000, "area_mha": 4.5},
    "RO": {"imoveis": 128_000, "area_mha": 17.2},
    "RR": {"imoveis": 29_500,  "area_mha": 10.4},
    "RS": {"imoveis": 362_000, "area_mha": 19.1},
    "SC": {"imoveis": 194_000, "area_mha": 7.8},
    "SE": {"imoveis": 55_000,  "area_mha": 2.1},
    "SP": {"imoveis": 351_000, "area_mha": 19.6},
    "TO": {"imoveis": 108_000, "area_mha": 22.5},
}

@st.cache_data(ttl=3600, show_spinner=False)
def consultar_sicar(uf: str) -> dict:
    """Tenta obter dados do CAR via API pública do SICAR; usa fallback se indisponível."""
    try:
        url = (
            f"https://car.gov.br/publico/estados/quantidadeImoveis"
            f"?sigla={uf}&situacao=AT"
        )
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            dados = r.json()
            if isinstance(dados, dict) and "quantidade" in dados:
                return {
                    "imoveis": int(dados["quantidade"]),
                    "area_mha": _SICAR_FALLBACK.get(uf, {}).get("area_mha", 0),
                    "fonte": "API SICAR (tempo real)"
                }
    except Exception:
        pass
    fb = _SICAR_FALLBACK.get(uf, {"imoveis": 0, "area_mha": 0})
    return {**fb, "fonte": "Relatório SICAR 2023 (offline)"}


def calcular_score_esg(
    total_valor: float,
    area_km2: float,
    fonte: str,
    bioma_nome: str,
    df_anomalia: "pd.DataFrame",
) -> dict:
    """
    Calcula o ESG Risk Score (0–100) em três pilares:
      • Exposição ao Fogo   (50%) — intensidade relativa ao benchmark do bioma
      • Recorrência         (30%) — quantos meses acima da média histórica
      • Sensibilidade       (20%) — fragilidade ecológica do bioma

    Retorna dict com scores parciais, score final e tier (Verde/Amarelo/Laranja/Vermelho).
    """
    # ── Benchmarks de % área queimada/ano por bioma (literatura INPE/MapBiomas) ──
    BENCH_PCT = {
        "Amazônia": 0.30, "Cerrado": 2.50, "Pantanal": 1.50,
        "Mata Atlântica": 0.10, "Caatinga": 1.00, "Pampa": 0.20,
    }
    # ── Benchmark de densidade de focos/1000km² por bioma ───────────────────────
    BENCH_FOCOS = {
        "Amazônia": 40, "Cerrado": 120, "Pantanal": 60,
        "Mata Atlântica": 20, "Caatinga": 80, "Pampa": 15,
    }
    # ── Sensibilidade ecológica (0–100) — IUCN / WWF ─────────────────────────────
    SENSIB = {
        "Mata Atlântica": 90, "Pantanal": 82, "Amazônia": 75,
        "Cerrado": 62, "Caatinga": 48, "Pampa": 38,
    }

    # Pilar 1 — Exposição
    if area_km2 > 0:
        if "MODIS" in fonte:
            pct = (total_valor / area_km2) * 100
            bench = BENCH_PCT.get(bioma_nome, 1.0)
            # > 4× benchmark = score 100
            score_exp = min(100.0, (pct / bench) * 25)
        else:                                          # INPE (focos)
            dens = (total_valor / area_km2) * 1000    # focos / 1000 km²
            bench = BENCH_FOCOS.get(bioma_nome, 60)
            score_exp = min(100.0, (dens / bench) * 25)
    else:
        score_exp = 0.0

    # Pilar 2 — Recorrência histórica
    if not df_anomalia.empty and "Anomalia (%)" in df_anomalia.columns:
        meses_criticos = int((df_anomalia["Anomalia (%)"] > 30).sum())
        score_rec = min(100.0, meses_criticos * 12.5)   # 8 meses = 100
    else:
        score_rec = 50.0   # desconhecido = moderado

    # Pilar 3 — Sensibilidade do bioma
    score_sens = float(SENSIB.get(bioma_nome, 60))

    # Score final ponderado
    score_final = 0.50 * score_exp + 0.30 * score_rec + 0.20 * score_sens

    # Tier
    if score_final < 25:
        tier, cor = "Verde",    "#27ae60"
    elif score_final < 50:
        tier, cor = "Amarelo",  "#f39c12"
    elif score_final < 75:
        tier, cor = "Laranja",  "#e67e22"
    else:
        tier, cor = "Vermelho", "#e74c3c"

    return {
        "score_final":  round(score_final, 1),
        "score_exp":    round(score_exp, 1),
        "score_rec":    round(score_rec, 1),
        "score_sens":   round(score_sens, 1),
        "tier":         tier,
        "cor":          cor,
    }


@st.cache_data(ttl=86400, show_spinner=False)
def calcular_anomalia_modis(geom_json_str, ano_ref):
    ee_geom = ee.Geometry(json.loads(geom_json_str))

    # ✅ Simplifica a geometria UMA VEZ aqui fora — antes era recalculada
    #    dentro de get_area_km2 a cada imagem processada (muito desperdício)
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
            .filterBounds(geom_simplificada)   # ✅ usa geom simplificada no filterBounds também
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

    # ✅ OTIMIZAÇÃO PRINCIPAL: 1 única chamada .getInfo() no lugar de 12 sequenciais
    # Antes: loop Python com getInfo() por mês → até 36 roundtrips ao EE (12 meses × 3 tentativas)
    # Agora: todo o cálculo dos 12 meses vai pro servidor de uma vez, volta num único resultado
    meses_ee = ee.List.sequence(1, 12)
    _fc_anomalia = ee.FeatureCollection(meses_ee.map(calc_mes_feature))
    resultado = _gee_retry(lambda: _fc_anomalia.getInfo())

    registros = []
    for feat in resultado['features']:
        p = feat['properties']
        mes      = int(p['mes'])
        val_ref  = round(float(p.get('area_ref')  or 0), 2)
        media    = round(float(p.get('media_hist') or 0), 2)
        anomalia = round(((val_ref - media) / media * 100), 1) if media > 0 else 0
        registros.append({
            'Mês': mes,
            'Mês Nome': meses_map[mes],
            f'Área {ano_ref} (km²)': val_ref,
            'Média Histórica (km²)': media,
            'Anomalia (%)': anomalia
        })

    return pd.DataFrame(sorted(registros, key=lambda x: x['Mês']))

def _construir_dnbr(geom_json_str, ano, mes, area_km2_hint=0, _mascara_modis=None):
    # 1. LEITURA CORRETA DA GEOMETRIA (Igual ao seu calcular_stats_nbr)
    geom_dict = json.loads(geom_json_str)
    if 'features' in geom_dict:
        poly = ee.Geometry(geom_dict['features'][0]['geometry'])
    else:
        poly = ee.Geometry(geom_dict)

    # 1b. SIMPLIFICAÇÃO DA GEOMETRIA — crítico para Amazônia
    # Geometria complexa do geobr tem milhares de vértices; simplificar
    # reduz o tempo de getMapId() e torna o mapa renderizável em ~2s.
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


# Adicione isso na seção da barra lateral (sidebar)
if st.sidebar.button("♻️ Limpar Cache do Sistema"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("Cache limpo com sucesso!")


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

        # ✅ Área calculada localmente (via GeoPandas/Shapely) — sem nenhuma chamada GEE
        # Isso elimina o poly.area().getInfo() que antes travava por 30-60s na Amazônia
        _limite_proj = limite.to_crs("EPSG:6933")          # projeção equal-area mundial
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
                            collection=muns_ee,
                            reducer=ee.Reducer.sum(),
                            scale=1000,
                            tileScale=4    # ✅ evita timeout em biomas grandes (ex: Amazônia)
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
                                scale=1000,
                                maxPixels=1e13,    # ✅ era 1e10, insuficiente pra Amazônia
                                tileScale=4,       # ✅ divide o cálculo em blocos
                                bestEffort=True    # ✅ ajusta escala automaticamente se necessário
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
    # --- RENDERIZAÇÃO ---
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

        # =============================================================
        # --- ABAS PRINCIPAIS ---
        # =============================================================
        aba_mapa, aba_graficos, aba_nbr, aba_custos, aba_esg, aba_export = st.tabs([
            "🗺️ Mapa de Focos",
            "📈 Gráficos & Anomalia",
            "🔬 Severidade (NBR Sentinel-2)",
            "💰 Custo de Combate",
            "🌿 ESG & CAR",
            "⬇️ Exportar Dados"
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
            if "INPE" in fonte_escolhida:
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
                        df_anomalia = pd.DataFrame()
                        try:
                            df_anomalia = calcular_anomalia_modis(geom_json_str, ano_modis)
                        except Exception as _e_anom:
                            st.error(
                                f"⚠️ Falha ao calcular anomalia histórica.\n\n"
                                f"**Tipo:** `{type(_e_anom).__name__}`\n\n"
                                f"**Mensagem:** {_e_anom}"
                            )
                            with st.expander("📋 Traceback completo", expanded=False):
                                st.code(traceback.format_exc(), language="python")

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

                with st.expander("📖 Como interpretar as classes de severidade dNBR?", expanded=False):
                    st.markdown("""
                    <style>
                    .dnbr-bar { display:flex; width:100%; height:36px; border-radius:6px; overflow:hidden; margin-bottom:6px; }
                    .dnbr-bar div { display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:600; }
                    .dnbr-ticks { display:flex; justify-content:space-between; font-size:11px; color:#888; margin-bottom:18px; padding:0 1px; }
                    .dnbr-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
                    .dnbr-card { display:flex; border:0.5px solid rgba(128,128,128,0.25); border-radius:8px; overflow:hidden; }
                    .dnbr-stripe { width:10px; flex-shrink:0; }
                    .dnbr-body { padding:8px 10px; }
                    .dnbr-name { font-size:13px; font-weight:600; margin:0 0 2px; }
                    .dnbr-range { font-size:11px; color:#888; margin:0 0 4px; font-family:monospace; }
                    .dnbr-desc { font-size:12px; color:#aaa; margin:0; line-height:1.4; }
                    .dnbr-formula { background:rgba(128,128,128,0.1); border:0.5px solid rgba(128,128,128,0.2);
                                    border-radius:8px; padding:10px 14px; font-size:13px;
                                    margin-bottom:16px; line-height:1.8; }
                    .dnbr-formula code { background:rgba(128,128,128,0.15); border-radius:4px;
                                         padding:1px 6px; font-size:12px; font-family:monospace; }
                    </style>

                    <div class="dnbr-formula">
                        <b>Como é calculado:</b><br>
                        <code>NBR = (B8 − B12) ÷ (B8 + B12)</code> &nbsp;→&nbsp; aplicado na imagem <b>pré-fogo</b> e na <b>pós-fogo</b><br>
                        <code>dNBR = NBR_pré − NBR_pós</code> &nbsp;×&nbsp; 1000 &nbsp;&nbsp;
                        <span style="color:#888; font-size:12px;">(multiplicado para trabalhar com inteiros)</span>
                    </div>

                    <div class="dnbr-bar">
                        <div style="width:13%; background:#1a9850; color:#fff;">Reg.</div>
                        <div style="width:22%; background:#91cf60; color:#3d5a10;">Não afetado</div>
                        <div style="width:18%; background:#fee08b; color:#7a5800;">Baixa</div>
                        <div style="width:18%; background:#fc8d59; color:#5c1a00;">Moderada</div>
                        <div style="width:15%; background:#d73027; color:#fff;">Mod-Alta</div>
                        <div style="width:14%; background:#7a0403; color:#fff;">Alta</div>
                    </div>
                    <div class="dnbr-ticks">
                        <span>≪ −100</span>
                        <span>−100</span>
                        <span>+100</span>
                        <span>+270</span>
                        <span>+440</span>
                        <span>+660</span>
                        <span>≫ 660</span>
                    </div>

                    <div class="dnbr-grid">
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#1a9850;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#1a9850;">🌱 Regeneração</p>
                                <p class="dnbr-range">dNBR &lt; −100</p>
                                <p class="dnbr-desc">Vegetação cresceu após incêndio anterior. NBR aumentou — brotos novos refletem mais NIR.</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#91cf60;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#5a8a20;">🌿 Não afetado</p>
                                <p class="dnbr-range">−100 a +100</p>
                                <p class="dnbr-desc">Vegetação intacta ou variação sazonal normal. Fogo não atingiu essa área.</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#fee08b;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#a07000;">🟡 Baixa severidade</p>
                                <p class="dnbr-range">+100 a +270</p>
                                <p class="dnbr-desc">Queima superficial de sub-bosque. Dossel parcialmente afetado, recuperação em meses.</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#fc8d59;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#c04010;">🟠 Moderada</p>
                                <p class="dnbr-range">+270 a +440</p>
                                <p class="dnbr-desc">Danos significativos ao dossel. Solo parcialmente exposto, recuperação lenta (1–3 anos).</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#d73027;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#d73027;">🔴 Moderada-Alta</p>
                                <p class="dnbr-range">+440 a +660</p>
                                <p class="dnbr-desc">Destruição extensa do dossel. Solo exposto, cinzas visíveis. Recuperação de 3–5 anos.</p>
                            </div>
                        </div>
                        <div class="dnbr-card">
                            <div class="dnbr-stripe" style="background:#7a0403;"></div>
                            <div class="dnbr-body">
                                <p class="dnbr-name" style="color:#7a0403;">⬛ Alta severidade</p>
                                <p class="dnbr-range">dNBR &gt; +660</p>
                                <p class="dnbr-desc">Destruição total da cobertura. Solo nu, carvão, cinzas. Risco de erosão alto.</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                col_nbr1, col_nbr2 = st.columns([1.4, 1])

                with col_nbr1:
                    with st.spinner("🛰️ Processando imagens Sentinel-2..."):
                        nbr_ok = False
                        stats_sev = {}
                        dnbr_img = None

                        # Aviso antecipado para Amazônia (> 1,5 M km²)
                        if area_km2_local > 1_500_000:
                            st.info(
                                "🌿 **Amazônia detectada:** o cálculo usa resolução de 1 km "
                                "para evitar timeout no GEE. Os gráficos de severidade "
                                "são estatisticamente representativos mesmo nessa escala."
                            )

                        _debug_nbr = st.toggle(
                            "🐛 Mostrar log de depuração do dNBR",
                            value=False,
                            key="toggle_debug_nbr"
                        )

                        def _log(msg):
                            if _debug_nbr:
                                st.caption(f"⚙️ {msg}")

                        try:
                            _log(f"Área local estimada: {area_km2_local:,.0f} km²")
                            _log(f"Parâmetros: ano={ano_modis}, mês={mes_modis}")
                            _log(f"Máscara MODIS disponível: {area_queimada_img is not None}")

                            # 1) Stats cacheados
                            _log("Passo 1/2 — chamando calcular_stats_nbr()...")
                            stats_sev = calcular_stats_nbr(
                                geom_json_str, ano_modis, mes_modis,
                                area_km2_hint=area_km2_local,
                                _mascara_modis=area_queimada_img
                            )
                            _log(f"Stats retornados: {stats_sev}")

                            # 2) Imagem GEE reconstruída
                            _log("Passo 2/2 — chamando _construir_dnbr()...")
                            _, dnbr_img = _construir_dnbr(
                                geom_json_str, ano_modis, mes_modis,
                                area_km2_hint=area_km2_local,
                                _mascara_modis=area_queimada_img
                            )
                            _log("dNBR construído com sucesso.")
                            nbr_ok = True

                        except ValueError as ve:
                            st.warning(f"⚠️ {ve}")
                            if _debug_nbr:
                                st.code(traceback.format_exc(), language="python")
                        except Exception as e:
                            st.error(f"⚠️ Erro ao processar Sentinel-2: **{type(e).__name__}**: {e}")
                            if _debug_nbr:
                                st.code(traceback.format_exc(), language="python")

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
                            
                        except Exception as erro_mapa:
                            st.error(f"⚠️ Erro ao desenhar o mapa dNBR: **{type(erro_mapa).__name__}**: {erro_mapa}")
                            if _debug_nbr:
                                st.code(traceback.format_exc(), language="python")

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
                        st.caption("💡 Abra o guia de interpretação acima para detalhes de cada classe.")

        # ----------------------------------------------------------
        # ABA 4 — CUSTO DE COMBATE
        # ----------------------------------------------------------
        with aba_custos:
            st.subheader("💰 Estimativa de Custo de Combate ao Incêndio")
            st.markdown(
                "Estimativa baseada nos **custos médios operacionais do IBAMA** "
                "(Relatório de Operações de Combate a Incêndios Florestais). "
                "Os valores refletem operações aéreas e terrestres em biomas brasileiros."
            )

            # ── Tabela de referência de custos (fonte: IBAMA / literatura técnica) ────────
            # Valores em R$/hectare. Fontes:
            #   • IBAMA (2022): Relatório de Operações PrevFogo — custo médio nacional R$ 40–80/ha
            #   • Soares & Batista (2007): custos diferenciados por bioma
            #   • Ajuste por inflação IPCA até 2024 (~35% sobre base 2020)
            CUSTOS_POR_BIOMA = {
                "Amazônia":          {"aereo": 142, "terrestre": 48,  "total": 190},
                "Cerrado":           {"aereo": 98,  "terrestre": 35,  "total": 133},
                "Pantanal":          {"aereo": 165, "terrestre": 52,  "total": 217},
                "Mata Atlântica":    {"aereo": 120, "terrestre": 42,  "total": 162},
                "Caatinga":          {"aereo": 75,  "terrestre": 30,  "total": 105},
                "Pampa":             {"aereo": 68,  "terrestre": 28,  "total":  96},
            }
            CUSTO_PADRAO = {"aereo": 110, "terrestre": 38, "total": 148}

            # ── Detectar bioma da seleção atual ────────────────────────────────────────────
            bioma_detectado = None
            for nome_bioma in CUSTOS_POR_BIOMA:
                if nome_bioma.lower() in val_sel.lower():
                    bioma_detectado = nome_bioma
                    break
            custos = CUSTOS_POR_BIOMA.get(bioma_detectado, CUSTO_PADRAO)

            # ── Área queimada disponível ────────────────────────────────────────────────────
            area_ha = 0.0
            if "MODIS" in fonte_escolhida and total_valor > 0:
                area_ha = total_valor * 100   # total_valor já está em km²  → × 100 = ha
            elif "INPE" in fonte_escolhida and total_valor > 0:
                # INPE retorna focos, não área — estimativa: ~15 ha médios por foco (literatura)
                area_ha = total_valor * 15

            # ── Controles do usuário ────────────────────────────────────────────────────────
            st.markdown("---")
            col_c1, col_c2 = st.columns([1.2, 1])

            with col_c1:
                st.markdown("#### ⚙️ Parâmetros da estimativa")

                # Slider para ajuste do custo
                custo_ajustado = st.slider(
                    "💲 Custo total estimado por hectare (R$/ha)",
                    min_value=30, max_value=500,
                    value=custos["total"],
                    step=5,
                    help=(
                        f"Valor padrão para **{bioma_detectado or 'seleção atual'}**: "
                        f"R$ {custos['total']}/ha  "
                        f"(aéreo R$ {custos['aereo']}/ha + terrestre R$ {custos['terrestre']}/ha). "
                        "Ajuste conforme o tipo de operação real."
                    )
                )

                # Área editável
                area_editavel = st.number_input(
                    "🔥 Área afetada (hectares)",
                    min_value=0.0,
                    value=float(max(area_ha, 0.0)),
                    step=100.0,
                    format="%.0f",
                    help=(
                        "Preenchido automaticamente com os dados da análise. "
                        "Você pode ajustar manualmente."
                    )
                )

                # Fator de mobilização (distância/logística)
                fator_logistica = st.select_slider(
                    "🚁 Fator de dificuldade logística",
                    options=[1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0],
                    value=1.5,
                    help=(
                        "1.0 = região de fácil acesso (rodovia próxima). "
                        "1.5 = acesso moderado (voos curtos, estradas vicinais). "
                        "3.0 = zona remota (floresta fechada, sem pistas, voos longos)."
                    )
                )

            with col_c2:
                st.markdown("#### 📋 Referência de custos por bioma (IBAMA)")
                df_ref = pd.DataFrame([
                    {"Bioma": k, "Aéreo (R$/ha)": v["aereo"],
                     "Terrestre (R$/ha)": v["terrestre"], "Total (R$/ha)": v["total"]}
                    for k, v in CUSTOS_POR_BIOMA.items()
                ])
                # Highlight da linha do bioma detectado
                def _highlight_bioma(row):
                    cor = "background-color: rgba(255,100,50,0.18);" if row["Bioma"] == bioma_detectado else ""
                    return [cor] * len(row)
                st.dataframe(
                    df_ref.style.apply(_highlight_bioma, axis=1),
                    hide_index=True,
                    use_container_width=True,
                    height=250,
                )
                st.caption("🔴 Linha destacada = bioma identificado na seleção atual")

            st.markdown("---")

            # ── Cálculo ────────────────────────────────────────────────────────────────────
            custo_base      = area_editavel * custo_ajustado
            custo_total     = custo_base * fator_logistica
            custo_aereo_est = area_editavel * custos["aereo"]  * fator_logistica
            custo_terr_est  = area_editavel * custos["terrestre"] * fator_logistica

            # ── KPI Cards ─────────────────────────────────────────────────────────────────
            st.markdown("#### 📊 Resultado da Estimativa")
            k1, k2, k3, k4 = st.columns(4)

            def _kpi(col, emoji, titulo, valor, sub=""):
                with col:
                    st.metric(
                        label=f"{emoji} {titulo}",
                        value=valor,
                        delta=sub if sub else None,
                        delta_color="off"
                    )

            _kpi(k1, "🔥", "Área afetada",
                 f"{area_editavel:,.0f} ha",
                 f"≈ {area_editavel/100:,.0f} km²")
            _kpi(k2, "💲", "Custo base",
                 f"R$ {custo_base:,.0f}",
                 f"R$ {custo_ajustado}/ha × área")
            _kpi(k3, "🚁", "Fator logística",
                 f"×{fator_logistica}",
                 "dificuldade de acesso")
            _kpi(k4, "💰", "Custo total estimado",
                 f"R$ {custo_total:,.0f}",
                 f"≈ R$ {custo_total/1e6:.2f} milhões" if custo_total >= 1e6 else "")

            # ── Gráfico de decomposição ────────────────────────────────────────────────────
            st.markdown("---")
            col_g1, col_g2 = st.columns([1.3, 1])

            with col_g1:
                df_pizza = pd.DataFrame({
                    "Componente": ["Combate aéreo", "Combate terrestre", "Overhead logístico"],
                    "Valor (R$)": [
                        custo_aereo_est,
                        custo_terr_est,
                        custo_total - custo_aereo_est - custo_terr_est
                    ]
                }).query("`Valor (R$)` > 0")

                fig_pizza = px.pie(
                    df_pizza,
                    names="Componente", values="Valor (R$)",
                    title="Decomposição do Custo de Combate",
                    color_discrete_sequence=["#e74c3c", "#e67e22", "#f39c12"],
                    hole=0.42
                )
                fig_pizza.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>R$ %{value:,.0f}<extra></extra>"
                )
                fig_pizza.update_layout(
                    template="plotly_dark",
                    height=340,
                    margin=dict(t=50, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_pizza, use_container_width=True)

            with col_g2:
                st.markdown("#### 📌 Contexto & benchmarks")
                st.markdown(f"""
| Indicador | Valor |
|-----------|-------|
| 🌍 Área afetada | **{area_editavel:,.0f} ha** |
| 💲 Custo/ha | **R$ {custo_ajustado}/ha** |
| 🚁 Fator logística | **{fator_logistica}×** |
| 🧾 **Custo total** | **R$ {custo_total:,.0f}** |
| 🏛️ Equivalente em escolas | **≈ {int(custo_total / 1_200_000):,} salas** |
| 🏥 Equiv. em UBS construídas | **≈ {int(custo_total / 800_000):,} unidades** |
                """)
                st.caption(
                    "Benchmarks: custo médio de construção de sala de aula escolar "
                    "R$ 1,2M (FNDE 2023); UBS padrão R$ 800K (MS 2023)."
                )

            # ── Aviso metodológico ─────────────────────────────────────────────────────────
            with st.expander("⚠️ Metodologia e limitações desta estimativa", expanded=False):
                st.markdown("""
**Fontes dos custos de referência:**
- IBAMA / PrevFogo — *Relatório Anual de Operações de Combate a Incêndios Florestais (2022)*
- Soares, R.V. & Batista, A.C. — *Incêndios Florestais: controle, efeitos e uso do fogo* (2007)
- Atualização monetária via IPCA acumulado até 2024

**O que está incluído no custo:**
- Horas de voo de helicópteros e aviões de combate (Air Tractor, Ipanema)
- Diárias e deslocamento de brigadistas
- Insumos diretos (retardante, bombas d'água)
- Overhead operacional (logística, comunicação, coordenação)

**O que NÃO está incluído:**
- Dano ambiental (carbono emitido, biodiversidade perdida)
- Custo de recuperação/reflorestamento pós-incêndio
- Perdas agrícolas e pecuárias
- Dano à saúde pública (fumaça, doenças respiratórias)
- Custo para o produtor rural

**Para uso acadêmico/MBA:** valores são estimativas de ordem de grandeza. Para análise de custo-benefício precisa, consulte os relatórios PrevFogo do IBAMA e os dados do SINAFLOR.
                """)

        # ----------------------------------------------------------
        # ABA 5 — ESG & CAR
        # ----------------------------------------------------------
        with aba_esg:
            st.subheader("🌿 Dashboard ESG — Risco de Queimada & Rastreabilidade CAR")
            st.markdown(
                "Ferramenta de apoio à decisão para **instituições financeiras, seguradoras e "
                "investidores ESG** do agronegócio. Cruza dados de satélite com o "
                "**Cadastro Ambiental Rural (SICAR)** para gerar um score de risco por região."
            )

            # ── Detectar bioma da seleção ──────────────────────────────────────────────
            _BIOMAS_CONHECIDOS = [
                "Amazônia", "Cerrado", "Pantanal",
                "Mata Atlântica", "Caatinga", "Pampa"
            ]
            bioma_esg = next(
                (b for b in _BIOMAS_CONHECIDOS if b.lower() in val_sel.lower()),
                "Cerrado"   # fallback mais comum no agro
            )

            # ── Busca SICAR ────────────────────────────────────────────────────────────
            with st.spinner("🔗 Consultando SICAR..."):
                dados_sicar = consultar_sicar(estado_dd)

            # ── Anomalia (pode já ter sido calculada na aba de gráficos) ──────────────
            df_anom_esg = pd.DataFrame()
            if "MODIS" in fonte_escolhida and not df_modis_temporal.empty and ano_modis > 2001:
                try:
                    df_anom_esg = calcular_anomalia_modis(geom_json_str, ano_modis)
                except Exception:
                    pass

            # ── Calcular score ESG ─────────────────────────────────────────────────────
            esg = calcular_score_esg(
                total_valor, area_km2_local,
                fonte_escolhida, bioma_esg, df_anom_esg
            )

            st.markdown("---")

            # ── Linha 1: KPIs do SICAR + Score ────────────────────────────────────────
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
            with c1:
                st.metric(
                    "🏘️ Imóveis no CAR", f"{dados_sicar['imoveis']:,}",
                    help=f"Estado {estado_dd} · {dados_sicar['fonte']}"
                )
            with c2:
                st.metric(
                    "🌾 Área cadastrada", f"{dados_sicar['area_mha']:.1f} Mha",
                    help="Área total registrada no CAR para o estado selecionado"
                )
            with c3:
                area_afetada_ha = total_valor * 100 if "MODIS" in fonte_escolhida else total_valor * 15
                pct_afetada = (area_afetada_ha / (dados_sicar["area_mha"] * 1e6) * 100) if dados_sicar["area_mha"] > 0 else 0
                st.metric(
                    "🔥 Área afetada/CAR",
                    f"{pct_afetada:.2f}%",
                    help="% da área CAR cadastrada potencialmente afetada por fogo no período"
                )
            with c4:
                st.markdown(
                    f"""
                    <div style="border:1px solid {esg['cor']}55; border-radius:10px;
                                padding:12px 16px; background:{esg['cor']}14; text-align:center;">
                        <div style="font-size:12px; color:#aaa; margin-bottom:2px;">
                            ESG Fire Risk Score
                        </div>
                        <div style="font-size:36px; font-weight:800; color:{esg['cor']}; line-height:1;">
                            {esg['score_final']}
                        </div>
                        <div style="font-size:13px; font-weight:600; color:{esg['cor']}; margin-top:4px;">
                            ● Risco {esg['tier']}
                        </div>
                        <div style="font-size:11px; color:#888; margin-top:4px;">
                            escala 0 (sem risco) → 100 (crítico)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("---")

            # ── Linha 2: Gauge + Radar + Tabela de pilares ────────────────────────────
            col_gauge, col_radar = st.columns([1, 1.1])

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=esg["score_final"],
                    number={"suffix": " / 100", "font": {"size": 28}},
                    delta={"reference": 50, "increasing": {"color": "#e74c3c"},
                           "decreasing": {"color": "#27ae60"}},
                    title={"text": f"ESG Fire Risk — {val_sel}", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1,
                                 "tickcolor": "#555", "tickvals": [0,25,50,75,100]},
                        "bar": {"color": esg["cor"], "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 25],  "color": "#27ae6022"},
                            {"range": [25, 50], "color": "#f39c1222"},
                            {"range": [50, 75], "color": "#e67e2222"},
                            {"range": [75, 100],"color": "#e74c3c22"},
                        ],
                        "threshold": {
                            "line": {"color": esg["cor"], "width": 3},
                            "thickness": 0.85,
                            "value": esg["score_final"]
                        }
                    }
                ))
                fig_gauge.update_layout(
                    template="plotly_dark", height=300,
                    margin=dict(t=50, b=10, l=20, r=20)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_radar:
                fig_radar = go.Figure(go.Scatterpolar(
                    r=[esg["score_exp"], esg["score_rec"], esg["score_sens"],
                       esg["score_exp"]],
                    theta=["Exposição ao Fogo", "Recorrência Histórica",
                           "Sensibilidade do Bioma", "Exposição ao Fogo"],
                    fill="toself",
                    fillcolor=esg["cor"] + "33",
                    line=dict(color=esg["cor"], width=2),
                    name="Score ESG"
                ))
                fig_radar.update_layout(
                    template="plotly_dark",
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100],
                                        tickfont=dict(size=9)),
                        angularaxis=dict(tickfont=dict(size=11))
                    ),
                    height=300,
                    margin=dict(t=40, b=20, l=40, r=40),
                    title=dict(text="Decomposição do Risco", font=dict(size=13))
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ── Tabela dos 3 pilares ───────────────────────────────────────────────────
            df_pilares = pd.DataFrame([
                {
                    "Pilar": "🔥 Exposição ao Fogo",
                    "Peso": "50%",
                    "Score (0–100)": esg["score_exp"],
                    "Descrição": (
                        f"{'Área queimada' if 'MODIS' in fonte_escolhida else 'Focos INPE'} "
                        f"vs benchmark do {bioma_esg}"
                    )
                },
                {
                    "Pilar": "📅 Recorrência Histórica",
                    "Peso": "30%",
                    "Score (0–100)": esg["score_rec"],
                    "Descrição": (
                        f"{int((df_anom_esg['Anomalia (%)'] > 30).sum())} meses acima "
                        f"de +30% da média histórica"
                        if not df_anom_esg.empty else "Dado indisponível (MODIS necessário)"
                    )
                },
                {
                    "Pilar": "🌿 Sensibilidade do Bioma",
                    "Peso": "20%",
                    "Score (0–100)": esg["score_sens"],
                    "Descrição": f"{bioma_esg} — fragilidade ecológica (IUCN/WWF)"
                },
                {
                    "Pilar": "⚖️ Score Final Ponderado",
                    "Peso": "100%",
                    "Score (0–100)": esg["score_final"],
                    "Descrição": f"Risco {esg['tier']} — benchmarks INPE/MapBiomas/IUCN"
                },
            ])

            def _color_score(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val < 25:   return "background-color:#27ae6025;"
                if val < 50:   return "background-color:#f39c1225;"
                if val < 75:   return "background-color:#e67e2225;"
                return "background-color:#e74c3c25;"

            st.dataframe(
                df_pilares.style.applymap(
                    _color_score, subset=["Score (0–100)"]
                ),
                hide_index=True, use_container_width=True
            )

            st.markdown("---")

            # ── Ranking de municípios por risco (se disponível) ───────────────────────
            if not df_top_mun_modis.empty and tipo_analise != "Por Município":
                st.markdown("#### 🏙️ Municípios com maior exposição a fogo")
                st.caption(
                    "Base para decisões de crédito rural, seguro agrícola e due diligence ESG. "
                    "Municípios com alta concentração de queimadas representam maior risco de "
                    "inadimplência climática e passivo ambiental."
                )

                df_mun_esg = df_top_mun_modis.copy()
                df_mun_esg.columns = ["Município", "Área Queimada (km²)"]
                df_mun_esg["Risco Relativo"] = (
                    df_mun_esg["Área Queimada (km²)"] /
                    df_mun_esg["Área Queimada (km²)"].max() * 100
                ).round(1)
                df_mun_esg["Classificação ESG"] = df_mun_esg["Risco Relativo"].apply(
                    lambda v: "🔴 Crítico" if v > 75
                    else ("🟠 Alto" if v > 50
                    else ("🟡 Moderado" if v > 25
                    else "🟢 Baixo"))
                )
                df_mun_esg["Ação Recomendada"] = df_mun_esg["Classificação ESG"].map({
                    "🔴 Crítico":  "Suspender crédito / acionar CAR",
                    "🟠 Alto":     "Revisão de apólice / due diligence",
                    "🟡 Moderado": "Monitoramento trimestral",
                    "🟢 Baixo":    "Manter — baixo risco sistêmico",
                })

                fig_mun = px.bar(
                    df_mun_esg, x="Área Queimada (km²)", y="Município",
                    orientation="h", color="Risco Relativo",
                    color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                    title="Ranking de Exposição ao Fogo por Município",
                    text="Área Queimada (km²)"
                )
                fig_mun.update_layout(
                    template="plotly_dark", height=320,
                    margin=dict(t=40, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                fig_mun.update_traces(texttemplate="%{text:.0f} km²", textposition="outside")
                st.plotly_chart(fig_mun, use_container_width=True)

                st.dataframe(
                    df_mun_esg[["Município","Área Queimada (km²)","Classificação ESG","Ação Recomendada"]],
                    hide_index=True, use_container_width=True
                )

                # Export ESG
                csv_esg = df_mun_esg.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ Exportar relatório ESG (.csv)",
                    csv_esg,
                    file_name=f"esg_risco_fogo_{val_sel.replace(' ','_')}_{ano_modis}.csv",
                    mime="text/csv"
                )

            elif tipo_analise == "Por Município":
                st.info(
                    "ℹ️ Para análise por município, o score acima já representa "
                    "o risco específico da propriedade selecionada. "
                    "Para comparar municípios, selecione análise **Por Estado** ou **Por Bioma**."
                )

            # ── Metodologia ───────────────────────────────────────────────────────────
            with st.expander("📐 Metodologia do ESG Fire Risk Score", expanded=False):
                st.markdown(f"""
**Fórmula:**
```
Score Final = 0,50 × Exposição + 0,30 × Recorrência + 0,20 × Sensibilidade
```

| Pilar | Fonte | Cálculo |
|-------|-------|---------|
| Exposição ao Fogo | MODIS MCD64A1 / INPE BDQueimadas | % área queimada ÷ benchmark do bioma × 25 |
| Recorrência | MODIS histórico 2001–presente | Meses com anomalia > +30% × 12,5 |
| Sensibilidade | IUCN Red List of Ecosystems / WWF | Score fixo por bioma (Mata Atlântica=90, Pampa=38) |

**Benchmarks de referência (área queimada/ano normal):**
Amazônia 0,3% · Cerrado 2,5% · Pantanal 1,5% · Mata Atlântica 0,1% · Caatinga 1,0% · Pampa 0,2%

**Fonte CAR:** {dados_sicar['fonte']}

**Escala de risco:**
- 🟢 **Verde (0–25):** Risco baixo — operações normais de crédito e seguro
- 🟡 **Amarelo (25–50):** Risco moderado — monitoramento recomendado
- 🟠 **Laranja (50–75):** Risco alto — revisão de apólices e garantias
- 🔴 **Vermelho (75–100):** Risco crítico — due diligence obrigatória, verificar conformidade CAR

**Fontes:** INPE BDQueimadas · MODIS MCD64A1 · SICAR/MAPA · MapBiomas Fire Collection · IUCN · WWF Living Planet
                """)

        # ----------------------------------------------------------
        # ABA 6 — EXPORTAR DADOS
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

else:
    st.info(
        "👈 Use os filtros ao lado para selecionar a Fonte de Dados, "
        "o local e o período de análise. Depois clique em **'Gerar Dashboard'**."
    )
