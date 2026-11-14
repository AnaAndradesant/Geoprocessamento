import streamlit as st
import pandas as pd
import geopandas as gpd
import shapely
from datetime import datetime, timedelta
from geobr import read_state, read_biomes, read_municipality, read_indigenous_land, read_conservation_units
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import requests, warnings, time, unicodedata, re, json, io
import ee
from io import BytesIO
import traceback
import numpy as np
import concurrent.futures
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

# LightGBM é o modelo principal; se não estiver instalado (ex: ainda não foi
# adicionado ao requirements.txt), cai para HistGradientBoostingClassifier
# (já vem no sklearn, zero dependência nova) sem quebrar o app.
try:
    from lightgbm import LGBMClassifier
    _LGBM_DISPONIVEL = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _LGBM_DISPONIVEL = False


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


def _ocultar_url_base(mapa):
    # O mapa de fundo criado por folium.Map(tiles=URL) aparecia no controle de
    # camadas com o próprio endereço da URL como nome; tira só ele do controle.
    for filho in mapa._children.values():
        if isinstance(filho, folium.raster_layers.TileLayer) and not filho.overlay:
            filho.control = False

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Monitor de Queimadas Brasil",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# PADRÃO DE FORMATAÇÃO DO BUSINESS CASE (Manual USP/Esalq, itens 17.1 a 17.2)
# Arial, texto preto e justificado; gráficos sem título, grade, borda ou
# preenchimento; eixos em linha preta sólida; legenda abaixo da figura e
# título acima da tabela, sempre com numeração sequencial e Fonte.
# =====================================================================
import plotly.io as pio

pio.templates["business_case"] = go.layout.Template(
    layout=dict(
        font=dict(family="Arial", size=11, color="black"),
        title=dict(text=""),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=20, r=20),
        xaxis=dict(showgrid=False, zeroline=False, showline=True,
                   linecolor="black", linewidth=1.5, mirror=False,
                   title=dict(font=dict(family="Arial", size=11, color="black"))),
        yaxis=dict(showgrid=False, zeroline=False, showline=True,
                   linecolor="black", linewidth=1.5, mirror=False,
                   title=dict(font=dict(family="Arial", size=11, color="black"))),
    )
)
pio.templates.default = "plotly+business_case"

st.markdown("""
<style>
html, body, [class*="css"], .stMarkdown, .stDataFrame, .stCaption {
    font-family: Arial, Helvetica, sans-serif !important;
}
.stMarkdown p, .stMarkdown li { text-align: justify; }
.bc-legenda { font-family: Arial, sans-serif; font-size: 11pt; text-align: justify;
              margin: 2px 0 14px 0; line-height: 1.2; }
.bc-titulo-tabela { font-family: Arial, sans-serif; font-size: 11pt; text-align: justify;
                    margin: 14px 0 2px 0; line-height: 1.2; }
</style>
""", unsafe_allow_html=True)

# Numeração pela ordem da primeira exibição; o registro por texto mantém o número
# estável quando só um fragmento (aba CAR/UC) reexecuta.
def _numero(chave, texto):
    reg = st.session_state.setdefault(chave, {})
    if texto not in reg:
        reg[texto] = len(reg) + 1
    return reg[texto]


def figura(fig, legenda, fonte="Resultados originais do estudo", nota=None, key=None):
    """Exibe gráfico Plotly com 'Figura N. legenda' e 'Fonte:' abaixo (sem ponto final)."""
    n = _numero("_reg_fig", legenda)
    fig.update_layout(title_text="")
    # theme=None: sem isso o tema do Streamlit (escuro) substitui o padrão do manual
    st.plotly_chart(fig, use_container_width=True, key=key, theme=None)
    html = f"<div class='bc-legenda'>Figura {n}. {legenda}<br>Fonte: {fonte}"
    if nota:
        html += f"<br>Nota: {nota}"
    st.markdown(html + "</div>", unsafe_allow_html=True)


def tabela_bc(titulo, df, fonte="Resultados originais do estudo", nota=None, **kwargs):
    """Exibe DataFrame com 'Tabela N. título' acima e 'Fonte:' abaixo."""
    n = _numero("_reg_tab", titulo)
    st.markdown(f"<div class='bc-titulo-tabela'>Tabela {n}. {titulo}</div>", unsafe_allow_html=True)
    st.dataframe(df, **kwargs)
    html = f"<div class='bc-legenda'>Fonte: {fonte}"
    if nota:
        html += f"<br>Nota: {nota}"
    st.markdown(html + "</div>", unsafe_allow_html=True)


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
def _cidades_cache(uf):
    # Só lista válida entra no cache: falha levanta exceção e não fica gravada
    # por 24 h (antes um timeout isolado deixava "Erro ao carregar cidades" fixo).
    # Fonte principal: geobr (mesma base usada para desenhar o limite do município).
    # A API do IBGE pode não responder a partir do Streamlit Cloud, então fica só
    # como plano B, com uma tentativa curta para não travar a barra lateral.
    try:
        gdf = read_municipality(code_muni=uf, year=2020)
        nomes = sorted(gdf["name_muni"].dropna().unique().tolist())
        if nomes:
            return nomes
    except Exception:
        pass
    url = f"https://servicodados.ibge.gov.br/api/v1/localidades/estados/{uf}/municipios"
    resp = requests.get(url, timeout=6)
    resp.raise_for_status()
    nomes = sorted(d['nome'] for d in resp.json())
    if not nomes:
        raise RuntimeError(f"Nenhum município encontrado para {uf}")
    return nomes


def buscar_cidades(uf):
    try:
        return _cidades_cache(uf)
    except Exception:
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
    # O geobr muda de assinatura/colunas entre versões; medido no geobr 2.1.1 (o do
    # Streamlit Cloud): read_indigenous_land(year=2025) → coluna name_indigenous_land;
    # read_conservation_units(date=202503) → colunas nome_uc, grupo e categoria.
    # Tenta do mais recente ao mais antigo e aceita os nomes de coluna das versões
    # anteriores também. (Antes passava year=202501 / esperava name_conservation_unit,
    # o que falhava nessa versão.)
    ultimo_erro = None
    gdf_areas = None
    if tipo_area == "Terras Indígenas":
        for ano_ref in (2025, 2024, 2023, 2022, 2021, 2020, 2019):
            try:
                gdf_areas = read_indigenous_land(year=ano_ref)
                break
            except Exception as e:
                ultimo_erro = e
        if gdf_areas is None:
            raise RuntimeError(f"Não foi possível baixar Terras Indígenas do geobr. Último erro: {ultimo_erro}")
        nomes = ("name_indigenous_land", "terrai_nom")
    else:
        for data_ref in (202503, 202402, 202112, 202001, 201909):
            try:
                gdf_areas = read_conservation_units(date=data_ref)
                break
            except Exception as e:
                ultimo_erro = e
        if gdf_areas is None:
            raise RuntimeError(f"Não foi possível baixar Unidades de Conservação do geobr. Último erro: {ultimo_erro}")
        nomes = ("nome_uc", "name_conservation_unit")
    col_nome = next((c for c in nomes if c in gdf_areas.columns), None)
    if col_nome is None:
        raise RuntimeError(f"Coluna com o nome da área não encontrada (colunas: {list(gdf_areas.columns)})")
    gdf_areas = gdf_areas.rename(columns={col_nome: "nome_area"})
    gdf_areas['geometry'] = gdf_areas['geometry'].make_valid()
    gdf_areas = gdf_areas.to_crs("EPSG:4326")
    gdf_areas['geometry'] = gdf_areas['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    extras = [c for c in ("grupo", "categoria") if c in gdf_areas.columns]
    return gdf_areas[['nome_area'] + extras + ['geometry']]


import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.util.ssl_ import create_urllib3_context

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

    # Colunas usadas pelo dashboard (mapa, gráficos, ranking e CSV exportado).
    colunas = ["longitude", "latitude", "data_hora_gmt", "satelite", "municipio", "estado",
               "bioma", "risco_fogo", "precipitacao", "numero_dias_sem_chuva", "frp"]

    def _buscar_bloco(bloco):
        b_ini, b_fim = bloco
        cql = (
            f"data_hora_gmt >= '{b_ini.strftime('%Y-%m-%d')}T00:00:00' "
            f"AND data_hora_gmt <= '{b_fim.strftime('%Y-%m-%d')}T23:59:59' "
            f"AND satelite IN ('{sat_str}') AND {filtro_base}"
        )
        # CSV só com as colunas usadas, em vez do GeoJSON completo (~30 colunas, um
        # dict Python por foco): medido, ~10x menos bytes e muito menos memória. Num
        # bioma inteiro (Amazônia) o JSON, com 12 downloads simultâneos, deixava o
        # app a ~1,1 GB só no carregamento — e ele caía ao desenhar as abas.
        for tentativa in range(3):
            try:
                r = session.get(
                    url,
                    params={
                        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                        "typeName": "bdqueimadas:focos", "outputFormat": "csv",
                        "propertyName": ",".join(colunas),
                        "CQL_FILTER": cql, "maxFeatures": 50000
                    },
                    headers=headers, verify=False, timeout=90
                )
                if r.status_code == 200 and "csv" in r.headers.get("content-type", ""):
                    bloco_df = pd.read_csv(io.BytesIO(r.content), usecols=lambda c: c in colunas)
                    for c in ("longitude", "latitude", "risco_fogo", "precipitacao", "frp"):
                        if c in bloco_df.columns:
                            bloco_df[c] = bloco_df[c].astype("float32")
                    return bloco_df
            except Exception:
                pass
            time.sleep(1.5 * (tentativa + 1))
        return None

    # Chamadas em paralelo, mas poucas: cada resposta fica na memória até ser
    # concatenada.
    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for bloco_df in executor.map(_buscar_bloco, blocos):
            if bloco_df is not None and not bloco_df.empty:
                all_dfs.append(bloco_df)

    if not all_dfs:
        return pd.DataFrame()

    df_final = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    df_final = df_final.drop_duplicates()
    for c in ("satelite", "estado", "bioma", "municipio"):
        if c in df_final.columns:
            df_final[c] = df_final[c].astype("category")
    return df_final


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


# =====================================================================
# 🎯 AMOSTRAGEM CASO-CONTROLE PAREADA POR LOCAL
# =====================================================================
# Limitação do modo "painel por município": treina com uma amostra pequena de
# municípios (3-20) num bioma que pode ter centenas. Cobertura espacial baixa.
#
# Aqui a unidade de amostra deixa de ser o município-dia e passa a ser o
# FOCO-dia. Cada foco real vira um caso POSITIVO; o MESMO local, numa data
# anterior sem ocorrência, vira o CONTROLE (negativo). A pergunta que o
# modelo aprende muda de "onde pega fogo?" para "neste local que já é
# vulnerável, o que diferencia um dia de risco de um dia comum?".
#
# Vantagem sobre background aleatório: o controle é o próprio local, então
# não há risco de o modelo aprender só "tem vegetação vs. não tem" (um ponto
# aleatório poderia cair em rio ou área urbana).
# Limitação a declarar: cobre bem locais COM histórico de fogo; município que
# nunca teve foco no período não entra no treino (cai no fallback da taxa geral).

@st.cache_data(ttl=3600, show_spinner=False, persist="disk")
def buscar_focos_com_coords(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim, dias_bloco=7, satelites=None):
    """
    Como buscar_historico_focos_diario, mas traz lat/lon ALÉM da data -- é o
    que a amostragem caso-controle precisa (cada foco é um ponto no espaço,
    não só uma contagem diária).
    """
    url = "https://terrabrasilis.dpi.inpe.br/queimadas/geoserver/bdqueimadas/ows"
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries, pool_maxsize=20))
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    dic_estados = {
        "AC": "ACRE", "AL": "ALAGOAS", "AP": "AMAP_", "AM": "AMAZONAS",
        "BA": "BAHIA", "CE": "CEAR_", "DF": "DISTRITO FEDERAL",
        "ES": "ESP%RITO SANTO", "GO": "GOI%S", "MA": "MARANH%O",
        "MT": "MATO GROSSO", "MS": "MATO GROSSO DO SUL", "MG": "MINAS GERAIS",
        "PA": "PAR_", "PB": "PARA_BA", "PR": "PARAN_", "PE": "PERNAMBUCO",
        "PI": "PIAU_", "RJ": "RIO DE JANEIRO", "RN": "RIO GRANDE DO NORTE",
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
        # Cada vogal/acentuada vira "_" (UM caractere qualquer), não "%" (qualquer
        # tamanho): o nome do município no INPE tem acento ("CORUMBÁ"), então a vogal
        # precisa de curinga, mas com "%" e um "%" no fim "Jataí" virava J%T%% e o
        # servidor devolvia vários municípios de nome parecido (depois recortados),
        # o que deixava cada requisição lenta. Com "_" casa só o tamanho certo.
        muni_curinga = "".join(
            "_" if (c.lower() in "aeiou" or ord(c) > 127) else ("%" if c == " " else c)
            for c in val_muni
        )
        filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}' AND municipio ILIKE '{muni_curinga}'"

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
        if satelites:
            cql += " AND satelite IN ('" + "','".join(satelites) + "')"
        # Até 3 tentativas: o servidor do INPE às vezes devolve uma resposta que não é
        # JSON (medido). Antes isso virava "0 focos nesse bloco" em silêncio, o que
        # subestimaria a contagem — agora a falha definitiva é contada e avisada.
        # CSV com só as 4 colunas usadas, em vez do GeoJSON completo (~30 colunas):
        # medido, 95 KB contra 995 KB para os mesmos focos. Com 10 blocos baixados
        # em paralelo, o JSON completo de um estado da Amazônia na seca passava de
        # 1 GB de memória e derrubava o app. lat/lon vêm dos atributos
        # longitude/latitude, então não precisa da geometria.
        # Até 3 tentativas: o servidor do INPE às vezes devolve uma resposta que não
        # é o CSV (medido). Antes isso virava "0 focos nesse bloco" em silêncio, o
        # que subestimaria a contagem — agora a falha definitiva é contada e avisada.
        for tentativa in range(3):
            try:
                r = session.get(
                    url,
                    params={
                        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
                        "typeName": "bdqueimadas:focos", "outputFormat": "csv",
                        "propertyName": "longitude,latitude,data_hora_gmt,municipio",
                        "CQL_FILTER": cql, "maxFeatures": 50000
                    },
                    headers=headers, verify=False, timeout=90
                )
                if r.status_code == 200 and "csv" in r.headers.get("content-type", ""):
                    bloco_df = pd.read_csv(
                        io.BytesIO(r.content),
                        usecols=["longitude", "latitude", "data_hora_gmt", "municipio"],
                    ).dropna(subset=["longitude", "latitude", "data_hora_gmt"])
                    bloco_df = pd.DataFrame({
                        "lon": bloco_df["longitude"].astype("float32"),
                        "lat": bloco_df["latitude"].astype("float32"),
                        "data": bloco_df["data_hora_gmt"].astype(str).str[:10],
                        "municipio": bloco_df["municipio"].fillna("?"),
                    })
                    return bloco_df, True
            except Exception:
                pass
            time.sleep(1.5 * (tentativa + 1))
        return None, False

    partes = []
    blocos_falhos = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for bloco_df, ok in executor.map(_buscar_bloco, blocos):
            if bloco_df is not None and not bloco_df.empty:
                partes.append(bloco_df)
            if not ok:
                blocos_falhos += 1

    if not partes:
        df = pd.DataFrame(columns=["lat", "lon", "data", "municipio"])
    else:
        df = pd.concat(partes, ignore_index=True)
        df["data"] = pd.to_datetime(df["data"])
    df.attrs["blocos_falhos"] = blocos_falhos
    df.attrs["blocos_total"] = len(blocos)
    return df


@st.cache_data(ttl=604800, show_spinner=False, persist="disk")
def _area_queimada_modis_km2(geom_json_str, ano, mes):
    """
    Área queimada (km²) MODIS para UM (geometria, ano, mês) — devolve só um
    número (float), o que é seguro pra persistir em disco (um ee.Image não é,
    por isso fica de fora daqui). Separada de buscar_total_modis pra ser
    reaproveitada também por calcular_anomalia_modis: sem isso, comparar um
    ano com o histórico desde 2001 recalculava cada (ano, mês) do zero toda
    vez que o ano de referência mudava, mesmo quando a maioria já tinha sido
    calculada numa consulta anterior.
    """
    ee_geom = ee.Geometry(json.loads(geom_json_str))
    ee_geom_simple = ee_geom.simplify(maxError=10000)
    data_ini = ee.Date.fromYMD(ano, mes, 1)
    colecao = (
        ee.ImageCollection('MODIS/061/MCD64A1')
        .filterDate(data_ini, data_ini.advance(1, 'month'))
        .filterBounds(ee_geom_simple)
    )
    if colecao.size().getInfo() == 0:
        return 0.0  # dados indisponiveis
    img = colecao.select('BurnDate').max().clip(ee_geom_simple)
    stats = (
        ee.Image.pixelArea().divide(1e6)
        .updateMask(img.gt(0))
        .rename('area_km2')
        .reduceRegion(reducer=ee.Reducer.sum(), geometry=ee_geom_simple,
                      scale=1000, maxPixels=1e13, bestEffort=True)
        .getInfo()
    )
    return round(stats.get('area_km2') or 0, 2)


@st.cache_data(ttl=3600, show_spinner=False)
def buscar_total_modis(geom_json_str, ano, mes):
    """Busca area queimada total MODIS + a imagem pra desenhar no mapa. Cache de 1h."""
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
    area = _area_queimada_modis_km2(geom_json_str, ano, mes)
    return img, area


def _burndate_seguro(colecao):
    """
    Reduz uma ImageCollection de BurnDate a uma única imagem, tratando o caso
    de COLEÇÃO VAZIA.

    Por que existe: `.max()` sobre uma coleção vazia devolve uma imagem com
    ZERO bandas. Qualquer operação seguinte que compare bandas -- como
    `.gt(0)` -- falha com "Image.gt: If one image has no bands, the other must
    also have no bands. Got 0 and 1". Isso acontece de forma rotineira nos
    meses do ano corrente cujos dados MODIS ainda não foram publicados
    (há atraso típico de 1 a 2 meses).

    Retornar uma constante 0 com a banda nomeada faz o cálculo seguir
    normalmente e resultar em 0 km² para aquele mês, que é a resposta correta.
    """
    return ee.Image(
        ee.Algorithms.If(
            colecao.size().gt(0),
            colecao.max(),
            ee.Image.constant(0).rename('BurnDate')
        )
    )


@st.cache_data(ttl=86400, show_spinner=False)
def calcular_anomalia_modis(geom_json_str, ano_ref):
    """
    Compara cada mês de ano_ref com a média histórica (2001 até ano_ref-1) do
    mesmo mês. Cada combinação (ano, mês) passa por _area_queimada_modis_km2,
    que tem cache próprio em disco — ao trocar o ano de referência, os anos
    históricos que já apareceram numa consulta anterior saem do cache em vez
    de recalcular do zero (antes, essa função montava uma única árvore de
    computação no Earth Engine cobrindo TODOS os anos históricos de uma vez,
    então mudar o ano de referência refazia tudo, mesmo a parte repetida).
    """
    anos_historico = list(range(2001, ano_ref))
    n_anos = len(anos_historico)
    if n_anos == 0:
        return pd.DataFrame()

    meses_map = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }

    pares = [(ano_ref, mes) for mes in range(1, 13)]
    pares += [(ano, mes) for ano in anos_historico for mes in range(1, 13)]

    MAX_RETRIES = 4

    def _buscar_par(par):
        ano, mes = par
        for tentativa in range(1, MAX_RETRIES + 1):
            try:
                return par, _area_queimada_modis_km2(geom_json_str, ano, mes)
            except Exception as e:
                msg = str(e)
                if ('Too many concurrent' in msg or '429' in msg) and tentativa < MAX_RETRIES:
                    time.sleep(2 ** tentativa)
                    continue
                return par, 0.0

    # Paraleliza TODOS os pares (ano, mês) de uma vez — não só os 12 meses do
    # ano de referência — porque cada par agora é uma chamada independente e
    # cacheada, então o ganho de paralelismo se estende ao histórico também.
    areas = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futuros = {executor.submit(_buscar_par, par): par for par in pares}
        for futuro in concurrent.futures.as_completed(futuros):
            par, area = futuro.result()
            areas[par] = area

    registros = []
    for mes in range(1, 13):
        val_ref = areas.get((ano_ref, mes), 0.0)
        media = round(
            sum(areas.get((ano, mes), 0.0) for ano in anos_historico) / n_anos, 2
        )
        anomalia = round(((val_ref - media) / media * 100), 1) if media > 0 else 0
        registros.append({
            'Mês': mes,
            'Mês Nome': meses_map[mes],
            f'Área {ano_ref} (km²)': val_ref,
            'Média Histórica (km²)': media,
            'Anomalia (%)': anomalia
        })

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

# =====================================================================
# 🏡 CAR (SICAR) — CRUZAMENTO COM IMÓVEIS RURAIS
# =====================================================================
# WFS público oficial do SICAR (Sistema Nacional de Cadastro Ambiental Rural),
# uma camada por UF (ex: sicar:sicar_imoveis_ms). Sem autenticação, sem
# captcha — mesmo padrão de acesso já usado pro INPE. A resposta NÃO traz
# nome/CPF do proprietário, só código do imóvel (hash), município e área —
# por isso é seguro exibir diretamente no dashboard.

class _SSLLegadoAdapter(HTTPAdapter):
    """
    O geoserver do SICAR (geoserver.car.gov.br) recusa o handshake TLS padrão
    do OpenSSL moderno usado por requests/urllib3 (SSLV3_ALERT_HANDSHAKE_
    FAILURE), mesmo com verify=False — é a config de cifras do servidor, não
    um problema de certificado. Baixar o "nível de segurança" das cifras
    aceitas (SECLEVEL=1) resolve; um navegador comum já negocia isso sozinho,
    por isso o problema só aparece em código, não ao testar a URL no navegador.
    """
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


@st.cache_data(ttl=604800, show_spinner=False, persist="disk")
def buscar_car_imoveis(uf, bbox_wgs84, max_features=5000):
    """
    Busca imóveis rurais do CAR via WFS, filtrados por bbox (nunca o estado
    inteiro — algumas UFs têm centenas de milhares de imóveis). Nunca levanta
    exceção: indisponibilidade do serviço (comum em portais públicos) devolve
    um GeoDataFrame vazio em vez de derrubar a aba.
    """
    colunas = ["cod_imovel", "municipio", "area", "geometry"]
    url = "https://geoserver.car.gov.br/geoserver/sicar/ows"
    minx, miny, maxx, maxy = bbox_wgs84
    params = {
        "service": "WFS", "version": "1.0.0", "request": "GetFeature",
        "typeName": f"sicar:sicar_imoveis_{uf.lower()}",
        "outputFormat": "application/json",
        "bbox": f"{minx},{miny},{maxx},{maxy},EPSG:4326",
        "maxFeatures": max_features,
    }
    session = requests.Session()
    session.mount("https://", _SSLLegadoAdapter())
    try:
        r = session.get(url, params=params, timeout=60, verify=False)
        if r.status_code != 200:
            return gpd.GeoDataFrame(columns=colunas)
        dados = r.json()
        if not dados.get("features"):
            return gpd.GeoDataFrame(columns=colunas)
        gdf = gpd.GeoDataFrame.from_features(dados["features"], crs="EPSG:4326")
        cols_presentes = [c for c in colunas if c in gdf.columns]
        return gdf[cols_presentes]
    except Exception:
        return gpd.GeoDataFrame(columns=colunas)


MESES_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _resumo_por_area(pts, n_janelas):
    """
    Resume os focos dentro de cada área protegida. `pts` tem uma linha por foco
    (pode repetir o mesmo foco se ele cai em duas áreas sobrepostas) com:
    area_id, lon, lat, data (datetime) e janela (0 = últimos 12 meses, 1 = os 12
    meses antes desses, ...).

    Devolve um DataFrame indexado por area_id com:
      focos            total de focos no período
      anos_com_fogo    em quantas das `n_janelas` janelas de 12 meses houve fogo
      dias_com_foco    dias distintos com ao menos um foco (episódios de fogo)
      mes_pico         mês (1-12) com mais focos
      pct_reincidente  % dos locais (células de ~1 km) que queimaram em 2+ janelas
      recorrencia      Recorrente / Frequente / Ocasional
    """
    p = pts.copy()
    p["dia"] = p["data"].dt.normalize()
    p["mes"] = p["data"].dt.month
    g = p.groupby("area_id")
    resumo = pd.DataFrame({
        "focos": g.size(),
        "anos_com_fogo": g["janela"].nunique(),
        "dias_com_foco": g["dia"].nunique(),
    })
    por_mes = p.groupby(["area_id", "mes"]).size().unstack(fill_value=0).reindex(columns=range(1, 13), fill_value=0)
    resumo["mes_pico"] = por_mes.idxmax(axis=1)
    # Reincidência por local: célula de 0,01° (~1 km). Um mesmo lugar que queima em
    # anos diferentes é fogo recorrente de verdade, diferente de focos espalhados.
    p["cx"] = np.floor(p["lon"].to_numpy() * 100).astype(int)
    p["cy"] = np.floor(p["lat"].to_numpy() * 100).astype(int)
    anos_por_celula = p.groupby(["area_id", "cx", "cy"])["janela"].nunique()
    resumo["pct_reincidente"] = (anos_por_celula >= 2).groupby("area_id").mean() * 100
    limite_freq = int(np.ceil(n_janelas / 2))
    resumo["recorrencia"] = np.select(
        [resumo["anos_com_fogo"] >= n_janelas, resumo["anos_com_fogo"] >= limite_freq],
        ["Recorrente", "Frequente"], default="Ocasional",
    )
    return resumo.sort_values("focos", ascending=False)


# =====================================================================
# CALENDÁRIO DO FOGO POR MUNICÍPIO (aba Planejamento de Brigadas)
# Série histórica MODIS MCD64A1: área queimada por município e por mês.
# =====================================================================

@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _modis_mensal_municipios_ano(fc_json_str, ano, escala):
    """
    Área queimada (km²) de cada município em cada mês de UM ano, numa única
    consulta ao Earth Engine (imagem de 12 bandas, uma por mês, somada por
    município com reduceRegions). Devolve só números, então pode ir para o
    cache em disco: um ano já calculado não é refeito em consultas seguintes.
    """
    feats = json.loads(fc_json_str)["features"]
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry(f["geometry"]), {"mid": f["properties"]["mid"]}) for f in feats
    ])
    col = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f"{ano}-01-01", f"{ano + 1}-01-01")
    bandas = [f"m{m:02d}" for m in range(1, 13)]
    meses = []
    for m in range(1, 13):
        burn = _burndate_seguro(col.filter(ee.Filter.calendarRange(m, m, "month"))).select("BurnDate")
        meses.append(ee.Image.pixelArea().divide(1e6).updateMask(burn.gt(0)).rename(bandas[m - 1]))
    img = ee.Image.cat(meses)
    res = img.reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=escala, tileScale=4)
    res = res.select(["mid"] + bandas, None, False).getInfo()
    linhas = []
    for f in res["features"]:
        p = f["properties"]
        for m in range(1, 13):
            linhas.append((int(p["mid"]), ano, m, float(p.get(bandas[m - 1]) or 0.0)))
    return pd.DataFrame(linhas, columns=["mid", "ano", "mes", "area_km2"])


def buscar_calendario_fogo(gdf_munis, anos, escala, progresso=None):
    """
    Junta a área queimada mensal de todos os anos pedidos. Cada ano é uma
    consulta em cache próprio; até 4 anos em paralelo, com nova tentativa
    quando o Earth Engine responde 'Too many concurrent aggregations' (429).
    Devolve (DataFrame longo mid/ano/mes/area_km2, lista de anos que falharam).
    """
    g = gdf_munis[["geometry"]].copy()
    g["mid"] = np.arange(len(g))
    # Coordenadas com 3 casas (~100 m) para reduzir o tamanho do pedido ao EE
    g["geometry"] = g.geometry.set_precision(0.001)
    g = g[~g.geometry.is_empty]
    fc_json_str = g.to_json()

    def _um_ano(ano):
        for tentativa in range(1, 5):
            try:
                return ano, _modis_mensal_municipios_ano(fc_json_str, ano, escala)
            except Exception as e:
                msg = str(e)
                if ("Too many concurrent" in msg or "429" in msg) and tentativa < 4:
                    time.sleep(2 ** tentativa)
                    continue
                return ano, None
        return ano, None

    partes, falhos = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futuros = [executor.submit(_um_ano, a) for a in anos]
        for i, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            ano, df = fut.result()
            if df is None:
                falhos.append(ano)
            else:
                partes.append(df)
            if progresso is not None:
                progresso.progress(i / len(anos), text=f"Anos processados: {i} de {len(anos)}")
    if not partes:
        return pd.DataFrame(columns=["mid", "ano", "mes", "area_km2"]), sorted(falhos)
    return pd.concat(partes, ignore_index=True), sorted(falhos)


def janela_critica(perfil_12, fracao=0.8):
    """
    Menor sequência de meses CONSECUTIVOS (podendo virar o ano, ex.: Dez–Fev)
    que concentra pelo menos `fracao` da área queimada média do ano.
    Devolve (mes_inicio, mes_fim, fracao_concentrada), meses de 1 a 12,
    ou (None, None, 0) se não houve fogo.
    """
    v = np.asarray(perfil_12, dtype=float)
    total = v.sum()
    if total <= 0:
        return None, None, 0.0
    v2 = np.concatenate([v, v])
    for tamanho in range(1, 13):
        somas = [v2[s:s + tamanho].sum() for s in range(12)]
        s_best = int(np.argmax(somas))
        if somas[s_best] >= fracao * total - 1e-9:
            return s_best + 1, (s_best + tamanho - 1) % 12 + 1, somas[s_best] / total
    return 1, 12, 1.0


def resumo_calendario(df_longo, n_anos):
    """
    Por município (mid): área média anual, anos com fogo, mês de pico,
    janela crítica (80% da área) e perfil médio mensal (km²/mês).
    """
    perfil = (
        df_longo.groupby(["mid", "mes"])["area_km2"].sum().unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0) / n_anos
    )
    por_ano = df_longo.groupby(["mid", "ano"])["area_km2"].sum()
    resumo = pd.DataFrame({
        "media_anual_km2": perfil.sum(axis=1),
        "anos_com_fogo": (por_ano > 0).groupby("mid").sum(),
    })
    resumo["mes_pico"] = perfil.idxmax(axis=1)
    jan = [janela_critica(perfil.loc[i].to_numpy()) for i in perfil.index]
    resumo["ini"] = [j[0] for j in jan]
    resumo["fim"] = [j[1] for j in jan]
    resumo["pct_janela"] = [j[2] * 100 for j in jan]
    return resumo, perfil


# =====================================================================
# RISCO PREDITIVO EM GRADE (aba Risco Preditivo)
# Unidade = célula da grade × semana; alvo = houve foco INPE na célula na
# semana. Entram TODAS as células da região, inclusive as que nunca queimam,
# para o modelo aprender ONDE e QUANDO o fogo acontece (o modelo
# caso-controle anterior só via locais que já tinham queimado).
# =====================================================================

EPOCA_SEMANAS = pd.Timestamp("2000-01-03")  # segunda-feira; semana k começa em EPOCA + 7k dias
BLOCO_SEMANAS = 13  # consultas de clima em blocos de ~3 meses, cada bloco com cache próprio

ASSET_MAPBIOMAS = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1"
)
ANO_MAPBIOMAS = 2023
GRUPOS_MAPBIOMAS = {
    "floresta": [3, 5, 6, 49],
    "savana": [4],
    "campo_natural": [11, 12, 29, 50],
    "pastagem": [15],
    "agricultura": [9, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62],
    "mosaico": [21],
}

FEATURES_GRADE = [
    "focos_sem_ant", "focos_4sem", "focos_52sem", "focos_ano_ant",
    "chuva_sem_ant", "chuva_4sem", "semanas_secas",
    "tmax_sem_ant", "ur_sem_ant", "ur_4sem", "vento_sem_ant",
    "ndvi_mes_ant", *GRUPOS_MAPBIOMAS.keys(), "sem_sin", "sem_cos",
]
NOMES_FEATURES_GRADE = {
    "focos_sem_ant": "Focos na semana anterior",
    "focos_4sem": "Focos nas 4 semanas anteriores",
    "focos_52sem": "Focos nas 52 semanas anteriores",
    "focos_ano_ant": "Focos na mesma época do ano anterior",
    "chuva_sem_ant": "Chuva na semana anterior (mm)",
    "chuva_4sem": "Chuva nas 4 semanas anteriores (mm)",
    "semanas_secas": "Semanas seguidas sem chuva (< 5 mm)",
    "tmax_sem_ant": "Temperatura máxima média (°C)",
    "ur_sem_ant": "Umidade relativa média (%)",
    "ur_4sem": "Umidade relativa média em 4 semanas (%)",
    "vento_sem_ant": "Vento médio (m/s)",
    "ndvi_mes_ant": "NDVI do mês anterior",
    "floresta": "% de floresta (MapBiomas)",
    "savana": "% de savana (MapBiomas)",
    "campo_natural": "% de campo e área úmida natural (MapBiomas)",
    "pastagem": "% de pastagem (MapBiomas)",
    "agricultura": "% de agricultura e silvicultura (MapBiomas)",
    "mosaico": "% de mosaico de usos (MapBiomas)",
    "sem_sin": "Época do ano (seno)",
    "sem_cos": "Época do ano (cosseno)",
}


def _data_semana(k):
    """Data de início (segunda-feira) da semana k."""
    return EPOCA_SEMANAS + pd.Timedelta(days=7 * int(k))


def _semana_de(datas):
    """Índice da semana de cada data (Series de datas -> Series de inteiros)."""
    return (pd.to_datetime(datas) - EPOCA_SEMANAS).dt.days // 7


def montar_grade(limite_gdf, max_celulas=2500):
    """
    Divide a região em células quadradas (em graus) que tocam o limite. Usa a
    menor célula entre 0,05° (~5 km) e 1° (~110 km) que mantenha no máximo
    `max_celulas` células: estado médio fica com 0,25° (~25 km), bioma grande
    com 0,5°. Devolve (GeoDataFrame com cid/i/j/lon_c/lat_c, tamanho em graus).
    """
    geom = limite_gdf.to_crs("EPSG:4326").geometry.union_all()
    shapely.prepare(geom)
    x0, y0, x1, y1 = geom.bounds
    for res in (0.05, 0.1, 0.25, 0.5, 1.0):
        i0, i1 = int(np.floor(x0 / res + 1e-9)), int(np.floor(x1 / res + 1e-9))
        j0, j1 = int(np.floor(y0 / res + 1e-9)), int(np.floor(y1 / res + 1e-9))
        if (i1 - i0 + 1) * (j1 - j0 + 1) > max_celulas * 6 and res < 1.0:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        ii, jj = ii.ravel(), jj.ravel()
        caixas = shapely.box(ii * res, jj * res, (ii + 1) * res, (jj + 1) * res)
        dentro = shapely.intersects(caixas, geom)
        if dentro.sum() <= max_celulas or res >= 1.0:
            grade = gpd.GeoDataFrame(
                {"i": ii[dentro], "j": jj[dentro]}, geometry=caixas[dentro], crs="EPSG:4326"
            ).reset_index(drop=True)
            grade["cid"] = np.arange(len(grade))
            grade["lon_c"] = (grade["i"] + 0.5) * res
            grade["lat_c"] = (grade["j"] + 0.5) * res
            return grade, res
    raise RuntimeError("Não foi possível montar a grade para a região.")


def _retangulos_grade(grade, res):
    """Retângulos das células em texto JSON (chave estável para o cache)."""
    return json.dumps([
        [round(i * res, 4), round(j * res, 4), round((i + 1) * res, 4), round((j + 1) * res, 4)]
        for i, j in zip(grade["i"], grade["j"])
    ])


def _fc_grade(retangulos_json):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Rectangle(r, "EPSG:4326", False), {"cid": c})
        for c, r in enumerate(json.loads(retangulos_json))
    ])


def _ur_diaria(img):
    """Umidade relativa (%) pela fórmula de Magnus, a partir da temperatura e do ponto de orvalho."""
    t = img.select("temperature_2m").subtract(273.15)
    td = img.select("dewpoint_temperature_2m").subtract(273.15)
    return td.expression(
        "100 * exp(17.625 * td / (243.04 + td)) / exp(17.625 * t / (243.04 + t))",
        {"t": t, "td": td},
    ).rename("ur")


def _vento_diario(img):
    return img.expression(
        "sqrt(u * u + v * v)",
        {"u": img.select("u_component_of_wind_10m"), "v": img.select("v_component_of_wind_10m")},
    ).rename("vento")


@st.cache_data(ttl=21600, show_spinner=False)
def _ultima_data_era5():
    """Último dia com dado no ERA5-Land diário (o produto sai com alguns dias de atraso)."""
    agora = datetime.now()
    ms = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate((agora - timedelta(days=150)).strftime("%Y-%m-%d"),
                    (agora + timedelta(days=1)).strftime("%Y-%m-%d"))
        .aggregate_max("system:time_start").getInfo()
    )
    if ms is None:
        raise RuntimeError("ERA5-Land sem dados recentes no Earth Engine.")
    return pd.Timestamp(ms, unit="ms").normalize()


@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _clima_grade_bloco(retangulos_json, k_ini, n_sem, escala):
    """
    Clima semanal ERA5-Land de cada célula, semanas k_ini..k_ini+n_sem-1, numa
    consulta só (4 bandas por semana: temperatura máxima, chuva, umidade, vento).
    """
    fc = _fc_grade(retangulos_json)
    col = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    imgs, bandas = [], []
    for w in range(n_sem):
        ini = _data_semana(k_ini + w)
        sub = col.filterDate(ini.strftime("%Y-%m-%d"), (ini + pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
        sfx = f"_{w:02d}"
        imgs += [
            sub.select("temperature_2m_max").mean().subtract(273.15).rename("tmax" + sfx),
            sub.select("total_precipitation_sum").sum().multiply(1000).rename("chuva" + sfx),
            sub.map(_ur_diaria).mean().rename("ur" + sfx),
            sub.map(_vento_diario).mean().rename("vento" + sfx),
        ]
        bandas += ["tmax" + sfx, "chuva" + sfx, "ur" + sfx, "vento" + sfx]
    res = ee.Image.cat(imgs).reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=escala, tileScale=2
    )
    res = res.select(["cid"] + bandas, None, False).getInfo()
    linhas = []
    for f in res["features"]:
        p = f["properties"]
        for w in range(n_sem):
            sfx = f"_{w:02d}"
            linhas.append((int(p["cid"]), k_ini + w, p.get("tmax" + sfx), p.get("chuva" + sfx),
                           p.get("ur" + sfx), p.get("vento" + sfx)))
    return pd.DataFrame(linhas, columns=["cid", "semana", "tmax", "chuva", "ur", "vento"])


@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _ndvi_grade_ano(retangulos_json, ano, ultimo_mes):
    """NDVI médio mensal (MODIS MOD13A2, 1 km) de cada célula, meses 1..ultimo_mes do ano."""
    fc = _fc_grade(retangulos_json)
    col = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI").filter(
        ee.Filter.calendarRange(ano, ano, "year"))
    imgs, bandas = [], []
    for m in range(1, ultimo_mes + 1):
        c = col.filter(ee.Filter.calendarRange(m, m, "month"))
        vazio = ee.Image.constant(0).updateMask(ee.Image.constant(0))
        img = ee.Image(ee.Algorithms.If(c.size().gt(0), c.mean(), vazio))
        imgs.append(img.multiply(0.0001).rename(f"n{m:02d}"))
        bandas.append(f"n{m:02d}")
    res = ee.Image.cat(imgs).reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=2
    )
    res = res.select(["cid"] + bandas, None, False).getInfo()
    linhas = [
        (int(f["properties"]["cid"]), ano, m, f["properties"].get(f"n{m:02d}"))
        for f in res["features"] for m in range(1, ultimo_mes + 1)
    ]
    return pd.DataFrame(linhas, columns=["cid", "ano", "mes", "ndvi"])


@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _mapbiomas_grade(retangulos_json):
    """Fração de cada grupo de uso do solo (MapBiomas) em cada célula."""
    fc = _fc_grade(retangulos_json)
    img = ee.Image(ASSET_MAPBIOMAS).select(f"classification_{ANO_MAPBIOMAS}")
    bandas = list(GRUPOS_MAPBIOMAS.keys())
    cat = ee.Image.cat([
        img.remap(classes, [1] * len(classes), 0).rename(nome)
        for nome, classes in GRUPOS_MAPBIOMAS.items()
    ])
    res = cat.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=500, tileScale=4)
    res = res.select(["cid"] + bandas, None, False).getInfo()
    return pd.DataFrame([
        {"cid": int(f["properties"]["cid"]), **{b: f["properties"].get(b) for b in bandas}}
        for f in res["features"]
    ])


@st.cache_data(ttl=2592000, show_spinner=False, persist="disk")
def _focos_semanais_periodo(tipo, uf, bioma, muni, d_ini, d_fim, res):
    """
    Focos INPE (satélites de referência) de um período, já somados por
    (coluna i, linha j da grade, semana). Guarda só a contagem, não os pontos,
    para caber vários anos de bioma inteiro na memória. Falha em mais de 20%
    das janelas de busca levanta erro (e não vai para o cache).
    """
    bloco = {"Por Município": 14, "Por Estado": 5}.get(tipo, 3)
    df = buscar_focos_com_coords(
        tipo, uf, bioma, muni, d_ini, d_fim, dias_bloco=bloco, satelites=SATELITES_PERIODO
    )
    falhos, total = df.attrs.get("blocos_falhos", 0), df.attrs.get("blocos_total", 0)
    if total and falhos > 0.2 * total:
        raise RuntimeError(f"INPE: {falhos} de {total} janelas de busca falharam ({d_ini} a {d_fim}).")
    if df.empty:
        return pd.DataFrame(columns=["i", "j", "semana", "n"]), falhos, total
    cnt = pd.DataFrame({
        "i": np.floor(df["lon"].to_numpy(dtype=float) / res + 1e-9).astype(int),
        "j": np.floor(df["lat"].to_numpy(dtype=float) / res + 1e-9).astype(int),
        "semana": _semana_de(df["data"]).to_numpy(),
    }).groupby(["i", "j", "semana"]).size().rename("n").reset_index()
    return cnt, falhos, total


def _executar_paralelo(funcao, lista_args, max_workers=4, progresso=None, texto=""):
    """
    Roda funcao(*args) para cada tupla de lista_args, até max_workers em
    paralelo, repetindo quando o Earth Engine responde 429 (muitas consultas
    simultâneas). Devolve a lista de resultados na mesma ordem (None = falhou)
    e a lista de mensagens de erro.
    """
    def _uma(args):
        for tentativa in range(1, 5):
            try:
                return funcao(*args), None
            except Exception as e:
                msg = str(e)
                if ("Too many concurrent" in msg or "429" in msg) and tentativa < 4:
                    time.sleep(2 ** tentativa)
                    continue
                return None, msg[:200]
        return None, "sem resposta"

    resultados, erros = [None] * len(lista_args), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_uma, a): n for n, a in enumerate(lista_args)}
        for feitos, fut in enumerate(concurrent.futures.as_completed(futuros), 1):
            valor, erro = fut.result()
            resultados[futuros[fut]] = valor
            if erro:
                erros.append(erro)
            if progresso is not None:
                progresso.progress(feitos / len(lista_args), text=f"{texto}: {feitos} de {len(lista_args)}")
    return resultados, erros


def montar_tabela_grade(grade, k_alvo_ini, k_ult, cnt_focos, clima, ndvi, uso):
    """
    Uma linha por célula × semana-alvo t, para t de k_alvo_ini até k_ult + 1.
    Todas as variáveis usam só informação até a semana t-1 (nunca a própria
    semana prevista). A semana k_ult + 1 não tem alvo: é a previsão.

    cnt_focos: i, j, semana, n | clima: cid, semana, tmax, chuva, ur, vento
    ndvi: cid, ano, mes, ndvi  | uso: cid + colunas de GRUPOS_MAPBIOMAS
    """
    n = len(grade)
    T = np.arange(k_alvo_ini, k_ult + 2)
    n_t = len(T)

    # --- Focos: matriz célula × semana a partir de k_alvo_ini - 54 ---
    k_f0 = k_alvo_ini - 54
    F = np.zeros((n, k_ult - k_f0 + 1), dtype=np.float64)
    if cnt_focos is not None and not cnt_focos.empty:
        ij2cid = {(int(i), int(j)): int(c) for i, j, c in zip(grade["i"], grade["j"], grade["cid"])}
        cids = np.array([ij2cid.get((int(i), int(j)), -1) for i, j in zip(cnt_focos["i"], cnt_focos["j"])])
        sem = cnt_focos["semana"].to_numpy(dtype=int)
        ok = (cids >= 0) & (sem >= k_f0) & (sem <= k_ult)
        np.add.at(F, (cids[ok], sem[ok] - k_f0), cnt_focos["n"].to_numpy(dtype=float)[ok])
    csF = np.concatenate([np.zeros((n, 1)), np.cumsum(F, axis=1)], axis=1)

    def soma_focos(lo, hi):  # soma das semanas t+lo .. t+hi (lo <= hi < 0)
        return csF[:, T + hi - k_f0 + 1] - csF[:, T + lo - k_f0]

    col = {
        "focos_sem_ant": F[:, T - 1 - k_f0],
        "focos_4sem": soma_focos(-4, -1),
        "focos_52sem": soma_focos(-52, -1),
        "focos_ano_ant": soma_focos(-54, -50),
    }

    # --- Clima: matriz célula × semana a partir de k_alvo_ini - 30 ---
    k_c0 = k_alvo_ini - 30
    nc = k_ult - k_c0 + 1

    def matriz_clima(var):
        M = np.full((n, nc), np.nan)
        if clima is not None and not clima.empty:
            c = clima[(clima["semana"] >= k_c0) & (clima["semana"] <= k_ult)]
            M[c["cid"].to_numpy(dtype=int), c["semana"].to_numpy(dtype=int) - k_c0] = (
                pd.to_numeric(c[var], errors="coerce").to_numpy(dtype=float)
            )
        return M

    chuva, tmax, ur, vento = (matriz_clima(v) for v in ("chuva", "tmax", "ur", "vento"))
    idx1 = T - 1 - k_c0
    csC = np.concatenate([np.zeros((n, 1)), np.cumsum(np.nan_to_num(chuva), axis=1)], axis=1)
    ur_val = ~np.isnan(ur)
    csU = np.concatenate([np.zeros((n, 1)), np.cumsum(np.where(ur_val, ur, 0), axis=1)], axis=1)
    csN = np.concatenate([np.zeros((n, 1)), np.cumsum(ur_val, axis=1)], axis=1)
    secas = np.zeros((n, nc))
    for w in range(nc):
        anterior = secas[:, w - 1] if w > 0 else 0
        secas[:, w] = np.where(chuva[:, w] < 5, anterior + 1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ur4 = (csU[:, idx1 + 1] - csU[:, idx1 - 3]) / (csN[:, idx1 + 1] - csN[:, idx1 - 3])
    col.update({
        "chuva_sem_ant": chuva[:, idx1],
        "chuva_4sem": csC[:, idx1 + 1] - csC[:, idx1 - 3],
        "semanas_secas": secas[:, idx1],
        "tmax_sem_ant": tmax[:, idx1],
        "ur_sem_ant": ur[:, idx1],
        "ur_4sem": ur4,
        "vento_sem_ant": vento[:, idx1],
    })

    # --- NDVI do mês anterior ao início da semana prevista ---
    inicio_t = pd.DatetimeIndex([_data_semana(t) for t in T])
    mes_ant = inicio_t.to_period("M") - 1
    ndvi_m = np.full((n, n_t), np.nan)
    if ndvi is not None and not ndvi.empty:
        per = pd.to_datetime(pd.DataFrame({"year": ndvi["ano"], "month": ndvi["mes"], "day": 1})).dt.to_period("M")
        piv = ndvi.assign(per=per, ndvi=pd.to_numeric(ndvi["ndvi"], errors="coerce")) \
                  .pivot_table(index="cid", columns="per", values="ndvi", aggfunc="mean")
        piv = piv.reindex(index=range(n))
        for col_t, per in enumerate(mes_ant):
            if per in piv.columns:
                ndvi_m[:, col_t] = piv[per].to_numpy(dtype=float)
    col["ndvi_mes_ant"] = ndvi_m

    # --- Uso do solo (fixo no tempo) ---
    uso_idx = uso.set_index("cid").reindex(range(n)) if (uso is not None and not uso.empty) else None
    for g in GRUPOS_MAPBIOMAS:
        v = uso_idx[g].to_numpy(dtype=float) if uso_idx is not None and g in uso_idx else np.full(n, np.nan)
        col[g] = np.repeat(v[:, None], n_t, axis=1)

    # --- Época do ano ---
    ang = 2 * np.pi * inicio_t.dayofyear.to_numpy() / 365.25
    col["sem_sin"] = np.repeat(np.sin(ang)[None, :], n, axis=0)
    col["sem_cos"] = np.repeat(np.cos(ang)[None, :], n, axis=0)

    # --- Alvo (só até k_ult) ---
    alvo = np.full((n, n_t), np.nan)
    alvo[:, :-1] = (F[:, T[:-1] - k_f0] > 0).astype(float)

    df = pd.DataFrame({
        "cid": np.tile(np.arange(n), n_t), "semana": np.repeat(T, n),
        "lon_c": np.tile(grade["lon_c"].to_numpy(), n_t), "lat_c": np.tile(grade["lat_c"].to_numpy(), n_t),
    })
    for nome in FEATURES_GRADE:
        df[nome] = col[nome].T.ravel().astype(np.float32)
    df["alvo"] = alvo.T.ravel()
    return df


def _fabricas_modelos_grade():
    """Candidatos comparados: regressão logística (referência linear), Random
    Forest e boosting. Sem class_weight, para a probabilidade sair na escala
    real (chance de haver foco na célula na semana)."""
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer

    def lr(n):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000))

    def rf(n):
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=20,
            max_samples=min(n, 40000), n_jobs=2, random_state=42))

    def boost(n):
        if _LGBM_DISPONIVEL:
            return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=42, verbosity=-1)
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                              min_samples_leaf=40, random_state=42)

    nome_boost = "LightGBM" if _LGBM_DISPONIVEL else "Gradient Boosting (sklearn)"
    return {"Regressão Logística": lr, "Random Forest": rf, nome_boost: boost}


def curva_captura(semanas, alvo, score, ks=range(1, 51)):
    """
    % das ocorrências de fogo (célula-semana com foco) que caem nas k% células
    priorizadas de cada semana, para k em ks. Empates são desfeitos ao acaso
    (semente fixa), para uma regra com muitos zeros não ganhar nem perder por ordem.
    """
    rng = np.random.default_rng(0)
    d = pd.DataFrame({
        "s": np.asarray(semanas), "y": np.asarray(alvo, dtype=float),
        "sc": np.asarray(score, dtype=float) + rng.uniform(0, 1e-6, len(alvo)),
    })
    total = d["y"].sum()
    if total == 0:
        return np.zeros(len(list(ks)))
    g = d.groupby("s")["sc"]
    d["pct"] = g.rank(ascending=False, method="first") / g.transform("size")
    return np.array([d.loc[d["pct"] <= k / 100, "y"].sum() / total * 100 for k in ks])


def treinar_avaliar_grade(df):
    """
    Divide no tempo: as últimas 52 semanas com alvo são o TESTE (o modelo
    nunca as vê no treino); das semanas anteriores, as últimas 26 servem de
    VALIDAÇÃO para escolher o algoritmo. O escolhido é medido no teste, comparado
    a regras simples, e por fim retreinado com tudo para prever a próxima semana.
    """
    from sklearn.metrics import average_precision_score, brier_score_loss
    from sklearn.inspection import permutation_importance

    feats = [f for f in FEATURES_GRADE if df[f].notna().any()]
    hist = df[df["alvo"].notna()]
    prev = df[df["alvo"].isna()]
    k_ult = int(hist["semana"].max())
    k_teste = k_ult - 51
    treino, teste = hist[hist["semana"] < k_teste], hist[hist["semana"] >= k_teste]
    k_val = k_teste - 26
    tr_in, val = treino[treino["semana"] < k_val], treino[treino["semana"] >= k_val]
    for nome, parte in (("treino", tr_in), ("validação", val), ("teste", teste)):
        if parte["alvo"].sum() < 10 or parte["alvo"].nunique() < 2:
            raise ValueError(
                f"Poucas ocorrências de fogo no período de {nome} para treinar com segurança. "
                "Aumente os anos de histórico ou escolha uma região maior."
            )

    fabricas = _fabricas_modelos_grade()
    comparacao = []
    for nome, fab in fabricas.items():
        m = fab(len(tr_in)).fit(tr_in[feats], tr_in["alvo"].astype(int))
        p = m.predict_proba(val[feats])[:, 1]
        comparacao.append({"Algoritmo": nome, "AUC-ROC": roc_auc_score(val["alvo"], p),
                           "PR-AUC": average_precision_score(val["alvo"], p)})
    comparacao = pd.DataFrame(comparacao)
    escolhido = comparacao.loc[comparacao["PR-AUC"].idxmax(), "Algoritmo"]
    fab = fabricas[escolhido]

    modelo = fab(len(treino)).fit(treino[feats], treino["alvo"].astype(int))
    p_teste = modelo.predict_proba(teste[feats])[:, 1]
    y_teste = teste["alvo"].to_numpy()

    # Regras simples para comparação (o que se faria sem modelo)
    regra_recente = (teste["focos_sem_ant"].to_numpy(dtype=float) * 1e6
                     + teste["focos_4sem"].to_numpy(dtype=float) * 1e3
                     + teste["focos_52sem"].to_numpy(dtype=float))
    regra_hist = teste["focos_52sem"].to_numpy(dtype=float)
    ks = list(range(1, 51))
    curvas = pd.DataFrame({
        "k": ks,
        "Modelo": curva_captura(teste["semana"], y_teste, p_teste, ks),
        "Regra prática (fogo recente)": curva_captura(teste["semana"], y_teste, regra_recente, ks),
        "Histórico do último ano": curva_captura(teste["semana"], y_teste, regra_hist, ks),
        "Sem critério (sorteio)": [float(k) for k in ks],
    })

    metricas = {
        "auc": roc_auc_score(y_teste, p_teste),
        "pr_auc": average_precision_score(y_teste, p_teste),
        "pr_auc_regra": average_precision_score(y_teste, regra_recente),
        "taxa_base": float(y_teste.mean()),
        "brier": brier_score_loss(y_teste, p_teste),
        "semanas_teste": (k_teste, k_ult),
        "semanas_treino": (int(treino["semana"].min()), k_teste - 1),
        "n_treino": len(treino), "n_teste": len(teste),
        "pos_treino": int(treino["alvo"].sum()), "pos_teste": int(y_teste.sum()),
    }

    # Validação espacial: blocos de ~1° nunca divididos entre treino e teste
    grupos = (np.floor(treino["lon_c"]).astype(int).astype(str) + "_"
              + np.floor(treino["lat_c"]).astype(int).astype(str)) if "lon_c" in treino else None
    metricas["auc_espacial"] = None
    if grupos is not None and grupos.nunique() >= 4:
        sub = treino.sample(n=min(len(treino), 80000), random_state=42)
        g_sub = grupos.loc[sub.index]
        aucs = []
        for i_tr, i_te in GroupKFold(n_splits=4).split(sub, groups=g_sub):
            a, b = sub.iloc[i_tr], sub.iloc[i_te]
            if b["alvo"].nunique() < 2 or a["alvo"].nunique() < 2:
                continue
            m = fab(len(a)).fit(a[feats], a["alvo"].astype(int))
            aucs.append(roc_auc_score(b["alvo"], m.predict_proba(b[feats])[:, 1]))
        if aucs:
            metricas["auc_espacial"] = float(np.mean(aucs))

    # Calibração: probabilidade prevista x frequência observada, por decil
    cal = pd.DataFrame({"p": p_teste, "y": y_teste})
    cal["faixa"] = pd.qcut(cal["p"].rank(method="first"), 10, labels=False)
    calibracao = cal.groupby("faixa").agg(prevista=("p", "mean"), observada=("y", "mean")).reset_index()

    # Importância por permutação no teste: quanto a PR-AUC cai ao embaralhar cada variável
    amostra = teste.sample(n=min(len(teste), 10000), random_state=42)
    if amostra["alvo"].nunique() == 2:
        imp = permutation_importance(modelo, amostra[feats], amostra["alvo"].astype(int),
                                     scoring="average_precision", n_repeats=2, random_state=42)
        importancias = pd.DataFrame({"variavel": feats, "queda": imp.importances_mean})
    else:
        importancias = pd.DataFrame({"variavel": feats, "queda": np.nan})

    # Modelo final com todo o histórico, para prever a próxima semana
    final = fab(len(hist)).fit(hist[feats], hist["alvo"].astype(int))
    p_prev = final.predict_proba(prev[feats])[:, 1]

    return {
        "escolhido": escolhido, "comparacao": comparacao, "metricas": metricas,
        "curvas": curvas, "calibracao": calibracao, "importancias": importancias,
        "previsao": pd.DataFrame({"cid": prev["cid"].to_numpy(), "prob": p_prev}),
        "semana_prevista": int(prev["semana"].iloc[0]), "features": feats,
    }


def _liberar_memoria():
    """
    Coleta o lixo e devolve ao sistema a memória que o Python já liberou. Sem o
    malloc_trim, o glibc (Linux do Streamlit Cloud) guarda essa memória para si:
    medido no log, a análise do bioma Amazônia subia 1,3 → 2,2 → 2,8 GB a cada
    reexecução da página até o app cair, mesmo sem dados novos.
    """
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Windows/macOS: não existe, e não precisa


def _log_mem(etapa):
    """Escreve no log do Streamlit Cloud a memória em uso (o app cai sem
    traceback quando passa de 1 GB; isso mostra em qual etapa a memória subiu)."""
    _liberar_memoria()
    try:
        with open("/proc/self/status") as f:
            rss = next(l for l in f if l.startswith("VmRSS")).split()[1]
        print(f"[MEM] {etapa}: {int(rss) // 1024} MB", flush=True)
    except Exception:
        pass


SATELITES_PERIODO = ("AQUA_M-T", "NPP-375", "NPP-375D")


def focos_do_periodo(tipo, uf, bioma, muni, anos, limite_gdf):
    """
    Focos do INPE dos últimos `anos` anos, só lat/lon/data (leve), recortados no
    limite exato da região. Usado pela aba CAR, que tem o próprio
    período — independente da barra lateral. Usa os mesmos satélites de
    referência do padrão da barra lateral; sem esse filtro entrariam TERRA,
    NOAA-20 e o MSG-03 (geoestacionário), que multiplicam os pontos.
    """
    import shapely
    agora = datetime.now()
    d_fim = agora.strftime("%Y-%m-%d")
    d_ini = (agora - timedelta(days=365 * anos)).strftime("%Y-%m-%d")
    bloco = {"Por Município": 14, "Por Estado": 5}.get(tipo, 3)
    df = buscar_focos_com_coords(
        tipo, uf, bioma, muni, d_ini, d_fim, dias_bloco=bloco, satelites=SATELITES_PERIODO
    )
    colunas = ["latitude", "longitude", "data_hora_gmt", "municipio"]
    falhos, total = df.attrs.get("blocos_falhos", 0), df.attrs.get("blocos_total", 0)
    if df.empty:
        saida = pd.DataFrame(columns=colunas)
    else:
        geom = limite_gdf.geometry.union_all()
        dentro = shapely.contains_xy(geom, df["lon"].to_numpy(), df["lat"].to_numpy())
        saida = df[dentro].rename(
            columns={"lat": "latitude", "lon": "longitude", "data": "data_hora_gmt"}
        )[colunas].reset_index(drop=True)
    saida.attrs["blocos_falhos"], saida.attrs["blocos_total"] = falhos, total
    return saida


def _avisar_blocos_falhos(focos):
    falhos = focos.attrs.get("blocos_falhos", 0)
    if falhos:
        st.warning(
            f"⚠️ {falhos} de {focos.attrs.get('blocos_total', '?')} janelas de busca no "
            "INPE falharam mesmo após novas tentativas — a contagem de focos pode estar "
            "SUBESTIMADA. Clique em 'Nova busca' e tente de novo em alguns minutos."
        )


def _adicionar_pontos_folium(mapa, gdf_pts, nome, col_cor, campos, aliases, max_pontos=12000):
    """
    Desenha os focos como UMA camada GeoJSON (leve). Um CircleMarker por foco
    gera um bloco de JavaScript por ponto — com milhares de focos de vários anos
    isso estoura a memória do navegador e do app. Passa do teto → amostra
    aleatória (as contagens e gráficos usam todos os pontos; só o desenho é limitado).
    Retorna True se precisou amostrar.
    """
    pts = gdf_pts
    amostrou = len(pts) > max_pontos
    if amostrou:
        pts = pts.sample(max_pontos, random_state=42)
    folium.GeoJson(
        pts[campos + [col_cor, "geometry"]].__geo_interface__,
        name=nome,
        marker=folium.CircleMarker(radius=4, weight=0.5),
        style_function=lambda f: {
            "fillColor": f["properties"][col_cor], "color": "#1c1c1c",
            "weight": 0.5, "fillOpacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(fields=campos, aliases=aliases),
    ).add_to(mapa)
    return amostrou


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
# Áreas protegidas agora têm aba própria (🌿 Unidades de Conservação); aqui fica
# fixo em "Nenhuma" para o cruzamento antigo da análise principal não rodar.
area_protegida = "Nenhuma"

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

_liberar_memoria()
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
        # Só para desenhar no mapa: o contorno de um bioma/estado tem centenas de
        # milhares de vértices, e o folium o reescreve inteiro no HTML a cada
        # interação. ~2 km de tolerância é invisível no zoom de um bioma/estado.
        limite_mapa = limite if tipo_analise == "Por Município" else limite.assign(
            geometry=limite.geometry.simplify(0.02, preserve_topology=True)
        )
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
                    try:
                        gdf_areas = carregar_areas_protegidas(area_protegida)
                    except Exception as e:
                        st.warning(
                            f"⚠️ Não foi possível carregar os limites de {area_protegida} "
                            f"agora ({e}). Mostrando os focos sem o recorte espacial."
                        )
                        gdf_areas = None

                if area_protegida != "Nenhuma" and gdf_areas is not None:
                    gdf_areas = gdf_areas.to_crs(gdf.crs)
                    st.caption(
                        f"🔎 Debug: {len(gdf_areas)} polígono(s) de {area_protegida} "
                        f"carregados no Brasil todo (CRS: {gdf_areas.crs})."
                    )

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
                    st.caption(
                        f"🔎 Debug: {len(areas_afetadas)} área(s) intersectam a "
                        f"região selecionada (limite CRS: {limite.crs})."
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
                # Usa funcao cacheada -- segunda consulta ao mesmo periodo é instantanea
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

                    if area_protegida != "Nenhuma":
                        st.write(f"🌳 Carregando limites de {area_protegida}...")
                        gdf_areas_br = carregar_areas_protegidas(tipo_area=area_protegida)
                        st.caption(
                            f"🔎 Debug: {len(gdf_areas_br)} polígono(s) de {area_protegida} "
                            f"carregados no Brasil todo (CRS: {gdf_areas_br.crs})."
                        )
                        gdf_areas = gpd.sjoin(
                            gdf_areas_br, limite, predicate='intersects'
                        ).drop(columns=['index_right'])
                        st.caption(
                            f"🔎 Debug: {gdf_areas['nome_area'].nunique() if not gdf_areas.empty else 0} "
                            f"área(s) intersectam a região selecionada (limite CRS: {limite.crs})."
                        )

                        # Desenha TODAS as áreas protegidas que tocam a região desde
                        # já — mesmo que acabem com 0 km² queimados dentro, ou que
                        # não haja NENHUMA queima na região inteira nesse mês/ano,
                        # elas continuam aparecendo no mapa como contexto.
                        areas_afetadas = (
                            gdf_areas[['nome_area', 'geometry']]
                            .drop_duplicates(subset='nome_area')
                            .reset_index(drop=True)
                        )

                        if not gdf_areas.empty and total_valor > 0:
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
                    
                    xaxis_title=nome_eixo_x,
                    yaxis={'categoryorder': 'total ascending'},
                    height=350, margin=dict(t=40, b=20),
                    coloraxis_showscale=False
                )
                figura(fig_areas, "Áreas com maior número de focos de calor no recorte analisado", "INPE (BDQueimadas)")
            with col_alerta2:
                st.markdown("**Lista Completa de Áreas Afetadas**")
                tabela_bc("Áreas afetadas por focos de calor, em ordem decrescente", 
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
        # Ordem de leitura: o que aconteceu -> quão grave -> risco e planejamento ->
        # áreas protegidas -> custo -> propriedades -> exportar
        aba_mapa, aba_graficos, aba_nbr, aba_risco, aba_brigadas, aba_uc, aba_impacto, aba_car, aba_export = st.tabs([
            "🗺️ Mapa de Focos",
            "📈 Gráficos & Anomalia",
            "🔬 Severidade (NBR Sentinel-2)",
            "🎯 Risco Preditivo (ML)",
            "🚒 Planejamento de Brigadas",
            "🌿 Unidades de Conservação",
            "💰 Impacto Econômico",
            "🏡 Propriedades Rurais (CAR)",
            "⬇️ Exportar Dados"
        ])

        # ----------------------------------------------------------
        # ABA 1 — MAPA
        # ----------------------------------------------------------
        _log_mem(f"Dashboard {val_sel}: dados principais carregados")
        with aba_mapa:
            col_controles1, col_controles2 = st.columns([1, 1.2])
            with col_controles1:
                estilo_mapa = st.radio(
                    "🎨 Estilo de Fundo:",
                    ["☀️ Claro (Institucional)", "🌑 Mapa Dark", "🛰️ Satélite (Google)", "🗺️ Mapa Padrão"],
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
                "☀️ Claro (Institucional)": {
                    "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                    "attr": "Esri",
                    "ref_url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                },
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

            # Borda da região selecionada — laranja, no estilo dos painéis do
            # TerraBrasilis/INPE, mais legível sobre o mapa base claro
            folium.GeoJson(
                limite_mapa.__geo_interface__,
                name="Região selecionada",
                style_function=lambda x: {
                    'fillColor': '#e67e22',
                    'fillOpacity': 0.03,
                    'color': '#e67e22',
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
                # radius/blur em PIXELS de tela, não em metros — por isso, afastado o
                # zoom, focos geograficamente distantes ficam pertos na tela e se
                # fundem num "blob" só, dando a falsa impressão de que a região
                # inteira está pegando fogo. Valores menores aqui (e um gradiente que
                # só fica vermelho intenso em densidade bem alta) reduzem esse efeito;
                # é uma limitação inerente de mapa de calor por densidade de pontos,
                # não dá pra eliminar 100% sem trocar de técnica de visualização.
                HeatMap(
                    df_rec[["latitude", "longitude"]].dropna().values.tolist(),
                    name="Densidade de focos",
                    radius=5,
                    blur=7,
                    max_zoom=14,
                    min_opacity=0.2,
                    gradient={0.3: '#ffffb2', 0.5: '#fecc5c',
                               0.7: '#fd8d3c', 0.87: '#f03b20', 1.0: '#bd0026'},
                ).add_to(m)

                # Legenda INPE
                legenda_inpe = """
                <div style="position:fixed; bottom:28px; left:12px; z-index:9999;
                            background:rgba(255,255,255,0.95); padding:10px 14px;
                            border-radius:8px; font-size:12px; color:#2c3e50;
                            line-height:1.9; border:1px solid rgba(0,0,0,0.08);
                            box-shadow:0 1px 4px rgba(0,0,0,0.18);">
                  <b style="font-size:13px; letter-spacing:.5px;">🔥 Intensidade de Focos</b><br>
                  <span style="background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026);
                               display:inline-block;width:130px;height:10px;border-radius:4px;
                               vertical-align:middle;margin-top:4px;"></span><br>
                  <span>Baixa</span>
                  <span style="float:right;">Alta</span>
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
                            background:rgba(255,255,255,0.95); padding:10px 14px;
                            border-radius:8px; font-size:12px; color:#2c3e50;
                            line-height:1.9; border:1px solid rgba(0,0,0,0.08);
                            box-shadow:0 1px 4px rgba(0,0,0,0.18);">
                  <b style="font-size:13px; letter-spacing:.5px;">🗺️ Área Queimada (MODIS)</b><br>
                  <span style="background:linear-gradient(to right,#fff7bc,#fec44f,#fe9929,#ec7014,#cc4c02,#8c2d04);
                               display:inline-block;width:130px;height:10px;border-radius:4px;
                               vertical-align:middle;margin-top:4px;"></span><br>
                  <span>Início do mês</span>
                  <span style="float:right;">Fim do mês</span>
                </div>"""
                m.get_root().html.add_child(folium.Element(legenda_modis))

            _ocultar_url_base(m)
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
                    
                    xaxis_title="Tempo", yaxis_title="Nº de Focos",
                    margin=dict(t=20, b=20)
                )
                figura(fig_line, "Evolução temporal do número de focos de calor", "INPE (BDQueimadas)")

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
                        
                        yaxis={'categoryorder': 'total ascending'},
                        height=320, margin=dict(t=20, b=20),
                        coloraxis_showscale=False
                    )
                    figura(fig_bar, "Municípios com maior número de focos de calor", "INPE (BDQueimadas)")

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
                                col_mes = (
                                    ee.ImageCollection('MODIS/061/MCD64A1')
                                    .filterDate(ini, ini.advance(1, 'month'))
                                    .select('BurnDate')
                                )
                                # _burndate_seguro trata o mês sem dado publicado
                                # (coleção vazia -> imagem com 0 bandas -> .gt(0) falha)
                                img_mes = _burndate_seguro(col_mes).clip(geom_temporal)
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
                        
                        xaxis_title="Mês", yaxis_title="Área Afetada (km²)",
                        margin=dict(t=20, b=20)
                    )
                    figura(fig_line, "Evolução mensal da área afetada pelo fogo (km²)", "MODIS MCD64A1 (NASA)")

                if tipo_analise != "Por Município" and not df_top_mun_modis.empty:
                    st.subheader("🏆 Top 5 Municípios Afetados")
                    fig_bar = px.bar(
                        df_top_mun_modis, x='Valor', y='Município', orientation='h',
                        text='Valor', color='Valor',
                        color_continuous_scale=px.colors.sequential.Reds
                    )
                    fig_bar.update_layout(
                        
                        xaxis_title="Área Afetada (km²)",
                        yaxis={'categoryorder': 'total ascending'},
                        height=350, margin=dict(t=20, b=20),
                        coloraxis_showscale=False
                    )
                    figura(fig_bar, "Cinco municípios com maior área afetada pelo fogo (km²)", "MODIS MCD64A1 (NASA)")

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
                            annotation_font_color='black',
                        )
                        fig_anom.update_layout(
                            
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
                        figura(fig_anom, "Desvio do ano analisado em relação à média histórica de área queimada (%)", "MODIS MCD64A1 (NASA)")

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
                            # Fundo cinza claro (Esri) para não competir com as cores da
                            # severidade; satélite fica como opção no controle de camadas.
                            m_nbr = folium.Map(
                                location=[centro_nbr.y, centro_nbr.x],
                                zoom_start=8, tiles=None
                            )
                            folium.TileLayer(
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri", name="Mapa claro",
                            ).add_to(m_nbr)
                            folium.TileLayer(
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri World Imagery", name="Satélite", show=False,
                            ).add_to(m_nbr)
                            
                            folium.GeoJson(
                                limite_mapa.__geo_interface__,
                                style_function=lambda x: {
                                    'fillColor': 'transparent',
                                    'color': '#e67e22', 'weight': 2
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
                            # Nomes de cidades e limites por cima da severidade
                            folium.TileLayer(
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri Ref", overlay=True, control=False,
                            ).add_to(m_nbr)
                            
                            legenda_html = """
                            <div style="position:fixed; bottom:28px; right:10px; z-index:9999;
                                        background:rgba(255,255,255,0.95); padding:12px 16px;
                                        border-radius:8px; font-size:12px; color:#2c3e50; line-height:2;
                                        border:1px solid rgba(0,0,0,0.08);
                                        box-shadow:0 1px 4px rgba(0,0,0,0.18);">
                                <b style="font-size:13px;">Severidade dNBR</b><br>
                                <span style="color:#1a9850;">■</span> Regeneração (dNBR &lt; -100)<br>
                                <span style="color:#91cf60;">■</span> Não afetado (-100 a 100)<br>
                                <span style="color:#fee08b;">■</span> Baixa (100 a 270)<br>
                                <span style="color:#fc8d59;">■</span> Moderada (270 a 440)<br>
                                <span style="color:#d73027;">■</span> Moderada-Alta (440 a 660)<br>
                                <span style="color:#7a0403;">■</span> Alta (&gt; 660)
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
                            
                            height=280, margin=dict(t=10, b=10)
                        )
                        figura(fig_pizza, "Distribuição da área queimada por classe de severidade (dNBR)", "Sentinel-2 (Copernicus)")

                        # Gráfico de barras
                        fig_bar_sev = px.bar(
                            df_sev_exp, x='Área (km²)', y='Classe', orientation='h',
                            color='Classe', color_discrete_map=cores_sev,
                            text='Área (km²)'
                        )
                        fig_bar_sev.update_layout(
                            showlegend=False,
                            yaxis={'categoryorder': 'total ascending'},
                            height=260, margin=dict(t=10, b=10)
                        )
                        figura(fig_bar_sev, "Área queimada por classe de severidade (dNBR)", "Sentinel-2 (Copernicus)")

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
                # Leitura dos parâmetros do session_state (definidos no expander abaixo).
                # A chave do bioma inclui a região analisada (val_sel) de propósito:
                # sem isso, o valor escolhido numa análise anterior (ex: Pantanal)
                # ficaria preso no session_state e sobrescreveria o bioma detectado
                # da região atual (ex: Amazônia), gerando cálculo com o parâmetro errado.
                _key_bioma = f"bioma_impacto_{val_sel}"
                bioma_padrao = (bioma_detectado if bioma_detectado in VALORES_ECOSSIS
                                else list(VALORES_ECOSSIS.keys())[0])
                bioma_calc = st.session_state.get(_key_bioma, bioma_padrao)
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
                    barmode="stack", height=360,
                    yaxis_title="R$ Bilhões",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=40, b=20),
                )
                figura(fig_cen, "Cenários de impacto econômico do fogo (R$ bilhões)", "Resultados originais do estudo, com base em TNC Brasil")
                tabela_bc("Cenários de impacto econômico do fogo", 
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
                    st.caption(
                        f"Bioma detectado automaticamente para esta análise: **{bioma_padrao}**. "
                        "Você pode sobrescrever abaixo — a escolha vale só para esta região."
                    )
                    pc1, pc2, pc3 = st.columns(3)
                    with pc1:
                        st.selectbox(
                            "Bioma:", list(VALORES_ECOSSIS.keys()),
                            index=list(VALORES_ECOSSIS.keys()).index(bioma_calc),
                            key=_key_bioma
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
        # ABA — PROPRIEDADES RURAIS (CAR)
        # ----------------------------------------------------------
        # Fragmento: clicar em algo DENTRO desta aba reexecuta só a aba, e não o
        # dashboard inteiro (que recarregava focos, limites e mapas a cada clique).
        @st.fragment
        def _aba_car():
            st.subheader("🏡 Propriedades Rurais (CAR) Afetadas por Queimada")
            st.caption(
                "Cruza os focos de calor detectados com os limites de imóveis rurais "
                "cadastrados no CAR (SICAR), via WFS público oficial. A resposta não "
                "traz nome ou CPF do proprietário — apenas o código do imóvel (hash), "
                "município e área."
            )

            # Região desta aba: UF e município escolhidos AQUI (independe da barra
            # lateral). O CAR é organizado por estado, e consultar um estado inteiro
            # é lento e passa do limite de imóveis, então a análise é de um município.
            _ufs = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
                    "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]
            _col_uf, _col_mun = st.columns([1, 3])
            estado_car = _col_uf.selectbox(
                "Estado:", _ufs, index=_ufs.index(estado_dd) if estado_dd in _ufs else 0,
                key="uf_car",
            )
            _cidades_car = ["— escolha um município —"] + list(buscar_cidades(estado_car))
            _muni_ini = municipio_dd if (tipo_analise == "Por Município" and estado_car == estado_dd) else None
            _sel_car = _col_mun.selectbox(
                "Município:", _cidades_car,
                index=_cidades_car.index(_muni_ini) if _muni_ini in _cidades_car else 0,
                key=f"muni_car_{estado_car}",
                help="O CAR é consultado só para o município escolhido.",
            )
            _car_ok = not _sel_car.startswith("—")
            if _car_ok:
                _tipo_c, _uf_c, _muni_c = "Por Município", estado_car, _sel_car
                _limite_c = carregar_fronteira("Por Município", _uf_c, None, _muni_c)
                _val_c = f"{_muni_c} ({_uf_c})"
            if not _car_ok:
                st.info("👆 Escolha um estado e um município para buscar os imóveis do CAR e os focos.")
            else:
                anos_car = st.select_slider(
                    "Período dos focos de calor:", options=[1, 2, 3, 4, 5], value=1,
                    format_func=lambda x: f"últimos {x} ano(s)", key="anos_car",
                    help="Escolhido aqui na aba, independente da barra lateral."
                )

                _car_key = f"car_res2_{_val_c}_{_uf_c}"
                if _car_key not in st.session_state:
                    st.session_state[_car_key] = None
                _res_car = st.session_state[_car_key]
                car_ja_buscou = isinstance(_res_car, dict)
                car_periodo_mudou = car_ja_buscou and _res_car.get("anos") != anos_car

                if (not car_ja_buscou) or car_periodo_mudou:
                    st.info(
                        "⚡ Busca os imóveis do CAR e os focos de calor do período escolhido, "
                        "dentro dos limites da região selecionada. Pode levar de alguns "
                        "segundos a alguns minutos, conforme o tamanho da região."
                    )
                    if st.button(
                        "🔄 Atualizar com este período" if car_periodo_mudou
                        else "🔎 Buscar Imóveis do CAR e Focos",
                        type="primary", use_container_width=True,
                        key=f"btn_car_{_val_c}_{_uf_c}"
                    ):
                        with st.spinner("🏡 Consultando o SICAR e os focos do INPE..."):
                            if car_ja_buscou:
                                gdf_car_bruto = _res_car["imoveis"]
                            else:
                                gdf_car_bruto = buscar_car_imoveis(_uf_c, tuple(_limite_c.total_bounds))
                            _log_mem("CAR: antes dos focos")
                            focos_car = focos_do_periodo(
                                _tipo_c, _uf_c, None, _muni_c, anos_car, _limite_c
                            )
                            st.session_state[_car_key] = {
                                "imoveis": gdf_car_bruto, "focos": focos_car, "anos": anos_car,
                            }

                _res_car = st.session_state[_car_key]
                if isinstance(_res_car, dict):
                    gdf_car_bruto = _res_car["imoveis"]
                    focos_car = _res_car["focos"]
                    st.caption(
                        f"Resultado abaixo: focos dos últimos **{_res_car['anos']} ano(s)**."
                    )
                else:
                    gdf_car_bruto = None
                    focos_car = None
                if gdf_car_bruto is not None:
                    if gdf_car_bruto.empty:
                        st.warning(
                            "⚠️ Nenhum imóvel do CAR retornado — o serviço pode estar "
                            "instável agora, ou a região realmente não tem imóveis "
                            "cadastrados no recorte consultado. Tente novamente em "
                            "alguns minutos."
                        )
                    elif focos_car.empty:
                        st.info(
                            "Nenhum foco de calor encontrado nesta região no período "
                            "escolhido (ou o serviço do INPE está instável agora)."
                        )
                    else:
                        if len(gdf_car_bruto) >= 5000:
                            st.warning(
                                "⚠️ Atingiu o limite de 5.000 imóveis retornados — pode "
                                "haver mais imóveis na região que não foram carregados. "
                                "Escolha um município menor para uma busca mais completa."
                            )

                        # Renomeia ANTES do cruzamento espacial — os focos já têm uma
                        # coluna 'municipio' própria; sem isso, o sjoin criaria
                        # 'municipio_left'/'municipio_right' e a agregação pegaria a errada.
                        gdf_car = gdf_car_bruto.rename(
                            columns={"municipio": "municipio_car", "area": "area_ha_car"}
                        )
                        _avisar_blocos_falhos(focos_car)

                        gdf_focos_pts = gpd.GeoDataFrame(
                            focos_car,
                            geometry=gpd.points_from_xy(focos_car["longitude"], focos_car["latitude"]),
                            crs="EPSG:4326"
                        )
                        gdf_cruzamento = gpd.sjoin(
                            gdf_focos_pts, gdf_car, predicate="within", how="inner"
                        )

                        if gdf_cruzamento.empty:
                            st.success(
                                "✅ Nenhum dos focos detectados caiu dentro de um imóvel "
                                f"do CAR, entre os {len(gdf_car)} imóveis carregados na região."
                            )
                        else:
                            contagem = (
                                gdf_cruzamento.groupby("cod_imovel")
                                .agg(
                                    n_focos=("cod_imovel", "count"),
                                    municipio=("municipio_car", "first"),
                                    area_ha=("area_ha_car", "first"),
                                )
                                .reset_index()
                                .sort_values("n_focos", ascending=False)
                            )

                            n_imoveis_afetados = len(contagem)
                            n_focos_em_imoveis = int(contagem["n_focos"].sum())

                            col_car1, col_car2, col_car3 = st.columns(3)
                            col_car1.metric("🏡 Imóveis com foco", n_imoveis_afetados)
                            col_car2.metric(
                                "🔥 Focos dentro de imóveis do CAR",
                                f"{n_focos_em_imoveis} de {len(focos_car)}"
                            )
                            col_car3.metric("📋 Imóveis carregados na região", len(gdf_car))

                            st.markdown("**🔥 Top 10 imóveis com mais focos:**")
                            top10 = contagem.head(10)
                            fig_car = px.bar(
                                top10, x="n_focos", y="cod_imovel", orientation="h",
                                color="n_focos",
                                color_continuous_scale=px.colors.sequential.Reds,
                                hover_data=["municipio", "area_ha"],
                                labels={"n_focos": "Nº de focos", "cod_imovel": "Imóvel (CAR)"},
                            )
                            fig_car.update_layout(
                                
                                yaxis={'categoryorder': 'total ascending'},
                                height=380, margin=dict(t=20, b=20),
                                coloraxis_showscale=False,
                            )
                            figura(fig_car, "Imóveis rurais (CAR) com maior número de focos de calor", "SICAR; INPE (BDQueimadas)")

                            tabela_bc("Imóveis rurais (CAR) com focos de calor", 
                                contagem.rename(columns={
                                    "cod_imovel": "Imóvel (CAR)", "n_focos": "Focos",
                                    "municipio": "Município", "area_ha": "Área do imóvel (ha)",
                                }),
                                hide_index=True, use_container_width=True
                            )

                            csv_car = contagem.to_csv(index=False).encode("utf-8-sig")
                            st.download_button(
                                "📄 Baixar CSV — Imóveis do CAR com Foco",
                                data=csv_car,
                                file_name=f"car_focos_{_val_c}_{hoje.strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                            )

                            st.markdown("---")
                            st.markdown("**🗺️ Mapa dos imóveis — limite de cada um, colorido pela intensidade de focos:**")

                            import branca.colormap as branca_cm

                            gdf_render_car = gdf_car.merge(
                                contagem[["cod_imovel", "n_focos"]], on="cod_imovel", how="left"
                            )
                            gdf_render_car["n_focos"] = gdf_render_car["n_focos"].fillna(0)

                            centro_car = _limite_c.geometry.union_all().centroid
                            m_car = folium.Map(
                                location=[centro_car.y, centro_car.x],
                                zoom_start=10,
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri", prefer_canvas=True,
                            )
                            folium.TileLayer(
                                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                                attr="Esri Ref", overlay=True, control=False,
                            ).add_to(m_car)

                            folium.GeoJson(
                                _limite_c.__geo_interface__,
                                name=f"Limite — {_val_c}",
                                style_function=lambda x: {
                                    'fillColor': '#e67e22', 'fillOpacity': 0.02,
                                    'color': '#e67e22', 'weight': 2.5, 'dashArray': '6 3',
                                },
                            ).add_to(m_car)

                            # Contorno de TODOS os imóveis carregados só até um teto —
                            # com milhares de polígonos o navegador engasga pra renderizar;
                            # os afetados por foco (o que importa) sempre aparecem abaixo.
                            if len(gdf_car) <= 1500:
                                folium.GeoJson(
                                    gdf_car.__geo_interface__,
                                    name="Todos os imóveis carregados",
                                    style_function=lambda x: {
                                        'fillOpacity': 0, 'color': '#7f8c8d', 'weight': 0.6,
                                    },
                                ).add_to(m_car)
                            else:
                                st.caption(
                                    f"ℹ️ {len(gdf_car)} imóveis carregados — o contorno "
                                    "individual de todos não é desenhado no mapa por "
                                    "performance do navegador (só os afetados por foco "
                                    "aparecem coloridos abaixo)."
                                )

                            # Faixas discretas ("de X até Y"), estilo TerraBrasilis/INPE,
                            # em vez de um degradê contínuo — mais fácil de ler no mapa.
                            max_focos_imovel = int(contagem["n_focos"].max())
                            paleta_car = ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']
                            n_bins_car = min(len(paleta_car), max(max_focos_imovel, 1))
                            bordas_car = sorted(set(
                                int(round(v)) for v in
                                np.linspace(1, max_focos_imovel + 1, n_bins_car + 1)
                            ))
                            if len(bordas_car) < 2:
                                bordas_car = [1, max_focos_imovel + 1]
                            colormap = branca_cm.StepColormap(
                                colors=paleta_car[:len(bordas_car) - 1],
                                index=bordas_car,
                                vmin=bordas_car[0], vmax=bordas_car[-1],
                                caption="Nº de focos dentro do imóvel"
                            )
                            gdf_afetados_geo = gdf_render_car[gdf_render_car["n_focos"] > 0]
                            folium.GeoJson(
                                gdf_afetados_geo.__geo_interface__,
                                name="Imóveis com foco (intensidade)",
                                style_function=lambda feat: {
                                    'fillColor': colormap(feat['properties']['n_focos']),
                                    'fillOpacity': 0.75,
                                    'color': '#1c1c1c', 'weight': 1,
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=["cod_imovel", "municipio_car", "n_focos"],
                                    aliases=["Imóvel:", "Município:", "Focos:"],
                                    style=(
                                        "font-size:12px; background:white; color:#2c3e50; "
                                        "border-radius:6px; box-shadow:2px 2px 6px rgba(0,0,0,0.25);"
                                    ),
                                ),
                            ).add_to(m_car)
                            colormap.add_to(m_car)

                            pts_car = gdf_cruzamento[["data_hora_gmt", "geometry"]].copy()
                            pts_car["cor"] = "#ffeb3b"
                            pts_car["data_txt"] = pd.to_datetime(
                                pts_car["data_hora_gmt"], errors="coerce"
                            ).dt.strftime("%d/%m/%Y").fillna("?")
                            _adicionar_pontos_folium(
                                m_car, pts_car, "Focos de calor", "cor", ["data_txt"], ["Data:"]
                            )

                            _ocultar_url_base(m_car)
                            folium.LayerControl(collapsed=False).add_to(m_car)
                            _car_map_key = f"mapa_car_{_val_c}_{_uf_c}_{len(gdf_cruzamento)}"
                            st_folium(m_car, width=None, height=650, returned_objects=[], key=_car_map_key)

                    st.markdown("")
                    if st.button(
                        "🔄 Nova Busca no CAR", key=f"btn_car_reset_{_val_c}_{_uf_c}"
                    ):
                        st.session_state[_car_key] = None
                        st.rerun(scope="fragment")

        with aba_car:
            _aba_car()

        # ----------------------------------------------------------
        # ABA — UNIDADES DE CONSERVAÇÃO / TERRAS INDÍGENAS
        # ----------------------------------------------------------
        # Fragmento: clicar em algo DENTRO desta aba reexecuta só a aba.
        @st.fragment
        def _aba_uc():
            st.subheader("🌿 Unidades de Conservação e Terras Indígenas")
            st.caption(
                "Para cada área protegida da região selecionada: quantos focos de calor, "
                "em quantos anos o fogo volta a ela (recorrência) e em que mês ele se "
                "concentra. A região é a da barra lateral; o período é escolhido aqui."
            )
            col_t, col_p = st.columns(2)
            tipo_uc = col_t.radio(
                "Tipo de área protegida:", ["Unidades de Conservação", "Terras Indígenas"],
                key="tipo_uc", horizontal=True,
            )
            anos_uc = col_p.select_slider(
                "Período analisado:", options=[2, 3, 4, 5], value=3,
                format_func=lambda x: f"últimos {x} anos", key="anos_uc",
                help="Cada ano é uma janela de 12 meses contada a partir de hoje. "
                     "Mínimo de 2 anos, senão não dá para medir recorrência.",
            )
            grupo_uc = "Todas"
            if tipo_uc == "Unidades de Conservação":
                grupo_uc = st.radio(
                    "Grupo da unidade:", ["Todas", "Proteção Integral", "Uso Sustentável"],
                    key="grupo_uc", horizontal=True,
                    help="Proteção Integral (parques, reservas biológicas...) não admite uso "
                         "direto dos recursos; Uso Sustentável (APAs, florestas nacionais...) admite.",
                )
            if tipo_analise != "Por Município" and anos_uc >= 4:
                st.warning(
                    "⏱️ Região grande com vários anos de focos: a busca pode levar alguns "
                    "minutos. Não feche a página."
                )

            _uc_key = f"uc_res_{val_sel}_{tipo_uc}_{grupo_uc}"
            _res_uc = st.session_state.get(_uc_key)
            uc_ja = isinstance(_res_uc, dict)
            uc_mudou = uc_ja and _res_uc.get("anos") != anos_uc

            if (not uc_ja) or uc_mudou:
                st.info(
                    "⚡ Busca os focos do período na região e cruza com os limites das áreas "
                    "protegidas. Pode levar de alguns segundos a alguns minutos."
                )
                if st.button(
                    "🔄 Atualizar com este período" if uc_mudou else "🔎 Analisar Recorrência do Fogo",
                    type="primary", use_container_width=True, key=f"btn_uc_{val_sel}_{tipo_uc}_{grupo_uc}",
                ):
                    with st.spinner("🌿 Carregando áreas protegidas e focos de calor do INPE..."):
                        try:
                            areas_br = carregar_areas_protegidas(tipo_uc)
                            if grupo_uc != "Todas" and "grupo" in areas_br.columns:
                                areas_br = areas_br[areas_br["grupo"] == grupo_uc]
                            areas_reg = gpd.sjoin(
                                areas_br, limite[["geometry"]], predicate="intersects", how="inner"
                            )
                            areas_reg = areas_reg[~areas_reg.index.duplicated()].drop(
                                columns=["index_right"], errors="ignore"
                            ).reset_index(drop=True)
                            areas_reg["area_id"] = np.arange(len(areas_reg))
                            _focos_uc = focos_do_periodo(
                                tipo_analise, estado_dd, bioma_dd, municipio_dd, anos_uc, limite
                            )
                            n_dentro_regiao = len(_focos_uc)
                            if _focos_uc.empty or areas_reg.empty:
                                pts_uc = pd.DataFrame(columns=["area_id", "lon", "lat", "data", "janela"])
                                n_unicos = 0
                            else:
                                gdf_f = gpd.GeoDataFrame(
                                    _focos_uc,
                                    geometry=gpd.points_from_xy(_focos_uc["longitude"], _focos_uc["latitude"]),
                                    crs="EPSG:4326",
                                )
                                dentro = gpd.sjoin(
                                    gdf_f, areas_reg[["area_id", "geometry"]], predicate="within", how="inner"
                                )
                                n_unicos = int(dentro.index.nunique())
                                datas = pd.to_datetime(dentro["data_hora_gmt"])
                                pts_uc = pd.DataFrame({
                                    "area_id": dentro["area_id"].to_numpy(),
                                    "lon": dentro["longitude"].to_numpy(dtype=float),
                                    "lat": dentro["latitude"].to_numpy(dtype=float),
                                    "data": datas.to_numpy(),
                                })
                                pts_uc["janela"] = (
                                    (pd.Timestamp(hoje.date()) - pts_uc["data"]).dt.days // 365
                                ).clip(lower=0, upper=anos_uc - 1)
                            st.session_state[_uc_key] = {
                                "areas": areas_reg, "pts": pts_uc, "anos": anos_uc,
                                "n_focos_regiao": n_dentro_regiao, "n_unicos": n_unicos,
                                "falhos": _focos_uc.attrs.get("blocos_falhos", 0),
                                "total": _focos_uc.attrs.get("blocos_total", 0),
                            }
                            del _focos_uc
                        except Exception as _e_uc:
                            st.error(f"⚠️ Não foi possível concluir a análise: {_e_uc}")
                _res_uc = st.session_state.get(_uc_key)

            if not isinstance(_res_uc, dict):
                return

            areas_uc, pts_uc, n_jan = _res_uc["areas"], _res_uc["pts"], _res_uc["anos"]
            st.caption(f"Resultado abaixo: últimos **{n_jan} anos** (janelas de 12 meses até hoje).")
            if _res_uc.get("falhos"):
                st.warning(
                    f"⚠️ {_res_uc['falhos']} de {_res_uc['total']} janelas de busca no INPE falharam "
                    "mesmo após novas tentativas — a contagem de focos pode estar SUBESTIMADA. "
                    "Clique em 'Nova busca' e tente de novo em alguns minutos."
                )
            if areas_uc.empty:
                st.info(f"Nenhuma área de '{tipo_uc}' intersecta a região selecionada.")
            elif pts_uc.empty:
                st.success(
                    f"✅ {len(areas_uc)} área(s) na região e **nenhum foco** dentro delas "
                    "no período (ou o serviço do INPE está instável agora)."
                )
            else:
                resumo = _resumo_por_area(pts_uc, n_jan).join(
                    areas_uc.set_index("area_id")[[c for c in ("nome_area", "grupo", "categoria") if c in areas_uc.columns]]
                )
                resumo["mes_pico_txt"] = resumo["mes_pico"].map(lambda m: MESES_PT[int(m) - 1])
                n_com_fogo = len(resumo)
                n_recorrentes = int((resumo["recorrencia"] == "Recorrente").sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Áreas na região", len(areas_uc))
                c2.metric("Áreas com fogo", n_com_fogo, f"{n_com_fogo / len(areas_uc) * 100:.0f}% do total", delta_color="off")
                c3.metric("Focos dentro delas", f"{_res_uc['n_unicos']:,}".replace(",", "."),
                          f"{_res_uc['n_unicos'] / max(_res_uc['n_focos_regiao'], 1) * 100:.0f}% dos focos da região", delta_color="off")
                c4.metric(f"Queimaram nos {n_jan} anos", n_recorrentes, "recorrentes", delta_color="off")

                # --- Quantas vezes queima por ano -----------------------------------
                hoje_d = pd.Timestamp(hoje.date())
                rotulos_jan = {
                    j: f"{(hoje_d - pd.Timedelta(days=365 * (j + 1))):%m/%Y}–{(hoje_d - pd.Timedelta(days=365 * j)):%m/%Y}"
                    for j in range(n_jan)
                }
                por_jan = pts_uc.groupby("janela").size().reindex(range(n_jan), fill_value=0)
                df_jan = pd.DataFrame({
                    "Período": [rotulos_jan[j] for j in range(n_jan)][::-1],
                    "Focos": por_jan.to_numpy()[::-1],
                })
                por_mes = pts_uc["data"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
                mes_top = int(por_mes.idxmax())
                df_mes = pd.DataFrame({"Mês": MESES_PT, "Focos": por_mes.to_numpy()})
                df_mes["destaque"] = np.where(np.arange(1, 13) == mes_top, "Mês de pico", "Demais meses")

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**📅 Focos por ano (janelas de 12 meses)**")
                    fig_jan = px.bar(df_jan, x="Período", y="Focos", text="Focos")
                    fig_jan.update_traces(marker_color="#e67e22")
                    fig_jan.update_layout(height=340, margin=dict(t=20, b=20))
                    figura(fig_jan, "Focos de calor por ano em unidades de conservação", "INPE (BDQueimadas)")
                with col_g2:
                    st.markdown("**🗓️ Em que mês o fogo se concentra**")
                    fig_mes = px.bar(
                        df_mes, x="Mês", y="Focos", color="destaque",
                        color_discrete_map={"Mês de pico": "#c0392b", "Demais meses": "#7f8c8d"},
                    )
                    fig_mes.update_layout(
                        height=340, margin=dict(t=20, b=20), showlegend=False,
                    )
                    figura(fig_mes, "Concentração mensal dos focos de calor em unidades de conservação", "INPE (BDQueimadas)")
                st.info(
                    f"🔥 O mês de pico é **{MESES_PT[mes_top - 1]}**, com "
                    f"**{por_mes[mes_top] / por_mes.sum() * 100:.0f}%** dos focos dentro das áreas protegidas."
                )

                # --- Ranking e mês de pico por área ---------------------------------
                st.markdown("---")
                topo = resumo.head(15)
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    st.markdown(f"**🏆 Top {len(topo)} áreas com mais focos**")
                    fig_rank = px.bar(
                        topo.reset_index(), x="focos", y="nome_area", orientation="h",
                        color="recorrencia", text="focos",
                        color_discrete_map={"Recorrente": "#c0392b", "Frequente": "#e67e22", "Ocasional": "#f1c40f"},
                        labels={"focos": "Focos", "nome_area": "", "recorrencia": "Recorrência"},
                    )
                    fig_rank.update_layout(
                        height=460, margin=dict(t=20, b=20),
                        yaxis={"categoryorder": "total ascending"},
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    figura(fig_rank, "Unidades de conservação com maior número de focos de calor", "INPE (BDQueimadas)")
                with col_r2:
                    st.markdown("**🔥 Mês de pico de cada uma (focos por mês)**")
                    mapa_calor = (
                        pts_uc[pts_uc["area_id"].isin(topo.index)]
                        .assign(mes=lambda d: d["data"].dt.month)
                        .groupby(["area_id", "mes"]).size().unstack(fill_value=0)
                        .reindex(index=topo.index, columns=range(1, 13), fill_value=0)
                    )
                    mapa_calor.index = topo["nome_area"].to_numpy()
                    mapa_calor.columns = MESES_PT
                    fig_calor = px.imshow(
                        mapa_calor, aspect="auto", color_continuous_scale="YlOrRd",
                        labels={"color": "Focos"},
                    )
                    fig_calor.update_layout(height=460, margin=dict(t=20, b=20))
                    figura(fig_calor, "Mês de pico de focos de calor em cada unidade de conservação", "INPE (BDQueimadas)")

                # --- Recorrência --------------------------------------------------
                st.markdown("---")
                st.markdown("**🔁 Recorrência: o fogo volta?**")
                st.caption(
                    f"**Recorrente** = houve fogo em todos os {n_jan} anos; **Frequente** = em pelo menos "
                    f"metade; **Ocasional** = em menos que isso. **Locais reincidentes** = % dos "
                    "pontos de ~1 km que queimaram em mais de um ano (fogo que volta ao mesmo lugar)."
                )
                tabela = resumo.reset_index(drop=True)[[
                    c for c in ("nome_area", "categoria", "grupo", "focos", "anos_com_fogo", "dias_com_foco",
                                "mes_pico_txt", "pct_reincidente", "recorrencia") if c in resumo.columns
                ]].rename(columns={
                    "nome_area": "Área", "categoria": "Categoria", "grupo": "Grupo", "focos": "Focos", "anos_com_fogo": f"Anos com fogo (de {n_jan})",
                    "dias_com_foco": "Dias com foco", "mes_pico_txt": "Mês de pico",
                    "pct_reincidente": "Locais reincidentes (%)", "recorrencia": "Recorrência",
                })
                tabela["Locais reincidentes (%)"] = tabela["Locais reincidentes (%)"].round(1)
                tabela_bc("Recorrência do fogo por unidade de conservação", tabela, hide_index=True, height=380, use_container_width=True)
                st.download_button(
                    "📄 Baixar CSV — Recorrência por área", data=tabela.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"uc_recorrencia_{val_sel}_{hoje.strftime('%Y%m%d')}.csv", mime="text/csv",
                )

                # --- Detalhe de uma área ------------------------------------------
                st.markdown("---")
                st.markdown("**🔍 Detalhe de uma área**")
                ids = resumo.index.tolist()
                id_sel = st.selectbox(
                    "Escolha a área:", ids, key=f"det_uc_{val_sel}_{tipo_uc}_{grupo_uc}",
                    format_func=lambda i: f"{resumo.loc[i, 'nome_area']} ({int(resumo.loc[i, 'focos'])} focos)",
                )
                pts_a = pts_uc[pts_uc["area_id"] == id_sel]
                lin = resumo.loc[id_sel]
                st.write(
                    f"**{lin['nome_area']}** — {int(lin['focos'])} focos em {int(lin['dias_com_foco'])} dias "
                    f"diferentes; fogo em **{int(lin['anos_com_fogo'])} de {n_jan} anos** "
                    f"({lin['recorrencia'].lower()}); mês de pico: **{lin['mes_pico_txt']}**."
                )
                col_d1, col_d2 = st.columns(2)
                mes_a = pts_a["data"].dt.month.value_counts().reindex(range(1, 13), fill_value=0)
                jan_a = pts_a.groupby("janela").size().reindex(range(n_jan), fill_value=0)
                with col_d1:
                    fig_da = px.bar(x=MESES_PT, y=mes_a.to_numpy(), labels={"x": "Mês", "y": "Focos"})
                    fig_da.update_traces(marker_color="#c0392b")
                    fig_da.update_layout(height=280, margin=dict(t=10, b=10))
                    figura(fig_da, "Focos de calor por mês na primeira área comparada", "INPE (BDQueimadas)")
                with col_d2:
                    fig_db = px.bar(
                        x=[rotulos_jan[j] for j in range(n_jan)][::-1], y=jan_a.to_numpy()[::-1],
                        labels={"x": "Período", "y": "Focos"},
                    )
                    fig_db.update_traces(marker_color="#e67e22")
                    fig_db.update_layout(height=280, margin=dict(t=10, b=10))
                    figura(fig_db, "Focos de calor por período na segunda área comparada", "INPE (BDQueimadas)")

                # --- Mapa ---------------------------------------------------------
                st.markdown("---")
                st.markdown("**🗺️ Mapa: áreas protegidas coloridas pelo total de focos**")
                gdf_uc_mapa = areas_uc.merge(
                    resumo.reset_index()[["area_id", "focos", "anos_com_fogo", "mes_pico_txt", "recorrencia"]],
                    on="area_id", how="inner",
                )
                cortes = np.unique(np.quantile(gdf_uc_mapa["focos"], [0.2, 0.4, 0.6, 0.8]))
                paleta = ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"]
                gdf_uc_mapa["cor"] = [
                    paleta[min(int(np.searchsorted(cortes, v, side="right")), len(paleta) - 1)]
                    for v in gdf_uc_mapa["focos"]
                ]
                centro_uc = limite_mapa.geometry.union_all().centroid
                m_uc = folium.Map(
                    location=[centro_uc.y, centro_uc.x], zoom_start=6,
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri", prefer_canvas=True,
                )
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri Ref", overlay=True, control=False,
                ).add_to(m_uc)
                folium.GeoJson(
                    limite_mapa.__geo_interface__, name="Região selecionada",
                    style_function=lambda x: {
                        "fillColor": "#e67e22", "fillOpacity": 0.02, "color": "#e67e22",
                        "weight": 2.5, "dashArray": "6 3",
                    },
                ).add_to(m_uc)
                folium.GeoJson(
                    gdf_uc_mapa[["nome_area", "focos", "anos_com_fogo", "mes_pico_txt", "recorrencia", "cor", "geometry"]].__geo_interface__,
                    name="Áreas protegidas com fogo",
                    style_function=lambda f: {
                        "fillColor": f["properties"]["cor"], "fillOpacity": 0.7,
                        "color": "#7f8c8d", "weight": 1,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["nome_area", "focos", "anos_com_fogo", "mes_pico_txt", "recorrencia"],
                        aliases=["Área:", "Focos:", "Anos com fogo:", "Mês de pico:", "Recorrência:"],
                    ),
                ).add_to(m_uc)
                b_uc = limite_mapa.geometry.total_bounds
                m_uc.fit_bounds([[b_uc[1], b_uc[0]], [b_uc[3], b_uc[2]]])
                m_uc.get_root().html.add_child(folium.Element(
                    """<div style="position:fixed; bottom:28px; left:12px; z-index:9999;
                    background:rgba(255,255,255,0.95); padding:10px 14px; border-radius:8px;
                    font-size:12px; color:#2c3e50; line-height:1.9; border:1px solid rgba(0,0,0,0.08);
                    box-shadow:0 1px 4px rgba(0,0,0,0.18);"><b>🌿 Focos por área</b><br>
                    <span style="background:linear-gradient(to right,#ffffb2,#fecc5c,#fd8d3c,#f03b20,#bd0026);
                    display:inline-block;width:130px;height:10px;border-radius:4px;"></span><br>
                    <span>Menos</span><span style="float:right;">Mais</span></div>"""
                ))
                st_folium(
                    m_uc, width=None, height=600, returned_objects=[],
                    key=f"mapa_uc_{val_sel}_{tipo_uc}_{grupo_uc}_{n_jan}_{len(gdf_uc_mapa)}",
                )

                with st.expander("📚 Sobre os dados e limitações"):
                    st.markdown(
                        "- **Focos de calor** (satélites de referência do INPE) são pixels de ~375 m a "
                        "1 km que detectam fogo ativo; não medem área queimada nem a gravidade.\n"
                        "- Os limites das áreas protegidas (geobr) são **simplificados em ~1 km** para "
                        "caber na memória: focos muito próximos da divisa podem cair do lado errado.\n"
                        "- Quando duas áreas **se sobrepõem** (ex.: uma APA e um parque), o mesmo foco "
                        "conta para as duas; os totais gerais contam cada foco uma vez só.\n"
                        "- Cada **ano** é uma janela de 12 meses contada a partir de hoje, então todos "
                        "têm o mesmo tamanho; o ano mais recente pode estar incompleto no INPE.\n"
                        "- **Recorrência** mede se há fogo em anos diferentes, não se o fogo foi "
                        "criminoso ou natural: parte do fogo no Cerrado e no Pantanal é manejo e "
                        "faz parte da dinâmica do bioma.\n"
                        "- Para Unidades de Conservação, o filtro de grupo (Proteção Integral x Uso "
                        "Sustentável) e a categoria vêm da base do geobr/CNUC."
                    )

            st.markdown("")
            if st.button("🔄 Nova Busca nesta aba", key=f"btn_uc_reset_{val_sel}_{tipo_uc}_{grupo_uc}"):
                st.session_state[_uc_key] = None
                st.rerun(scope="fragment")

        with aba_uc:
            _aba_uc()

        # ----------------------------------------------------------
        # ABA — PLANEJAMENTO DE BRIGADAS (calendário do fogo)
        # ----------------------------------------------------------
        @st.fragment
        def _aba_brigadas():
            st.subheader("🚒 Planejamento de Brigadas — Calendário do Fogo")
            st.caption(
                "Para cada município da região selecionada: em que meses o fogo se concentra, "
                "em quantos anos ele aparece e quanto queima em média. Base: série histórica de "
                "área queimada MODIS (MCD64A1, NASA), disponível desde 2001."
            )
            # Último ano completo: o MODIS publica cada mês com ~2 a 3 meses de atraso
            ultimo_ano = hoje.year - 1 if hoje.month >= 4 else hoje.year - 2
            anos_br = st.select_slider(
                "Período do histórico:", options=list(range(2001, ultimo_ano + 1)),
                value=(ultimo_ano - 9, ultimo_ano), key="anos_brigadas",
                help="Quanto mais anos, mais confiável o calendário. A primeira consulta de cada "
                     "ano é feita no Earth Engine; depois fica guardada e sai na hora.",
            )
            anos_lista = list(range(anos_br[0], anos_br[1] + 1))
            if tipo_analise == "Por Bioma":
                st.warning(
                    "⏱️ Bioma inteiro: centenas de municípios. A primeira consulta pode levar "
                    "alguns minutos. Não feche a página."
                )

            _br_key = f"brig_res_{val_sel}"
            _res = st.session_state.get(_br_key)
            ja = isinstance(_res, dict)
            mudou = ja and _res.get("anos") != anos_br

            if (not ja) or mudou:
                if st.button(
                    "🔄 Atualizar com este período" if mudou else "🗓️ Montar Calendário do Fogo",
                    type="primary", use_container_width=True, key=f"btn_brig_{val_sel}",
                ):
                    try:
                        with st.spinner("🗺️ Carregando os municípios da região..."):
                            munis = obter_municipios_regiao(
                                tipo_analise, estado_dd, bioma_dd, municipio_dd, max_municipios=6000
                            )
                        if munis.empty:
                            st.error("Nenhum município encontrado para a região selecionada.")
                        else:
                            barra = st.progress(0.0, text="Consultando a série MODIS no Earth Engine...")
                            escala = 1000 if tipo_analise == "Por Bioma" else 500
                            df_longo, falhos = buscar_calendario_fogo(munis, anos_lista, escala, barra)
                            barra.empty()
                            st.session_state[_br_key] = {
                                "munis": munis[["name_muni", "abbrev_state"]].reset_index(drop=True),
                                "df": df_longo, "anos": anos_br, "falhos": falhos, "escala": escala,
                            }
                    except Exception as _e_br:
                        st.error(f"⚠️ Não foi possível montar o calendário: {_e_br}")
                    _res = st.session_state.get(_br_key)

            if not isinstance(_res, dict) or _res.get("anos") != anos_br:
                return

            df_longo, munis, falhos = _res["df"], _res["munis"], _res["falhos"]
            n_anos = len(anos_lista) - len(falhos)
            if falhos:
                st.warning(
                    f"⚠️ Não foi possível consultar {len(falhos)} ano(s) no Earth Engine "
                    f"({', '.join(map(str, falhos))}); o calendário usa os outros {n_anos}. "
                    "Clique em 'Nova Busca nesta aba' para tentar de novo."
                )
            if df_longo.empty or n_anos == 0:
                st.info("Sem dados MODIS para a região e o período escolhidos.")
                return

            resumo, perfil = resumo_calendario(df_longo, n_anos)
            resumo = resumo.join(munis, how="left")
            perfil_reg = perfil.sum(axis=0).to_numpy()
            ini_r, fim_r, frac_r = janela_critica(perfil_reg)
            com_fogo = resumo[resumo["media_anual_km2"] > 0]
            if com_fogo.empty or ini_r is None:
                st.success(f"✅ Nenhuma área queimada detectada pelo MODIS na região entre {anos_br[0]} e {anos_br[1]}.")
                return

            def _txt_jan(i, f):
                return MESES_PT[i - 1] if i == f else f"{MESES_PT[i - 1]}–{MESES_PT[f - 1]}"

            metade = int(np.ceil(n_anos / 2))
            mes_pico_r = int(np.argmax(perfil_reg)) + 1
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Municípios analisados", len(resumo))
            c2.metric(f"Com fogo em {metade}+ dos {n_anos} anos", int((resumo["anos_com_fogo"] >= metade).sum()))
            c3.metric("Mês de pico da região", MESES_PT[mes_pico_r - 1])
            c4.metric("Janela crítica da região", _txt_jan(ini_r, fim_r),
                      f"{frac_r * 100:.0f}% da área queimada", delta_color="off")
            mes_mobil = MESES_PT[(ini_r - 2) % 12]
            st.info(
                f"📌 **Leitura para o planejamento:** na região, {frac_r * 100:.0f}% da área queimada "
                f"de {anos_br[0]} a {anos_br[1]} se concentrou entre **{_txt_jan(ini_r, fim_r)}**. "
                f"Brigadas e ações de prevenção (aceiros, queima prescrita, campanhas) devem estar "
                f"prontas até **{mes_mobil}**, o mês anterior ao início da janela. "
                "A tabela abaixo mostra a janela de cada município, que pode ser diferente da regional."
            )

            # --- Calendário da região ------------------------------------------
            meses_jan = set(((ini_r - 1 + k) % 12) + 1 for k in range(((fim_r - ini_r) % 12) + 1))
            df_reg = pd.DataFrame({
                "Mês": MESES_PT, "Área (km²)": perfil_reg.round(1),
                "Período": ["Janela crítica" if m in meses_jan else "Demais meses" for m in range(1, 13)],
            })
            fig_reg = px.bar(
                df_reg, x="Mês", y="Área (km²)", color="Período",
                color_discrete_map={"Janela crítica": "#c0392b", "Demais meses": "#bdbdbd"},
                labels={"Área (km²)": "Área queimada média (km²)"},
            )
            fig_reg.update_layout(height=320, legend=dict(orientation="h", y=1.08, title=""))
            figura(fig_reg, f"Área queimada média mensal na região de {anos_br[0]} a {anos_br[1]}, "
                            "com destaque para a janela crítica", "MODIS MCD64A1 (NASA)")

            # --- Mapa de calor município × mês ---------------------------------
            n_max = len(com_fogo)
            n_top = n_max if n_max <= 10 else st.slider(
                "Municípios no mapa de calor (os de maior área queimada):", 10, min(60, n_max),
                min(25, n_max), key="n_top_brigadas",
            )
            topo = com_fogo.sort_values("media_anual_km2", ascending=False).head(n_top)
            rotulos = [f"{r.name_muni} ({r.abbrev_state})" for r in topo.itertuples()]
            fig_cal = px.imshow(
                perfil.loc[topo.index].to_numpy(), x=MESES_PT, y=rotulos,
                color_continuous_scale="Reds", aspect="auto",
                labels={"x": "Mês", "y": "", "color": "km²/mês"},
            )
            fig_cal.update_layout(height=max(320, 22 * n_top + 80))
            figura(fig_cal, "Área queimada média por mês nos municípios com maior área queimada (km²)",
                   "MODIS MCD64A1 (NASA)")

            # --- Tabela por município -------------------------------------------
            tab = com_fogo.sort_values("media_anual_km2", ascending=False)
            tab_out = pd.DataFrame({
                "Município": tab["name_muni"],
                "UF": tab["abbrev_state"],
                "Área queimada média anual (km²)": tab["media_anual_km2"].round(1),
                "Anos com fogo": [f"{int(a)} de {n_anos}" for a in tab["anos_com_fogo"]],
                "Mês de pico": [MESES_PT[int(m) - 1] for m in tab["mes_pico"]],
                "Janela crítica (80%)": [_txt_jan(int(i), int(f)) for i, f in zip(tab["ini"], tab["fim"])],
                "Área na janela (%)": tab["pct_janela"].round(0),
                "Brigada pronta até": [MESES_PT[(int(i) - 2) % 12] for i in tab["ini"]],
            })
            tabela_bc(
                f"Calendário do fogo por município, de {anos_br[0]} a {anos_br[1]}",
                tab_out, fonte="MODIS MCD64A1 (NASA); resultados originais do estudo",
                hide_index=True, height=420, use_container_width=True,
            )
            st.download_button(
                "📄 Baixar CSV — Calendário do fogo",
                data=tab_out.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"calendario_fogo_{val_sel}_{anos_br[0]}_{anos_br[1]}.csv", mime="text/csv",
                key=f"dl_brig_{val_sel}",
            )

            with st.expander("ℹ️ Como o calendário é calculado e limitações"):
                st.markdown(
                    "- **Área queimada mensal:** soma, dentro de cada município, dos pixels que o "
                    f"MODIS MCD64A1 marcou como queimados no mês (resolução de cálculo: {_res['escala']} m).\n"
                    "- **Janela crítica:** a menor sequência de meses seguidos que concentra pelo "
                    "menos 80% da área queimada média do ano. Pode virar o ano (ex.: Dez–Fev, "
                    "comum em Roraima).\n"
                    "- **Anos com fogo:** em quantos anos do período o município teve alguma área "
                    "queimada detectada.\n"
                    "- **Brigada pronta até:** o mês anterior ao início da janela crítica.\n"
                    "- **Limitações:** o MODIS não detecta bem queimadas menores que cerca de "
                    "20 a 25 ha nem o fogo rasteiro sob a copa da floresta, então a área real "
                    "tende a ser maior. Municípios que só têm fogo pequeno podem aparecer sem fogo."
                )

            st.markdown("")
            if st.button("🔄 Nova Busca nesta aba", key=f"btn_brig_reset_{val_sel}"):
                st.session_state[_br_key] = None
                st.rerun(scope="fragment")

        with aba_brigadas:
            _aba_brigadas()

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
        # ABA — RISCO PREDITIVO (modelo em grade: célula × semana)
        # ----------------------------------------------------------
        _log_mem("Abas renderizadas (antes do Risco)")

        def _classe_risco_grade(p):
            if p >= 0.5:
                return "Muito alto", "#bd0026"
            if p >= 0.3:
                return "Alto", "#f03b20"
            if p >= 0.1:
                return "Moderado", "#feb24c"
            return "Baixo", "#ffffb2"

        @st.fragment
        def _aba_risco_grade():
            st.subheader("🎯 Risco Preditivo — onde o fogo deve aparecer na próxima semana")
            st.caption(
                "A região é dividida em células (quadrados de 5 a 50 km, conforme o tamanho da "
                "região) acompanhadas semana a semana. O modelo aprende com os anos anteriores "
                "quais condições antecedem o fogo (fogo recente, chuva, umidade, temperatura, "
                "vegetação e uso do solo) e estima, para cada célula, a chance de haver pelo "
                "menos um foco de calor na próxima semana."
            )
            st.info(
                "📅 Esta aba não usa o ano/mês da barra lateral: ela treina com os últimos anos até "
                "hoje e prevê a próxima semana. A região é a da barra lateral."
            )
            anos_r = st.select_slider(
                "Anos de histórico:", options=[2, 3, 4, 5], value=3,
                format_func=lambda x: f"{x} anos", key="anos_risco_grade",
                help="O último ano fica separado como TESTE (o modelo não o vê no treino). "
                     "Com 3 anos: treina em 2 e testa no mais recente.",
            )
            if tipo_analise == "Por Bioma":
                st.warning(
                    "⏱️ Bioma inteiro: a primeira execução pode levar vários minutos (focos de "
                    "vários anos e clima de milhares de células). Depois fica guardada em cache."
                )

            _rk = f"risco_grade_{val_sel}"
            _res = st.session_state.get(_rk)
            ja = isinstance(_res, dict)
            mudou = ja and _res.get("anos") != anos_r

            if (not ja) or mudou:
                if st.button(
                    "🔄 Atualizar com este histórico" if mudou else "🎯 Treinar modelo e gerar mapa de risco",
                    type="primary", use_container_width=True, key=f"btn_risco_grade_{val_sel}",
                ):
                    avisos = []
                    try:
                        barra = st.progress(0.0, text="Montando a grade da região...")
                        grade, res_g = montar_grade(limite)
                        rets = _retangulos_grade(grade, res_g)
                        ultima = min(_ultima_data_era5(), pd.Timestamp(hoje.date()) - pd.Timedelta(days=1))
                        k_ult = int(((ultima - EPOCA_SEMANAS).days - 6) // 7)
                        k_alvo_ini = k_ult - 52 * anos_r + 1

                        # 1) Focos INPE, um período por ano civil (anos completos ficam em cache)
                        d_lo = _data_semana(k_alvo_ini - 54)
                        d_hi = _data_semana(k_ult) + pd.Timedelta(days=6)
                        periodos, cur = [], d_lo
                        while cur <= d_hi:
                            fim = min(pd.Timestamp(cur.year, 12, 31), d_hi)
                            periodos.append((cur, fim))
                            cur = fim + pd.Timedelta(days=1)
                        args_f = [(tipo_analise, estado_dd, bioma_dd, municipio_dd,
                                   a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), res_g) for a, b in periodos]
                        res_f, err_f = _executar_paralelo(_focos_semanais_periodo, args_f, max_workers=2,
                                                          progresso=barra, texto="Focos do INPE (por ano)")
                        if any(r is None for r in res_f):
                            raise RuntimeError(
                                "O serviço do INPE não respondeu para todo o período "
                                f"({'; '.join(err_f[:2])}). Tente de novo em alguns minutos."
                            )
                        cnt_focos = pd.concat([r[0] for r in res_f], ignore_index=True)
                        falhos = sum(r[1] for r in res_f)
                        if falhos:
                            avisos.append(f"{falhos} janela(s) de busca no INPE falharam; a contagem de focos "
                                          "pode estar levemente subestimada.")

                        # 2) Clima ERA5-Land em blocos de ~3 meses (blocos fechados ficam em cache)
                        escala_clima = int(min(11132, res_g * 111000 / 3))
                        k_c0 = k_alvo_ini - 30
                        b0 = k_c0 - (k_c0 % BLOCO_SEMANAS)
                        args_c = [(rets, b, min(BLOCO_SEMANAS, k_ult - b + 1), escala_clima)
                                  for b in range(b0, k_ult + 1, BLOCO_SEMANAS)]
                        res_c, err_c = _executar_paralelo(_clima_grade_bloco, args_c, max_workers=4,
                                                          progresso=barra, texto="Clima ERA5-Land (blocos de 3 meses)")
                        n_falhos_c = sum(r is None for r in res_c)
                        if n_falhos_c > 0.2 * len(res_c):
                            raise RuntimeError(f"O Earth Engine não devolveu o clima ({'; '.join(err_c[:2])}).")
                        if n_falhos_c:
                            avisos.append(f"{n_falhos_c} bloco(s) de clima falharam; essas semanas ficam sem clima.")
                        clima = pd.concat([r for r in res_c if r is not None], ignore_index=True)

                        # 3) NDVI mensal por ano
                        ano_ndvi0 = (_data_semana(k_alvo_ini) - pd.Timedelta(days=40)).year
                        args_n = [(rets, a, 12 if a < ultima.year else max(1, ultima.month - 1))
                                  for a in range(ano_ndvi0, ultima.year + 1)]
                        res_n, _ = _executar_paralelo(_ndvi_grade_ano, args_n, max_workers=4,
                                                      progresso=barra, texto="NDVI MODIS (por ano)")
                        ndvi = pd.concat([r for r in res_n if r is not None], ignore_index=True) \
                            if any(r is not None for r in res_n) else None
                        if ndvi is None:
                            avisos.append("NDVI indisponível: o modelo foi treinado sem essa variável.")

                        # 4) Uso do solo MapBiomas
                        barra.progress(1.0, text="Uso do solo (MapBiomas)...")
                        try:
                            uso = _mapbiomas_grade(rets)
                        except Exception as e_uso:
                            uso = None
                            avisos.append(f"MapBiomas indisponível ({str(e_uso)[:120]}): treinado sem uso do solo.")

                        # 5) Tabela célula × semana, treino, teste e previsão
                        barra.progress(1.0, text="Treinando e testando os modelos (pode levar 1 a 3 minutos)...")
                        df_grade = montar_tabela_grade(grade, k_alvo_ini, k_ult, cnt_focos, clima, ndvi, uso)
                        del clima, ndvi, cnt_focos
                        resultado = treinar_avaliar_grade(df_grade)
                        del df_grade
                        _liberar_memoria()

                        # 6) Células -> municípios (para o ranking e o tooltip)
                        munis = obter_municipios_regiao(tipo_analise, estado_dd, bioma_dd, municipio_dd,
                                                        max_municipios=6000)
                        pares = gpd.sjoin(grade[["cid", "geometry"]], munis[["name_muni", "abbrev_state", "geometry"]],
                                          predicate="intersects", how="inner")[["cid", "name_muni", "abbrev_state"]]
                        centro = gpd.GeoDataFrame(grade[["cid"]], geometry=gpd.points_from_xy(grade["lon_c"], grade["lat_c"]),
                                                  crs="EPSG:4326")
                        nome_centro = gpd.sjoin(centro, munis[["name_muni", "geometry"]], predicate="within", how="left") \
                            .drop_duplicates("cid").set_index("cid")["name_muni"]
                        barra.empty()

                        st.session_state[_rk] = {
                            "anos": anos_r, "res": res_g, "ultima": ultima, "avisos": avisos,
                            "grade": grade[["cid", "geometry"]].assign(municipio=grade["cid"].map(nome_centro).fillna("—")),
                            "pares": pares.reset_index(drop=True), "r": resultado,
                        }
                    except Exception as _e_rg:
                        st.error(f"⚠️ Não foi possível treinar o modelo: {_e_rg}")
                        with st.expander("Detalhes técnicos do erro"):
                            st.code(traceback.format_exc(), language="python")
                    _res = st.session_state.get(_rk)

            if not isinstance(_res, dict) or _res.get("anos") != anos_r:
                return

            r, m = _res["r"], _res["r"]["metricas"]
            for aviso in _res["avisos"]:
                st.warning(f"⚠️ {aviso}")
            ini_prev = _data_semana(r["semana_prevista"])
            fim_prev = ini_prev + pd.Timedelta(days=6)
            t0, t1 = (_data_semana(m["semanas_teste"][0]),
                      _data_semana(m["semanas_teste"][1]) + pd.Timedelta(days=6))
            tr0, tr1 = (_data_semana(m["semanas_treino"][0]),
                        _data_semana(m["semanas_treino"][1]) + pd.Timedelta(days=6))
            st.markdown(
                f"**📅 Previsão para a semana de {ini_prev:%d/%m} a {fim_prev:%d/%m/%Y}** "
                f"(clima ERA5-Land disponível até {_res['ultima']:%d/%m/%Y}). "
                f"Células de {_res['res']}° (~{_res['res'] * 111:.0f} km)."
            )
            if ini_prev <= pd.Timestamp(hoje.date()):
                st.caption(
                    "ℹ️ A semana prevista já começou: o ERA5-Land é publicado com alguns dias de "
                    "atraso, então a previsão usa o clima da última semana completa disponível."
                )

            # --- Resultado no ano de teste ------------------------------------
            curvas = r["curvas"]
            cap_m = float(curvas.loc[curvas["k"] == 10, "Modelo"].iloc[0])
            cap_r = float(curvas.loc[curvas["k"] == 10, "Regra prática (fogo recente)"].iloc[0])
            cap_h = float(curvas.loc[curvas["k"] == 10, "Histórico do último ano"].iloc[0])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AUC-ROC no teste", f"{m['auc']:.3f}",
                      help="Capacidade de ordenar células com e sem fogo (0,5 = sorteio; 1 = perfeito).")
            c2.metric("PR-AUC no teste", f"{m['pr_auc']:.3f}", f"regra prática: {m['pr_auc_regra']:.3f}",
                      delta_color="off",
                      help=f"Mais exigente que o AUC quando o fogo é raro. Sorteio daria {m['taxa_base']:.3f} "
                           "(a proporção de células-semana com fogo).")
            c3.metric("Fogo capturado nas 10% células priorizadas", f"{cap_m:.0f}%",
                      f"regra prática: {cap_r:.0f}%", delta_color="off")
            c4.metric("AUC espacial", f"{m['auc_espacial']:.3f}" if m["auc_espacial"] else "—",
                      help="Validação em blocos de ~110 km que o modelo nunca viu no treino.")
            ganho = cap_m - cap_r
            leitura = (
                f"Se a cada semana de {t0:%m/%Y} a {t1:%m/%Y} as equipes fossem concentradas nas **10% "
                f"células de maior risco** apontadas pelo modelo, elas estariam onde ocorreram **{cap_m:.0f}%** "
                f"dos registros de fogo. Pela regra prática (onde houve fogo recente) seriam {cap_r:.0f}%, "
                f"pelo histórico do último ano {cap_h:.0f}% e, sem critério, 10%."
            )
            if ganho >= 1:
                st.success(f"🟢 {leitura} **O modelo agrega valor sobre a regra prática (+{ganho:.0f} pontos).**")
            elif ganho > -1:
                st.warning(f"🟡 {leitura} **O modelo empata com a regra prática neste recorte**: o fogo recente "
                           "já explica quase tudo; o ganho do modelo está em apontar áreas sem fogo recente.")
            else:
                st.error(f"🔴 {leitura} **Neste recorte a regra prática foi melhor que o modelo ({ganho:.0f} pontos).**")

            # --- Mapa de risco -------------------------------------------------
            prob = r["previsao"].set_index("cid")["prob"]
            g = _res["grade"].copy()
            g["prob"] = g["cid"].map(prob).fillna(0.0)
            classes = g["prob"].map(_classe_risco_grade)
            g["classe"] = [c[0] for c in classes]
            g["cor"] = [c[1] for c in classes]
            g["prob_txt"] = (g["prob"] * 100).round(0).astype(int).astype(str) + "%"
            centro_m = limite.geometry.union_all().centroid
            m_rg = folium.Map(
                location=[centro_m.y, centro_m.x], zoom_start=6 if tipo_analise != "Por Município" else 9,
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", prefer_canvas=True,
            )
            folium.GeoJson(
                g[["municipio", "prob_txt", "classe", "cor", "geometry"]].to_json(),
                style_function=lambda f: {"fillColor": f["properties"]["cor"], "fillOpacity": 0.65,
                                          "color": f["properties"]["cor"], "weight": 0.3},
                tooltip=folium.GeoJsonTooltip(fields=["municipio", "prob_txt", "classe"],
                                              aliases=["Município:", "Chance de foco na semana:", "Risco:"]),
            ).add_to(m_rg)
            folium.GeoJson(
                limite.__geo_interface__,
                style_function=lambda x: {"fillColor": "transparent", "color": "#333333", "weight": 1.5},
            ).add_to(m_rg)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}",
                attr="Esri Ref", overlay=True, control=False,
            ).add_to(m_rg)
            m_rg.get_root().html.add_child(folium.Element("""
                <div style="position:fixed; bottom:28px; left:12px; z-index:9999;
                            background:rgba(255,255,255,0.95); padding:10px 14px; border-radius:8px;
                            font-size:12px; color:#2c3e50; line-height:1.9; border:1px solid rgba(0,0,0,0.08);
                            box-shadow:0 1px 4px rgba(0,0,0,0.18);">
                  <b style="font-size:13px;">Chance de foco na semana</b><br>
                  <span style="color:#bd0026;">■</span> Muito alto (≥ 50%)<br>
                  <span style="color:#f03b20;">■</span> Alto (30 a 50%)<br>
                  <span style="color:#feb24c;">■</span> Moderado (10 a 30%)<br>
                  <span style="color:#e6d97a;">■</span> Baixo (&lt; 10%)
                </div>"""))
            st_folium(m_rg, width=None, height=620, returned_objects=[],
                      key=f"mapa_risco_grade_{val_sel}_{anos_r}")

            # --- Ranking de municípios -----------------------------------------
            pares = _res["pares"].merge(g[["cid", "prob"]], on="cid")
            if not pares.empty:
                rk = pares.groupby(["name_muni", "abbrev_state"]).agg(
                    pmax=("prob", "max"), pmed=("prob", "mean"),
                    n_alto=("prob", lambda s: int((s >= 0.3).sum())), n_cel=("prob", "size"),
                ).reset_index().sort_values(["pmax", "pmed"], ascending=False).head(20)
                tab_rk = pd.DataFrame({
                    "Município": rk["name_muni"], "UF": rk["abbrev_state"],
                    "Chance máxima entre as células (%)": (rk["pmax"] * 100).round(0),
                    "Chance média das células (%)": (rk["pmed"] * 100).round(0),
                    "Células com risco alto ou muito alto": [f"{a} de {b}" for a, b in zip(rk["n_alto"], rk["n_cel"])],
                })
                tabela_bc(
                    f"Municípios com maior risco de foco de calor na semana de {ini_prev:%d/%m} a {fim_prev:%d/%m/%Y}",
                    tab_rk, fonte="Resultados originais do estudo", hide_index=True, use_container_width=True,
                )

            # --- Prova de valor: curva de captura no ano de teste ------------
            longo = curvas.melt(id_vars="k", var_name="Estratégia", value_name="Fogo capturado (%)")
            fig_cap = px.line(
                longo, x="k", y="Fogo capturado (%)", color="Estratégia",
                color_discrete_map={"Modelo": "#c0392b", "Regra prática (fogo recente)": "#2c3e50",
                                    "Histórico do último ano": "#7f8c8d", "Sem critério (sorteio)": "#bdbdbd"},
                labels={"k": "Células priorizadas a cada semana (%)"},
            )
            fig_cap.update_layout(height=380, legend=dict(orientation="h", y=-0.25, title=""))
            figura(fig_cap, f"Ocorrências de fogo capturadas conforme a porcentagem de células priorizadas, "
                            f"de {t0:%m/%Y} a {t1:%m/%Y} (período de teste)",
                   "Resultados originais do estudo, com focos do INPE")

            with st.expander("🔬 Detalhes do modelo: algoritmos, variáveis e calibração"):
                comp = r["comparacao"].copy()
                comp["Algoritmo"] = [f"⭐ {a}" if a == r["escolhido"] else a for a in comp["Algoritmo"]]
                tabela_bc(
                    "Comparação dos algoritmos no período de validação (escolha pelo maior PR-AUC)",
                    comp.round(3), fonte="Resultados originais do estudo", hide_index=True,
                    use_container_width=True,
                )
                imp = r["importancias"].dropna().sort_values("queda", ascending=True).tail(12)
                if not imp.empty:
                    imp["Variável"] = imp["variavel"].map(NOMES_FEATURES_GRADE).fillna(imp["variavel"])
                    fig_imp = px.bar(imp, x="queda", y="Variável", orientation="h",
                                     labels={"queda": "Queda do PR-AUC ao embaralhar a variável", "Variável": ""})
                    fig_imp.update_traces(marker_color="#c0392b")
                    fig_imp.update_layout(height=40 * len(imp) + 80)
                    figura(fig_imp, f"Importância das variáveis no modelo escolhido ({r['escolhido']})")
                cal = r["calibracao"]
                fig_cal = go.Figure()
                lim = float(max(cal["prevista"].max(), cal["observada"].max(), 0.05))
                fig_cal.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="Calibração perfeita",
                                             line=dict(color="#bdbdbd", dash="dash")))
                fig_cal.add_trace(go.Scatter(x=cal["prevista"], y=cal["observada"], mode="lines+markers",
                                             name="Modelo", line=dict(color="#c0392b")))
                fig_cal.update_layout(height=340, xaxis_title="Chance prevista (média da faixa)",
                                      yaxis_title="Frequência observada de fogo",
                                      legend=dict(orientation="h", y=-0.3, title=""))
                figura(fig_cal, "Calibração do modelo no período de teste, por faixa de chance prevista")
                def _mil(x):
                    return f"{int(x):,}".replace(",", ".")

                st.markdown(
                    f"- **Treino:** {tr0:%d/%m/%Y} a {tr1:%d/%m/%Y} — {_mil(m['n_treino'])} células-semana, "
                    f"{_mil(m['pos_treino'])} com fogo.\n"
                    f"- **Teste:** {t0:%d/%m/%Y} a {t1:%d/%m/%Y} — {_mil(m['n_teste'])} células-semana, "
                    f"{_mil(m['pos_teste'])} com fogo ({m['taxa_base'] * 100:.1f}%).\n"
                    f"- **Brier score no teste:** {m['brier']:.4f} (quanto menor, melhor)."
                )

            with st.expander("ℹ️ Como o modelo funciona e limitações"):
                st.markdown(
                    "- **Unidade:** cada célula da grade em cada semana (segunda a domingo). O alvo é "
                    "haver pelo menos um foco de calor do INPE (satélites de referência AQUA e NPP).\n"
                    "- **Sem vazamento:** as variáveis de uma semana usam só dados até a semana anterior.\n"
                    "- **Variáveis:** focos na semana anterior, nas 4 e nas 52 semanas anteriores e na mesma "
                    "época do ano anterior; chuva, umidade, temperatura máxima e vento (ERA5-Land); semanas "
                    "seguidas sem chuva; NDVI do mês anterior (MODIS); uso do solo (MapBiomas "
                    f"{ANO_MAPBIOMAS}); época do ano.\n"
                    "- **Validação:** o último ano é o teste, que o modelo não vê no treino; os 6 meses "
                    "anteriores servem para escolher o algoritmo; a validação espacial usa blocos de "
                    "~110 km. Depois da avaliação, o modelo é retreinado com todo o histórico para prever a "
                    "próxima semana.\n"
                    "- **Comparação:** a 'regra prática' prioriza onde houve fogo na semana anterior (depois "
                    "nas 4 semanas e no último ano), que é o que se faria sem modelo.\n"
                    "- **Limitações:** o ERA5-Land tem alguns dias de atraso; o uso do solo é de um ano só "
                    f"({ANO_MAPBIOMAS}); focos do INPE incluem falsos positivos e não medem área; células na "
                    "borda da região contam só os focos de dentro da região."
                )

            st.markdown("")
            if st.button("🔄 Nova Busca nesta aba", key=f"btn_risco_grade_reset_{val_sel}"):
                st.session_state[_rk] = None
                st.rerun(scope="fragment")

        with aba_risco:
            _aba_risco_grade()

else:
    st.info(
        "👈 Use os filtros ao lado para selecionar a Fonte de Dados, "
        "o local e o período de análise. Depois clique em **'Gerar Dashboard'**."
    )
