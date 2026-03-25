import streamlit as st
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from geobr import read_state, read_biomes, read_municipality 
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import requests, warnings, time, unicodedata, re

# Configuração da página
st.set_page_config(page_title="Monitor de Queimadas", page_icon="🔥", layout="wide")
warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# --- FUNÇÕES COM CACHE ---
@st.cache_data(ttl=86400)
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

@st.cache_data
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

@st.cache_data(ttl=3600)
def buscar_focos_inpe(tipo, val_estado, val_bioma, val_muni, d_ini, d_fim):
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
    
    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=5), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('AQUA_M-T','NPP-375','NPP-375D') AND pais_complete_id=33 AND {filtro_base}"
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

# Escala Geográfica
tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)

# Filtros de Localização (Lógica de inativação aplicada)
estado_dd = st.sidebar.selectbox('Selecione o Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25, disabled=(tipo_analise == 'Por Bioma'))

bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))

cidades_lista = buscar_cidades(estado_dd)
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', cidades_lista, disabled=(tipo_analise != 'Por Município'))

# Filtro de Mês Único (Visual de selectbox igual ao da cidade)
st.sidebar.markdown("---")
mes_selecionado = st.sidebar.selectbox(
    "Selecione o Mês de Análise:", 
    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    index=datetime.now().month - 1 # Sugere o mês atual por padrão
)

gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

# --- FONTE COM MAIS INFORMAÇÕES (SIDEBAR) ---
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Sobre os Dados"):
    st.markdown("""
    **Fonte Primária:** Programa Queimadas do Instituto Nacional de Pesquisas Espaciais (**INPE**).
    
    **Satélites Utilizados:**
    * **AQUA_M-T:** Referência para séries históricas.
    * **NPP-375:** Alta resolução espacial.
    
    **Metodologia:**
    Os focos representam detecções de calor na superfície. O sistema processa imagens orbitais para identificar anomalias térmicas vinculadas a possíveis queimadas.
    
    **Limites Geoespaciais:**
    Bases do IBGE via biblioteca `geobr`.
    """)

# --- INTERFACE (TELA PRINCIPAL) ---
st.title("🔥 Dashboard de Queimadas 🔥")

if gerar:
    # Definindo período (Último ano disponível para filtrar pelo mês escolhido)
    hoje = datetime.now()
    dt_ini = hoje - timedelta(days=365) 

    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.spinner(f"Consultando base do INPE para {val_sel}..."):
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        df_bruto = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"))

    if df_bruto.empty: 
        st.error("⚠️ Nenhum foco de calor detectado neste período/local pelo INPE.")
    else:
        # Filtragem pelo Mês selecionado
        df_bruto['mes_num'] = pd.to_datetime(df_bruto[next(c for c in df_bruto.columns if 'data' in c)]).dt.month
        df = df_bruto[df_bruto['mes_num'] == mes_selecionado]

        if df.empty:
            st.warning(f"⚠️ Não foram detectados focos no mês {mes_selecionado} dentro do período analisado.")
        else:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
            gdf = gpd.sjoin(gdf, limite, predicate="within")
            df_rec = pd.DataFrame(gdf.drop(columns="geometry"))

            if df_rec.empty: 
                st.error("⚠️ Os focos detectados estão fora do limite geográfico selecionado.")
            else:
                st.success(f"✅ **Sucesso:** {len(df_rec):,} focos validados em {val_sel} para o mês {mes_selecionado}.")
                
                col1, col2 = st.columns([1.3, 1])
                
                with col1:
                    st.subheader("🗺️ Mapa de Calor Espacial")
                    centro = limite.geometry.union_all().centroid
                    m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
                    folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
                    HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15, blur=20).add_to(m)
                    st_folium(m, width=700, height=500, returned_objects=[])
                
                with col2:
                    st.subheader("📊 Distribuição por Satélite")
                    # Gráfico para mostrar quais satélites detectaram mais focos
                    fig_sat = px.pie(df_rec, names='satelite', hole=.4, color_discrete_sequence=px.colors.sequential.Reds_r)
                    fig_sat.update_layout(showlegend=True)
                    st.plotly_chart(fig_sat, use_container_width=True)
                    
                    st.subheader("📈 Focos por Dia")
                    data_col = next(c for c in df_rec.columns if 'data' in c)
                    df_rec[data_col] = pd.to_datetime(df_rec[data_col])
                    df_g = df_rec.set_index(data_col).resample('D').size().reset_index(name='focos')
                    fig_line = px.line(df_g, x=data_col, y='focos', markers=True, line_shape='spline')
                    fig_line.update_traces(line_color='#e64a19')
                    fig_line.update_layout(template='plotly_dark', xaxis_title="Dia do Mês", yaxis_title="Nº de Focos")
                    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("👈 Configure os parâmetros ao lado e clique em 'Gerar Dashboard' para visualizar os dados geoprocessados.")
