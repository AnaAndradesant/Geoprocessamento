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
    menu_items={
        'About': """
        ### 🛰️ Monitor de Queimadas (INPE)
        Dashboard interativo para monitoramento de focos de calor e análise de risco em áreas protegidas.
        Desenvolvido por Ana Carolina Andrade.
        """
    }
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
    st.error(f"⚠️ Erro ao conectar com o Google Earth Engine.")

# Método para o Folium renderizar GEE com aviso de erro
def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=0.7):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        folium.raster_layers.TileLayer(
            tiles=map_id_dict['tile_fetcher'].url_format,
            attr='Map Data © Google Earth Engine',
            name=name,
            overlay=True,
            control=True,
            show=show,
            opacity=opacity
        ).add_to(self)
    except Exception as e:
        print(f"Erro ao adicionar camada EE no mapa: {e}") # Mostra o erro no terminal para ajudar no debug
        
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
    sat_str = "','".join(satelites)
    
    while dt_ini <= dt_fim:
        dt_bloco_fim = min(dt_ini + timedelta(days=5), dt_fim)
        cql = f"data_hora_gmt >= '{dt_ini.strftime('%Y-%m-%d')}T00:00:00' AND data_hora_gmt <= '{dt_bloco_fim.strftime('%Y-%m-%d')}T23:59:59' AND satelite IN ('{sat_str}') AND pais_complete_id=33 AND {filtro_base}"
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

tipo_analise = st.sidebar.radio('Escala Geográfica:', ['Por Estado', 'Por Bioma', 'Por Município'], index=2)

estado_dd = st.sidebar.selectbox('Selecione o Estado:', ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"], index=25, disabled=(tipo_analise == 'Por Bioma'))
bioma_dd = st.sidebar.selectbox('Selecione o Bioma:', ["Amazônia", "Cerrado", "Mata Atlântica", "Caatinga", "Pampa", "Pantanal"], disabled=(tipo_analise != 'Por Bioma'))
cidades_lista = buscar_cidades(estado_dd)
municipio_dd = st.sidebar.selectbox('Selecione a Cidade:', cidades_lista, disabled=(tipo_analise != 'Por Município'))

unidade_dd = st.sidebar.selectbox("Analisar por:", ["Dias", "Meses", "Anos"], index=1)
if unidade_dd == "Dias": op_qtd = list(range(1, 91))
elif unidade_dd == "Meses": op_qtd = list(range(1, 61))
else: op_qtd = list(range(1, 11))
quantidade_sel = st.sidebar.selectbox(f"Quantidade de {unidade_dd}:", options=op_qtd, index=1)

# --- FONTES DE DADOS DINÂMICAS ---
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Fontes de Dados")

ativar_inpe = st.sidebar.checkbox("🔥 Focos de Calor (INPE)", value=True)
satelites_sel = []
if ativar_inpe:
    satelites_lista = ['AQUA_M-T', 'NPP-375', 'NPP-375D', 'TERRA_M-T', 'NOAA-20', 'MSG-03']
    satelites_sel = st.sidebar.multiselect("Satélites de Referência:", satelites_lista, default=['AQUA_M-T', 'NPP-375', 'NPP-375D'])

st.sidebar.markdown("")

ativar_modis = st.sidebar.checkbox("🗺️ Cicatrizes (NASA MODIS)", value=True)
if ativar_modis:
    ano_modis = st.sidebar.selectbox("Ano (MODIS):", list(range(2001, datetime.now().year + 1)), index=datetime.now().year - 2002)
    mes_modis = st.sidebar.selectbox("Mês (MODIS):", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("🌳 Análise de Risco (Cruzamento Espacial):", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

# --- INTERFACE (TELA PRINCIPAL) ---
st.title("🔥 Dashboard de Queimadas 🔥")

if gerar:
    if not ativar_inpe and not ativar_modis:
        st.error("⚠️ Selecione pelo menos uma Fonte de Dados (INPE ou MODIS) na barra lateral.")
        st.stop()

    if ativar_inpe and not satelites_sel:
        st.error("⚠️ Você precisa selecionar pelo menos um satélite do INPE para realizar a busca.")
        st.stop()

    hoje = datetime.now()
    if unidade_dd == "Dias": dt_ini = hoje - timedelta(days=quantidade_sel)
    elif unidade_dd == "Meses": dt_ini = hoje - timedelta(days=30*quantidade_sel)
    else: dt_ini = hoje - timedelta(days=365*quantidade_sel)

    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.status(f"🛰️ Processando dados para: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando fronteiras geográficas...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        
        df_rec = pd.DataFrame()
        if ativar_inpe:
            st.write("📡 Consultando satélites do INPE...")
            df = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel)
            
            if not df.empty:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                gdf = gpd.sjoin(gdf, limite, predicate="within")
                df_rec = pd.DataFrame(gdf.drop(columns="geometry"))

        status.update(label="✅ Consulta finalizada com sucesso!", state="complete", expanded=False)

    if ativar_inpe and df_rec.empty:
        st.warning("⚠️ Nenhum foco de calor detectado pelo INPE dentro do limite geográfico neste período.")

    # Só para de rodar se não tiver INPE E também não tiver MODIS selecionado
    if not ativar_modis and df_rec.empty:
        st.stop()

    # --- PROCESSAMENTO DAS ÁREAS PROTEGIDAS (Apenas INPE) ---
    focos_em_areas = pd.DataFrame()
    areas_afetadas = gpd.GeoDataFrame()
    
    if ativar_inpe and not df_rec.empty and area_protegida != "Nenhuma":
        with st.status(f"🌳 Cruzando dados do INPE com {area_protegida}...", expanded=True) as status_area:
            try:
                st.write("Baixando polígonos oficiais...")
                gdf_areas = carregar_areas_protegidas(area_protegida)
                
                st.write("Realizando interseção espacial...")
                gdf_areas = gdf_areas.to_crs(gdf.crs)
                
                if 'index_right' in gdf.columns: gdf = gdf.drop(columns=['index_right'])
                if 'index_left' in gdf.columns: gdf = gdf.drop(columns=['index_left'])
                
                gdf_focos_risco = gpd.sjoin(gdf, gdf_areas, predicate='within')
                if not gdf_focos_risco.empty:
                    focos_em_areas = pd.DataFrame(gdf_focos_risco.drop(columns="geometry"))
                    areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])]
                    
                status_area.update(label="✅ Análise de risco concluída!", state="complete", expanded=False)
            except Exception as e:
                status_area.update(label="❌ Falha no cruzamento de dados.", state="error", expanded=False)
                st.error(f"🔍 Detalhe técnico do erro para debug: {e}")

    # --- CARD DE RESUMO ESTILIZADO ---
    total_focos = len(df_rec) if ativar_inpe else 0
    data_limite = hoje.strftime("%d/%m/%Y")
    
    # Monta o texto do card de forma dinâmica
    texto_resumo = f"🔥 Total INPE Confirmado: {total_focos:,} focos" if ativar_inpe else "🛰️ Exibindo apenas a camada do NASA MODIS"
    
    card_html = f"""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 8px solid #ff4b4b; margin-bottom: 15px; box-shadow: 1px 1px 4px rgba(0,0,0,0.05);">
        <h3 style="color: #c0392b; margin: 0; font-size: 22px; font-weight: bold;">
            {texto_resumo}
        </h3>
        <p style="color: #636e72; margin: 4px 0 0 0; font-size: 15px;">
            Análise: {val_sel} | INPE (até {data_limite}) | MODIS (Ano: {ano_modis if ativar_modis else 'Inativo'})
        </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # --- ALERTAS DE RISCO (Apenas se tiver INPE ativo e cruzamento feito) ---
    if ativar_inpe and not df_rec.empty:
        if not focos_em_areas.empty:
            qtd_risco = len(focos_em_areas)
            df_ranking_areas = focos_em_areas['nome_area'].value_counts().reset_index()
            df_ranking_areas.columns = ['Área Protegida', 'Focos']
            
            st.error(f"🚨 **ALERTA CRÍTICO:** {qtd_risco} focos detectados DENTRO de áreas protegidas!")
            
            col_alerta1, col_alerta2 = st.columns([1.5, 1])
            with col_alerta1:
                qtd_areas = min(10, len(df_ranking_areas))
                titulo_dinamico = f"🔥 Top {qtd_areas} Áreas Mais Afetadas" if qtd_areas > 1 else "🔥 Área Mais Afetada"
                
                fig_areas = px.bar(
                    df_ranking_areas.head(10), 
                    x='Focos', 
                    y='Área Protegida', 
                    orientation='h', 
                    text='Focos',
                    color='Focos', 
                    color_continuous_scale=px.colors.sequential.Reds,
                    title=titulo_dinamico
                )
                fig_areas.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=40, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_areas, use_container_width=True)
                
            with col_alerta2:
                st.markdown("**Lista Completa de Áreas Afetadas**")
                st.dataframe(df_ranking_areas, hide_index=True, height=350, use_container_width=True)
                
        elif area_protegida != "Nenhuma":
            st.success(f"✅ Nenhum foco do INPE detectado dentro de {area_protegida} na região analisada.")
    
    st.markdown("---") 
    
    # 2. BOTÃO DE DOWNLOAD (SÓ APARECE SE TIVER DADOS DO INPE)
    if ativar_inpe and not df_rec.empty:
        csv_dados = df_rec.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Baixar Dados INPE (CSV)", data=csv_dados, file_name=f"focos_{val_sel.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
    
    col1, col2 = st.columns([1.3, 1])
    
    with col1:
        st.subheader("🗺️ Mapa de Calor Espacial")
        centro = limite.geometry.union_all().centroid
        m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
        
        folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
        
        # --- CAMADA MODIS ---
        if ativar_modis:
            try:
                bounds = limite.total_bounds
                ee_geom = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]])
                
                data_ini = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                data_fim = data_ini.advance(1, 'month')
                
                # Filtra a coleção
                colecao_modis = ee.ImageCollection('MODIS/061/MCD64A1') \
                    .filterDate(data_ini, data_fim) \
                    .filterBounds(ee_geom)
                
                # VERIFICAÇÃO CRÍTICA: Checa se a NASA já tem dados para este mês
                qtd_imagens = colecao_modis.size().getInfo()
                
                if qtd_imagens == 0:
                    st.warning(f"⚠️ A imagem de cicatrizes do MODIS para **{mes_modis}/{ano_modis}** ainda não está disponível no servidor ou não há dados para esta região.")
                else:
                    area_queimada = colecao_modis.select('BurnDate').max().clip(ee_geom)
                    area_queimada = area_queimada.updateMask(area_queimada.gt(0))
                    
                    m.add_ee_layer(area_queimada, {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']}, 'Cicatrizes MODIS')
                    
            except Exception as e:
                st.error(f"❌ Erro ao processar MODIS: {e}")

        # --- CAMADA INPE ---
        if ativar_inpe and not df_rec.empty:
            if not areas_afetadas.empty:
                estilo_tooltip = "font-size: 12px; max-width: 250px; white-space: normal; background-color: white; color: black; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
                folium.GeoJson(
                    areas_afetadas.__geo_interface__, 
                    style_function=lambda x: {'fillColor': '#e74c3c', 'color': '#c0392b', 'weight': 2, 'fillOpacity': 0.4},
                    tooltip=folium.GeoJsonTooltip(fields=['nome_area'], aliases=['Área Protegida:'], style=estilo_tooltip)
                ).add_to(m)

            HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15, blur=20).add_to(m)
        
        st_folium(m, width=700, height=750, returned_objects=[])
    
    with col2:
        if ativar_inpe and not df_rec.empty:
            st.subheader("📈 Evolução Temporal dos Focos (INPE)")
            data_col = next(c for c in df_rec.columns if 'data' in c)
            df_rec[data_col] = pd.to_datetime(df_rec[data_col])
            
            freq = 'D' if (hoje - dt_ini).days <= 90 else 'MS'
            df_g = df_rec.set_index(data_col).resample(freq).size().reset_index(name='focos')
            
            fig_line = px.line(df_g, x=data_col, y='focos', markers=True, height=350)
            fig_line.update_traces(line_color='#e64a19', line_width=3)
            fig_line.update_layout(template='plotly_dark', xaxis_title="Tempo", yaxis_title="Nº de Focos", margin=dict(t=20, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

            if 'municipio' in df_rec.columns and tipo_analise != "Por Município":
                df_top_mun = df_rec['municipio'].value_counts().reset_index()
                qtd_mun = min(5, len(df_top_mun))
                titulo_mun = f"🏆 Top {qtd_mun} Municípios Afetados (INPE)" if qtd_mun > 1 else "🏆 Município Afetado"
                st.subheader(titulo_mun)
                
                df_top_mun = df_top_mun.head(5)
                df_top_mun.columns = ['Município', 'Focos']
                
                fig_bar = px.bar(df_top_mun, x='Focos', y='Município', orientation='h', text='Focos', color='Focos', color_continuous_scale=px.colors.sequential.Reds)
                fig_bar.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=320, margin=dict(t=20, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_bar, use_container_width=True)
        elif ativar_modis:
            st.info("ℹ️ Os gráficos estatísticos são gerados apenas quando os Focos de Calor do INPE estão ativados, pois dependem de dados tabulares pontuais. O MODIS processa apenas a imagem raster no mapa.")

else:
    st.info("👈 Use os filtros ao lado para selecionar o local e o período de análise.")

# --- RODAPÉ ---
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 13px; color: #636e72;">
    Desenvolvido por <br>
    <b style="font-size: 15px;">Ana Carolina Andrade</b> <br>
    <a href="https://www.linkedin.com/in/ana-carolina-santos-3920931b3" target="_blank" style="text-decoration: none; color: #e64a19; font-weight: bold;">LinkedIn</a> | 
    <a href="https://github.com/AnaAndradesant" target="_blank" style="text-decoration: none; color: #e64a19; font-weight: bold;">GitHub</a>
</div>
""", unsafe_allow_html=True)
