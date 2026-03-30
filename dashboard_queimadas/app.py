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

# ==========================================
# 🛠️ TRUQUE DE SOBREVIVÊNCIA (KEEP-ALIVE)
# Força o Streamlit a manter a conexão ativa
if 'last_heartbeat' not in st.session_state:
    st.session_state.last_heartbeat = datetime.now()

if (datetime.now() - st.session_state.last_heartbeat).seconds > 60:
    st.session_state.last_heartbeat = datetime.now()
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

def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=0.7):
    try:
        map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
        tiles_url = map_id_dict.get('tile_fetcher', {}).url_format if 'tile_fetcher' in map_id_dict else map_id_dict.get('urlFormat', map_id_dict.get('url_format', ''))
        folium.raster_layers.TileLayer(
            tiles=tiles_url, attr='Map Data © Google Earth Engine', name=name,
            overlay=True, control=True, show=show, opacity=opacity
        ).add_to(self)
    except Exception as e:
        st.error(f"🚨 Erro crítico ao desenhar a camada: {e}")

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

    if tipo == "Por Estado": filtro_base = f"estado ILIKE '{dic_estados.get(val_estado, val_estado)}'"
    elif tipo == "Por Bioma":
        tradutor = {"Amazônia": "Amaz%nia", "Mata Atlântica": "Mata Atl%ntica"}
        filtro_base = f"bioma ILIKE '{tradutor.get(val_bioma, val_bioma)}'"
    elif tipo == "Por Município":
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
# ==========================================
# 🐛 BOTÃO DO MODO DE VALIDAÇÃO (QA)
# ==========================================
modo_debug = st.sidebar.toggle("🐛 Modo de Validação (QA)", value=False, help="Mostra os dados brutos e cálculos por trás dos panos")

gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

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
<a href="https://mail.google.com/mail/?view=cm&fs=1&to=anacarolinasantos580@gmail.com" target="_blank" style="flex: 1; text-decoration: none; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); display: flex; justify-content: center; align-items: center;">
<img src="{gmail_logo_url}" alt="Gmail Logo" style="width: 26px; height: auto; display: block; margin: 0 auto;">
</a>
<a href="https://www.linkedin.com/in/ana-carolina-santos-3920931b3" target="_blank" style="flex: 1; text-decoration: none; background-color: #0077B5; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: flex; justify-content: center; align-items: center;">
<img src="{linkedin_logo_url}" alt="LinkedIn Logo" style="width: 22px; height: auto; display: block; margin: 0 auto;">
</a>
</div>
</div>
"""

st.sidebar.markdown(html_contato_novo, unsafe_allow_html=True)
# -------------------------------------


# --- INTERFACE (TELA PRINCIPAL) ---
st.title("🔥 Dashboard de Queimadas 🔥")

if gerar:
    hoje = datetime.now()
    val_sel = bioma_dd if tipo_analise == "Por Bioma" else (estado_dd if tipo_analise == "Por Estado" else f"{municipio_dd} ({estado_dd})")

    with st.status(f"🛰️ Processando dados para: **{val_sel}**", expanded=True) as status:
        st.write("🌍 Carregando fronteiras geográficas...")
        limite = carregar_fronteira(tipo_analise, estado_dd, bioma_dd, municipio_dd)
        geom_unida = limite.geometry.union_all()
        ee_geom_complex = ee.Geometry(geom_unida.__geo_interface__)

        df_ranking_areas = pd.DataFrame()
        areas_afetadas = gpd.GeoDataFrame()
        total_valor = 0
        df_rec = pd.DataFrame() 
        area_queimada_img = None 
        df_top_mun_modis = pd.DataFrame()
        df_modis_temporal = pd.DataFrame()

        # ==========================================
        # RAMIFICAÇÃO: INPE
        # ==========================================
        if "INPE" in fonte_escolhida:
            if not satelites_sel:
                st.error("⚠️ Você precisa selecionar pelo menos um satélite.")
                st.stop()
                
            if unidade_dd == "Dias": dt_ini = hoje - timedelta(days=quantidade_sel)
            elif unidade_dd == "Meses": dt_ini = hoje - timedelta(days=30*quantidade_sel)
            else: dt_ini = hoje - timedelta(days=365*quantidade_sel)

            st.write("📡 Consultando satélites do INPE...")
            df = buscar_focos_inpe(tipo_analise, estado_dd, bioma_dd, municipio_dd, dt_ini.strftime("%Y-%m-%d"), hoje.strftime("%Y-%m-%d"), satelites_sel)
            
            if not df.empty:
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                gdf = gpd.sjoin(gdf, limite, predicate="within")
                df_rec = pd.DataFrame(gdf.drop(columns="geometry"))
                total_valor = len(df_rec)
                
                # Cruzamento Espacial INPE
                if area_protegida != "Nenhuma" and not df_rec.empty:
                    st.write(f"🌳 Isolando focos em {area_protegida}...")
                    gdf_areas = carregar_areas_protegidas(area_protegida)
                    gdf_areas = gdf_areas.to_crs(gdf.crs)
                    
                    # CORREÇÃO: Remover 'index_right' do sjoin anterior
                    if 'index_right' in gdf.columns:
                        gdf = gdf.drop(columns=['index_right'])
                        
                    gdf_focos_risco = gpd.sjoin(gdf, gdf_areas, predicate='within')
                    
                    if not gdf_focos_risco.empty:
                        areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])]
                        focos_em_areas = pd.DataFrame(gdf_focos_risco.drop(columns="geometry"))
                        df_ranking_areas = focos_em_areas['nome_area'].value_counts().reset_index()
                        df_ranking_areas.columns = ['Área Protegida', 'Valor']
                        
                        # Filtra o DataFrame principal (Deixa só os focos nas TIs/UCs)
                        df_rec = focos_em_areas
                        total_valor = len(df_rec)
                    else:
                        df_rec = pd.DataFrame()
                        total_valor = 0

        # ==========================================
        # RAMIFICAÇÃO: MODIS
        # ==========================================
        else:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                # 1. Carrega imagem base do mês selecionado para o mapa principal
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ini_ee, data_ini_ee.advance(1, 'month')).filterBounds(ee_geom_complex)
                
                if colecao.size().getInfo() > 0:
                    area_queimada_img = colecao.select('BurnDate').max().clip(ee_geom_complex)
                    img_area_km2 = ee.Image.pixelArea().divide(1000000).updateMask(area_queimada_img.gt(0)).rename('area_km2')
                    
                    stats_total = img_area_km2.reduceRegion(
                        reducer=ee.Reducer.sum(), geometry=ee_geom_complex, scale=500, maxPixels=1e13, bestEffort=True 
                    ).getInfo()
                    total_valor = round(stats_total.get('area_km2', 0) if stats_total.get('area_km2') else 0, 2)

                    # 2. Cruzamento Espacial MODIS (Recorte da Imagem e Tabela de Áreas)
                    if area_protegida != "Nenhuma" and total_valor > 0:
                        st.write(f"🌳 Isolando km² afetados em {area_protegida}...")
                        gdf_areas_br = carregar_areas_protegidas(area_protegida)
                        gdf_areas = gpd.sjoin(gdf_areas_br, limite, predicate='intersects').drop(columns=['index_right'])
                        
                        if not gdf_areas.empty:
                            features_ee = [ee.Feature(ee.Geometry(row['geometry'].__geo_interface__), {'nome_area': row['nome_area']}) for _, row in gdf_areas.iterrows()]
                            fc_areas = ee.FeatureCollection(features_ee)
                            
                            stats = img_area_km2.reduceRegions(collection=fc_areas, reducer=ee.Reducer.sum(), scale=500).getInfo()
                            recs = [{'Área Protegida': f['properties']['nome_area'], 'Valor': round(f['properties'].get('sum', 0), 2)} for f in stats['features'] if f['properties'].get('sum', 0) > 0]
                            
                            df_ranking_areas = pd.DataFrame(recs).sort_values(by='Valor', ascending=False)
                            if not df_ranking_areas.empty:
                                areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(df_ranking_areas['Área Protegida'])]
                                
                                # Atualiza total_valor e recorta a imagem para as TIs/UCs afetadas
                                total_valor = round(df_ranking_areas['Valor'].sum(), 2)
                                ee_geom_afetadas = ee.Geometry(areas_afetadas.geometry.union_all().__geo_interface__)
                                area_queimada_img = area_queimada_img.clip(ee_geom_afetadas)
                                img_area_km2 = ee.Image.pixelArea().divide(1000000).updateMask(area_queimada_img.gt(0))
                            else:
                                total_valor = 0

                    # 3. Ranking de Municípios MODIS (Calculado sempre que não for análise por município)
                    if tipo_analise != "Por Município" and total_valor > 0:
                        st.write("🏙️ Calculando ranking de municípios (MODIS)...")
                        muns_ee = ee.FeatureCollection("FAO/GAUL/2015/level2").filterBounds(ee_geom_complex)
                        # O img_area_km2 já está recortado pela TI se o filtro estiver ativo!
                        stats_mun = img_area_km2.reduceRegions(collection=muns_ee, reducer=ee.Reducer.sum(), scale=1000).getInfo()
                        
                        recs_mun = [{'Município': f['properties']['ADM2_NAME'], 'Valor': round(f['properties'].get('sum', 0), 2)} for f in stats_mun['features'] if f['properties'].get('sum', 0) > 0]
                        if recs_mun:
                            df_top_mun_modis = pd.DataFrame(recs_mun).sort_values(by='Valor', ascending=False).head(5)

                    # 4. Evolução Temporal MODIS (12 meses do ano selecionado)
                    if total_valor > 0:
                        st.write("📊 Calculando série temporal anual (MODIS)...")
                        # Define qual geometria usar para a série temporal
                        geom_temporal = ee_geom_afetadas if (area_protegida != "Nenhuma" and not areas_afetadas.empty) else ee_geom_complex
                        
                        def calc_mes(m):
                            m_num = ee.Number(m)
                            ini = ee.Date.fromYMD(ano_modis, m_num, 1)
                            fim = ini.advance(1, 'month')
                            img_mes = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(ini, fim).select('BurnDate').max().clip(geom_temporal)
                            area_calc = ee.Image.pixelArea().divide(1000000).updateMask(img_mes.gt(0))
                            val = area_calc.reduceRegion(reducer=ee.Reducer.sum(), geometry=geom_temporal, scale=1000, maxPixels=1e10).get('area')
                            return ee.Feature(None, {'mes': m_num, 'area': val})
                        
                        # Mapeia a função sobre os 12 meses do ano
                        meses_list = ee.List.sequence(1, 12)
                        fc_meses = ee.FeatureCollection(meses_list.map(calc_mes)).getInfo()
                        
                        dados_temp = [{'Mês': f['properties']['mes'], 'Área (km²)': round(f['properties'].get('area') or 0, 2)} for f in fc_meses['features']]
                        df_modis_temporal = pd.DataFrame(dados_temp)
                        meses_map = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun', 7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
                        df_modis_temporal['Mês Nome'] = df_modis_temporal['Mês'].map(meses_map)

            except Exception as e:
                st.warning(f"⚠️ Erro ao processar MODIS: {e}")

        status.update(label="✅ Análise concluída!", state="complete", expanded=False)

    # ==========================================
    # RENDERIZAÇÃO VISUAL (Serve para INPE e MODIS)
    # ==========================================
    if total_valor == 0:
        st.error("⚠️ Nenhum registro detectado nos limites selecionados.")
    else:
        # 1. CARD DE RESUMO ESTILIZADO
        texto_titulo = f"Total Confirmado: {total_valor:,} focos" if "INPE" in fonte_escolhida else f"Área Queimada Total: {total_valor:,.2f} km²"
        texto_sub = f"Período: {quantidade_sel} {unidade_dd}" if "INPE" in fonte_escolhida else f"Período: Mês {mes_modis} de {ano_modis} (Mapa) / Ano {ano_modis} (Evolução)"
        
        card_html = f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 8px solid #ff4b4b; margin-bottom: 15px; box-shadow: 1px 1px 4px rgba(0,0,0,0.05);">
            <h3 style="color: #c0392b; margin: 0; font-size: 22px; font-weight: bold;">
                🔥 {texto_titulo}
            </h3>
            <p style="color: #636e72; margin: 4px 0 0 0; font-size: 15px;">
                Análise: {val_sel} | {texto_sub}
            </p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # 2. ALERTA DE RISCO COM GRÁFICO DINÂMICO
        if not df_ranking_areas.empty:
            metrica = "focos detectados" if "INPE" in fonte_escolhida else "km² queimados"
            st.error(f"🚨 **ANÁLISE FOCADA:** {total_valor} {metrica} limitados dentro de {area_protegida}!")
            
            col_alerta1, col_alerta2 = st.columns([1.5, 1])
            with col_alerta1:
                qtd_areas = min(10, len(df_ranking_areas))
                titulo_dinamico = f"🔥 Top {qtd_areas} Áreas Mais Afetadas" if qtd_areas > 1 else "🔥 Área Mais Afetada"
                
                fig_areas = px.bar(
                    df_ranking_areas.head(10), x='Valor', y='Área Protegida', orientation='h', 
                    text='Valor', color='Valor', color_continuous_scale=px.colors.sequential.Reds, title=titulo_dinamico
                )
                nome_eixo_x = "Nº de Focos" if "INPE" in fonte_escolhida else "Área Afetada (km²)"
                fig_areas.update_layout(template='plotly_dark', xaxis_title=nome_eixo_x, yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=40, b=20), coloraxis_showscale=False)
                st.plotly_chart(fig_areas, use_container_width=True)
                
            with col_alerta2:
                st.markdown("**Lista Completa de Áreas Afetadas**")
                st.dataframe(df_ranking_areas, hide_index=True, height=350, use_container_width=True)
        
        st.markdown("---") 

        # 3. CONSTRUÇÃO DO MAPA E GRÁFICOS INFERIORES
        col1, col2 = st.columns([1.3, 1])
        
        with col1:
            st.subheader("🗺️ Mapa Espacial")
            centro = limite.geometry.union_all().centroid
            m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
            
            folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
            
            if not areas_afetadas.empty:
                estilo_tooltip = "font-size: 12px; max-width: 250px; white-space: normal; background-color: white; color: black; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
                folium.GeoJson(
                    areas_afetadas.__geo_interface__, 
                    style_function=lambda x: {'fillColor': 'transparent', 'color': '#c0392b', 'weight': 3},
                    tooltip=folium.GeoJsonTooltip(fields=['nome_area'], aliases=['Área Protegida:'], style=estilo_tooltip)
                ).add_to(m)

            if "INPE" in fonte_escolhida and not df_rec.empty:
                HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15, blur=20).add_to(m)
            elif "MODIS" in fonte_escolhida and area_queimada_img:
                m.add_ee_layer(area_queimada_img.updateMask(area_queimada_img.gt(0)), {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']}, 'MODIS Burned Area')

            st_folium(m, width=700, height=750, returned_objects=[])
        
        with col2:
            # --- SEÇÃO DE GRÁFICOS: INPE ---
            if "INPE" in fonte_escolhida:
                st.subheader("📈 Evolução Temporal dos Focos")
                data_col = next(c for c in df_rec.columns if 'data' in c)
                df_rec[data_col] = pd.to_datetime(df_rec[data_col])
                
                freq = 'D' if (hoje - dt_ini).days <= 90 else 'MS'
                df_g = df_rec.set_index(data_col).resample(freq).size().reset_index(name='focos')
                
                fig_line = px.line(df_g, x=data_col, y='focos', markers=True, height=350)
                fig_line.update_traces(line_color='#e64a19', line_width=3)
                fig_line.update_layout(template='plotly_dark', xaxis_title="Tempo", yaxis_title="Nº de Focos", margin=dict(t=20, b=20))
                st.plotly_chart(fig_line, use_container_width=True)

                # Gráfico de Municípios INPE (Aparece mesmo com filtro de área protegida)
                if 'municipio' in df_rec.columns and tipo_analise != "Por Município":
                    df_top_mun = df_rec['municipio'].value_counts().reset_index()
                    qtd_mun = min(5, len(df_top_mun))
                    st.subheader(f"🏆 Top {qtd_mun} Municípios Afetados")
                    
                    df_top_mun = df_top_mun.head(5)
                    df_top_mun.columns = ['Município', 'Focos']
                    
                    fig_bar = px.bar(df_top_mun, x='Focos', y='Município', orientation='h', text='Focos', color='Focos', color_continuous_scale=px.colors.sequential.Reds)
                    fig_bar.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=320, margin=dict(t=20, b=20), coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

            # --- SEÇÃO DE GRÁFICOS: MODIS ---
            else:
                if not df_modis_temporal.empty:
                    st.subheader(f"📈 Evolução Temporal ({ano_modis})")
                    fig_line = px.line(df_modis_temporal, x='Mês Nome', y='Área (km²)', markers=True, height=350)
                    fig_line.update_traces(line_color='#e64a19', line_width=3)
                    fig_line.update_layout(template='plotly_dark', xaxis_title="Mês", yaxis_title="Área Afetada (km²)", margin=dict(t=20, b=20))
                    st.plotly_chart(fig_line, use_container_width=True)

                # Gráfico de Municípios MODIS (Aparece mesmo com filtro de área protegida)
                if tipo_analise != "Por Município" and not df_top_mun_modis.empty:
                    st.subheader("🏆 Top 5 Municípios Afetados")
                    fig_bar = px.bar(df_top_mun_modis, x='Valor', y='Município', orientation='h', text='Valor', color='Valor', color_continuous_scale=px.colors.sequential.Reds)
                    fig_bar.update_layout(template='plotly_dark', xaxis_title="Área Afetada (km²)", yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=20, b=20), coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

        # ==========================================
        # 🐛 MODO DEBUG (QA - QUALITY ASSURANCE)
        # ==========================================
        if modo_debug:
            st.markdown("---")
            st.header("🐛 Painel de Validação")
            
            with st.expander("🔍 Abrir Raio-X dos Dados", expanded=True):
                col_db1, col_db2 = st.columns(2)
                
                with col_db1:
                    st.markdown("**1. Verificação de Matemática (SJOIN)**")
                    if "INPE" in fonte_escolhida:
                        qtd_bruta = len(df) if 'df' in locals() else 0
                        qtd_limite = len(df_rec)
                        
                        st.write(f"- Focos totais baixados da API (Bruto): **{qtd_bruta}**")
                        st.write(f"- Focos na área final exibida: **{qtd_limite}**")
                        
                        if area_protegida != "Nenhuma" and 'focos_em_areas' in locals():
                            qtd_risco = len(focos_em_areas)
                            st.write(f"- Focos originais que caíram em {area_protegida}: **{qtd_risco}**")
                            
                            # Alerta de possível duplicação
                            if qtd_risco > qtd_bruta:
                                st.warning("⚠️ ALERTA: Você tem mais focos cruzados do que baixados no total. Isso significa que há polígonos sobrepostos na base de áreas protegidas gerando dados duplicados!")
                    else:
                        st.write(f"- Área Queimada Calculada Total (MODIS): **{total_valor} km²**")

                with col_db2:
                    st.markdown("**2. Dados para o Teste do Espelho**")
                    st.info(f"Use estes parâmetros no site oficial:\n- **Local:** {val_sel}\n- **Fonte:** {fonte_escolhida}\n- **Área Protegida:** {area_protegida}")
                
                st.markdown("**3. Tabela Bruta (Amostra para inspeção)**")
                if "INPE" in fonte_escolhida and not df_rec.empty:
                    st.dataframe(df_rec.head(100), use_container_width=True)
                elif "MODIS" in fonte_escolhida and not df_modis_temporal.empty:
                    st.dataframe(df_modis_temporal, use_container_width=True)

else:
    st.info("👈 Use os filtros ao lado para selecionar a Fonte de Dados, o local e o período de análise.")
