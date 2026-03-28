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
        ### 🛰️ Monitor de Queimadas
        Dashboard interativo para monitoramento de focos de calor e área queimada.
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

# Condicional: Mostra os filtros específicos dependendo do que o usuário escolheu
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
    mes_modis = st.sidebar.selectbox("Mês de Referência:", list(range(1, 13)), index=7)

st.sidebar.markdown("---")
area_protegida = st.sidebar.selectbox("🌳 Análise de Risco (Cruzamento Espacial):", ["Nenhuma", "Terras Indígenas", "Unidades de Conservação"])

gerar = st.sidebar.button("▶️ Gerar Dashboard", type="primary", use_container_width=True)

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

        # Variáveis globais para os dois cenários
        df_ranking_areas = pd.DataFrame()
        areas_afetadas = gpd.GeoDataFrame()
        total_valor = 0
        df_rec = pd.DataFrame() # Apenas para o INPE
        area_queimada_img = None # Apenas para o MODIS

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
                    st.write(f"🌳 Cruzando focos com {area_protegida}...")
                    gdf_areas = carregar_areas_protegidas(area_protegida)
                    gdf_areas = gdf_areas.to_crs(gdf.crs)
                    gdf_focos_risco = gpd.sjoin(gdf, gdf_areas, predicate='within')
                    
                    if not gdf_focos_risco.empty:
                        areas_afetadas = gdf_areas[gdf_areas['nome_area'].isin(gdf_focos_risco['nome_area'])]
                        focos_em_areas = pd.DataFrame(gdf_focos_risco.drop(columns="geometry"))
                        df_ranking_areas = focos_em_areas['nome_area'].value_counts().reset_index()
                        df_ranking_areas.columns = ['Área Protegida', 'Valor']

        # ==========================================
        # RAMIFICAÇÃO: MODIS
        # ==========================================
        else:
            st.write("☁️ Analisando satélite MODIS no GEE...")
            try:
                data_ini_ee = ee.Date.fromYMD(ano_modis, mes_modis, 1)
                colecao = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(data_ini_ee, data_ini_ee.advance(1, 'month')).filterBounds(ee_geom_complex)
                
                if colecao.size().getInfo() > 0:
                    area_queimada_img = colecao.select('BurnDate').max().clip(ee_geom_complex)
                    
                    # CÁLCULO EM KM2 (divide por 1 milhão)
                    img_area_km2 = ee.Image.pixelArea().divide(1000000).updateMask(area_queimada_img.gt(0)).rename('area_km2')
                    
                    stats_total = img_area_km2.reduceRegion(
                        reducer=ee.Reducer.sum(), 
                        geometry=ee_geom_complex, 
                        scale=500, 
                        maxPixels=1e13,
                        bestEffort=True 
                    ).getInfo()
                    
                    total_valor = round(stats_total.get('area_km2', 0) if stats_total.get('area_km2') else 0, 2)

                    # Cruzamento Espacial MODIS
                    if area_protegida != "Nenhuma" and total_valor > 0:
                        st.write(f"🌳 Calculando km² afetados em {area_protegida}...")
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

            except Exception as e:
                st.warning(f"⚠️ Erro ao processar MODIS: {e}")

        status.update(label="✅ Análise concluída!", state="complete", expanded=False)

    # ==========================================
    # RENDERIZAÇÃO VISUAL (Serve para INPE e MODIS)
    # ==========================================
    if total_valor == 0:
        st.error("⚠️ Nenhum registro detectado neste período/local pela fonte de dados selecionada.")
    else:
        # 1. CARD DE RESUMO ESTILIZADO
        texto_titulo = f"Total Confirmado no Mapa: {total_valor:,} focos" if "INPE" in fonte_escolhida else f"Área Queimada Total: {total_valor:,.2f} km²"
        texto_sub = f"Período: {quantidade_sel} {unidade_dd}" if "INPE" in fonte_escolhida else f"Período: Mês {mes_modis} do Ano {ano_modis}"
        
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
            st.error(f"🚨 **ALERTA CRÍTICO:** Registros de {metrica} DENTRO de áreas protegidas!")
            
            col_alerta1, col_alerta2 = st.columns([1.5, 1])
            with col_alerta1:
                qtd_areas = min(10, len(df_ranking_areas))
                titulo_dinamico = f"🔥 Top {qtd_areas} Áreas Mais Afetadas" if qtd_areas > 1 else "🔥 Área Mais Afetada"
                
                fig_areas = px.bar(
                    df_ranking_areas.head(10), 
                    x='Valor', 
                    y='Área Protegida', 
                    orientation='h', 
                    text='Valor',
                    color='Valor', 
                    color_continuous_scale=px.colors.sequential.Reds,
                    title=titulo_dinamico
                )
                nome_eixo_x = "Nº de Focos" if "INPE" in fonte_escolhida else "Área Afetada (km²)"
                fig_areas.update_layout(
                    template='plotly_dark', xaxis_title=nome_eixo_x,
                    yaxis={'categoryorder':'total ascending'}, 
                    height=350, margin=dict(t=40, b=20), coloraxis_showscale=False
                )
                st.plotly_chart(fig_areas, use_container_width=True)
                
            with col_alerta2:
                st.markdown("**Lista Completa de Áreas Afetadas**")
                st.dataframe(df_ranking_areas, hide_index=True, height=350, use_container_width=True)
                
        elif area_protegida != "Nenhuma":
            st.success(f"✅ Nenhum registro detectado dentro de {area_protegida} na região analisada.")
        
        st.markdown("---") 
        
        # Botão de Download (Apenas INPE tem CSV de Focos Ponto a Ponto)
        if "INPE" in fonte_escolhida:
            csv_dados = df_rec.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("📥 Baixar Dados Focos (CSV)", data=csv_dados, file_name=f"focos_{val_sel.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)

        # 3. CONSTRUÇÃO DO MAPA E GRÁFICOS INFERIORES
        col1, col2 = st.columns([1.3, 1])
        
        with col1:
            st.subheader("🗺️ Mapa Espacial")
            centro = limite.geometry.union_all().centroid
            m = folium.Map(location=[centro.y, centro.x], zoom_start=10 if tipo_analise == "Por Município" else 6, tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}', attr='Google Satélite')
            
            folium.GeoJson(limite.__geo_interface__, style_function=lambda x: {'fillColor': 'transparent', 'color': '#00d4ff', 'weight': 3}).add_to(m)
            
            # Pinta as áreas protegidas de vermelho translúcido se houver interseção
            if not areas_afetadas.empty:
                estilo_tooltip = "font-size: 12px; max-width: 250px; white-space: normal; background-color: white; color: black; border-radius: 4px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
                folium.GeoJson(
                    areas_afetadas.__geo_interface__, 
                    style_function=lambda x: {'fillColor': '#e74c3c', 'color': '#c0392b', 'weight': 2, 'fillOpacity': 0.4},
                    tooltip=folium.GeoJsonTooltip(fields=['nome_area'], aliases=['Área Protegida:'], style=estilo_tooltip)
                ).add_to(m)

            # Lógica de renderização no Mapa (Heatmap pro INPE, Raster pro MODIS)
            if "INPE" in fonte_escolhida:
                HeatMap(df_rec[["latitude", "longitude"]].dropna().values.tolist(), radius=15, blur=20).add_to(m)
            elif "MODIS" in fonte_escolhida and area_queimada_img:
                m.add_ee_layer(area_queimada_img.updateMask(area_queimada_img.gt(0)), {'min': 1, 'max': 366, 'palette': ['orange', 'red', 'darkred']}, 'MODIS Burned Area')

            st_folium(m, width=700, height=750, returned_objects=[])
        
        with col2:
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

                if 'municipio' in df_rec.columns and tipo_analise != "Por Município":
                    df_top_mun = df_rec['municipio'].value_counts().reset_index()
                    qtd_mun = min(5, len(df_top_mun))
                    titulo_mun = f"🏆 Top {qtd_mun} Municípios" if qtd_mun > 1 else "🏆 Município Afetado"
                    st.subheader(titulo_mun)
                    
                    df_top_mun = df_top_mun.head(5)
                    df_top_mun.columns = ['Município', 'Focos']
                    
                    fig_bar = px.bar(df_top_mun, x='Focos', y='Município', orientation='h', text='Focos', color='Focos', color_continuous_scale=px.colors.sequential.Reds)
                    fig_bar.update_layout(template='plotly_dark', yaxis={'categoryorder':'total ascending'}, height=320, margin=dict(t=20, b=20), coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
            else:
                # O MODIS é mensal, não faz sentido gráfico de linha para 1 mês.
                st.info("ℹ️ Os dados do MODIS selecionados representam o consolidado mensal da área queimada. O detalhamento geográfico por município para cicatrizes de fogo requer processamento avançado, por isso focamos o dashboard nas Áreas de Risco (Terras Indígenas e UCs) e no panorama visual através do mapa ao lado.")

else:
    st.info("👈 Use os filtros ao lado para selecionar a Fonte de Dados, o local e o período de análise.")

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
