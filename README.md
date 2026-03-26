# 🌍 Geo-Tools & Monitoramento Ambiental

Bem-vindo ao meu repositório de ferramentas geoespaciais! Aqui você encontrará soluções desenvolvidas para otimizar fluxos de trabalho no QGIS e monitorar dados ambientais críticos em tempo real.

## 🛰️ 1. Dashboard de Queimadas (Web App)

Um painel interativo e de alta performance para monitoramento de focos de calor em todo o território brasileiro, utilizando dados oficiais do INPE (BDQueimadas).

🔗 **[Acesse o Dashboard Online Aqui](https://monitor-focos-queimadas.streamlit.app/)**

### 📌 Funcionalidades
* **Filtro Geográfico:** Análise por Estado, Município ou Bioma.
* **Escala Temporal:** Consulta customizada por Dias, Meses ou Anos.
* **Mapa de Calor:** Visualização espacial utilizando satélites de referência (AQUA e NPP).
* **Análise de Dados:** Gráficos de evolução temporal e estatísticas confirmadas.

### 🛠️ Tecnologias Utilizadas
* **Python** (Streamlit, Pandas, Geopandas)
* **Mapas:** Folium e Leaflet
* **Dados:** API TerraBrasilis / INPE

---

## 🔌 2. Plugin Connect (Para QGIS)

O **Connect** é um plugin desenvolvido para o QGIS 3 que permite navegar entre grupos de camadas de forma rápida e prática, ideal para quem trabalha com projetos complexos e muitas camadas.

### 📌 O que ele faz
O Connect permite alternar entre grupos da árvore de camadas, ativando apenas um grupo por vez. Isso facilita comparações visuais, análises de cenários e organização geral do projeto.

### 🚀 Funcionalidades Principais
* **Integração Nativa:** Adicione grupos ao Connect diretamente pelo menu de contexto (botão direito).
* **Navegação Ágil:** Use as teclas `↑` e `↓` para alternar entre grupos.
* **Atalho Inteligente:** Tecla `F9` para abrir/fechar o painel rapidamente.
* **Organização Flexível:** Reordene grupos por *drag-and-drop* (arrastar e soltar).
* **Persistência:** Salva automaticamente o estado das listas dentro do seu projeto `.qgs` / `.qgz`.

### 🎮 Como Usar
1. Clique com o botão direito em um grupo na árvore de camadas do QGIS.
2. Selecione *"Connect: Adicionar grupo"*.
3. No painel do Connect, use as setas ou botões para navegar. O plugin ocultará o grupo atual e ativará o próximo automaticamente.

### ⚙️ Compatibilidade
* **QGIS 3.22** ou superior.

---

## 👤 Autora
**Ana Carolina Andrade** Especialista em geoprocessamento e desenvolvimento de soluções geoespaciais.

💡 *Este repositório é atualizado constantemente com novas melhorias e ferramentas.*
