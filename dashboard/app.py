import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Sistema Inteligente COVID-19 | Maestría IA",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    .stMetric {background-color: #ffffff; padding: 14px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.06); border-left: 4px solid #1f77b4;}
    .box-info {background-color: #ffffff; padding: 18px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 12px;}
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")

@st.cache_resource
def cargar_modelos():
    reg_model = joblib.load(os.path.join(BASE_DIR, "modelo_regresion_rf.pkl"))
    clf_model = joblib.load(os.path.join(BASE_DIR, "modelo_clasificacion_rf.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "escalador_robusto.pkl"))
    return reg_model, clf_model, scaler

@st.cache_data
def cargar_datos():
    df_raw = pd.read_csv(os.path.join(DATA_DIR, "time-series-19-covid-combined.csv"))
    df_nac = pd.read_csv(os.path.join(DATA_DIR, "datos_epidemiologicos_limpios.csv"))
    df_nac["Date"] = pd.to_datetime(df_nac["Date"])
    df_clust = pd.read_csv(os.path.join(DATA_DIR, "perfiles_clustering_paises.csv"))
    return df_raw, df_nac, df_clust

rf_reg, rf_clf, scaler = cargar_modelos()
df_raw, df_nacional, df_clusters = cargar_datos()

# ==============================================================================
# ENCABEZADO EJECUTIVO
# ==============================================================================
st.title("🧬 Sistema Inteligente de Monitoreo, Diagnóstico y Predicción Epidemiológica")
st.markdown("**Proyecto Integrador de Ciencia de Datos e Inteligencia Artificial** | Universidad de La Salle | *Dataset: Johns Hopkins University (189 Países)*")

# ==============================================================================
# MÓDULO 1: FILTROS DINÁMICOS EN BARRA LATERAL (SOPORTE GLOBAL Y POR PAÍS)
# ==============================================================================
st.sidebar.header("🎛️ Panel de Control y Filtros")

modo_alcance = st.sidebar.radio(
    "Alcance Territorial:",
    options=["Todos los Países (189 Naciones)", "Seleccionar Países Específicos", "Filtrar por Clúster Epidemiológico"],
    index=0
)

paises_disponibles = sorted(df_nacional["Country/Region"].unique())

if modo_alcance == "Todos los Países (189 Naciones)":
    paises_sel = paises_disponibles
    top_n_chart = st.sidebar.slider("Cantidad de países principales a graficar en series:", 5, 25, 10)
elif modo_alcance == "Seleccionar Países Específicos":
    paises_predeterminados = [
        "US", "India", "Brazil", "Russia", "Colombia",
        "Peru", "Mexico", "Spain", "Italy", "United Kingdom"
    ]
    paises_sel = st.sidebar.multiselect(
        "Elija los países a comparar:",
        options=paises_disponibles,
        default=[pais for pais in paises_predeterminados if pais in paises_disponibles]
    )
    if not paises_sel:
        paises_sel = ["Colombia"]
    top_n_chart = len(paises_sel)
else:
    cluster_sel = st.sidebar.selectbox(
        "Seleccione Perfil de Clúster:",
        options=sorted(df_clusters["Perfil_Cluster"].unique())
    )
    paises_sel = df_clusters[df_clusters["Perfil_Cluster"] == cluster_sel]["Country/Region"].tolist()
    top_n_chart = min(15, len(paises_sel))

fecha_min = df_nacional["Date"].min().date()
fecha_max = df_nacional["Date"].max().date()
rango_fechas = st.sidebar.date_input(
    "Rango Temporal:",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)

umbral_casos = st.sidebar.slider("Contagios diarios mínimos:", 0, 10000, 0, 100)

if len(rango_fechas) == 2:
    f_ini, f_fin = pd.to_datetime(rango_fechas[0]), pd.to_datetime(rango_fechas[1])
else:
    f_ini, f_fin = pd.to_datetime(fecha_min), pd.to_datetime(fecha_max)

df_filtrado = df_nacional[
    (df_nacional["Country/Region"].isin(paises_sel)) &
    (df_nacional["Date"] >= f_ini) &
    (df_nacional["Date"] <= f_fin) &
    (df_nacional["Daily_Confirmed"] >= umbral_casos)
]

# ==============================================================================
# MÓDULO 2: TARJETAS KPI EJECUTIVAS
# ==============================================================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_confirmados = df_filtrado.groupby("Country/Region")["Confirmed"].max().sum()
total_fallecidos = df_filtrado.groupby("Country/Region")["Deaths"].max().sum()
total_recuperados = df_filtrado.groupby("Country/Region")["Recovered"].max().sum()
letalidad_prom = (total_fallecidos / total_confirmados * 100) if total_confirmados > 0 else 0
dias_alto_riesgo = int(df_filtrado["Risk_Level"].sum())

kpi1.metric("Países Analizados", f"{len(paises_sel)} / 189")
kpi2.metric("Confirmados Acumulados", f"{int(total_confirmados):,}")
kpi3.metric("Fallecidos Acumulados", f"{int(total_fallecidos):,}")
kpi4.metric("Recuperados Registrados", f"{int(total_recuperados):,}")
kpi5.metric("Letalidad / Días Alerta", f"{letalidad_prom:.2f}% | {dias_alto_riesgo:,} d")

st.divider()

# ==============================================================================
# PESTAÑAS ESTRUCTURADAS SEGÚN LAS 4 FASES DEL PROYECTO
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧭 1. Planteamiento y Preguntas",
    "🧹 2. Calidad, Limpieza y Comportamiento",
    "🌍 3. Panorama Global (189 Países)",
    "🤖 4. Modelos y Simulador en Vivo",
    "🌐 5. Segmentación (Clustering)"
])

# ------------------------------------------------------------------------------
# PESTAÑA 1: PREGUNTAS DE INVESTIGACIÓN Y TRAZABILIDAD DE FASES
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("🧭 Hilo Conductor Metodológico del Proyecto")
    
    st.markdown("""
    <div class="box-info">
    <h4>🎯 Pregunta de Investigación Principal (Fase 1)</h4>
    <p><i>"¿De qué manera el desarrollo de un sistema de inteligencia artificial y ciencia de datos facilita la estimación anticipada de decesos diarios, la detección temprana de estados de alerta sanitaria y la segmentación estratégica de regiones frente a la propagación de una epidemia global?"</i></p>
    </div>
    """, unsafe_allow_html=True)

    c_q1, c_q2, c_q3 = st.columns(3)
    with c_q1:
        st.info("""**Pregunta Secundaria 1 (Regresión)**

*¿Qué tan efectiva puede ser la predicción de muertes diarias en los territorios a partir de la velocidad de nuevos casos?*

""")
    with c_q2:
        st.warning("""**Pregunta Secundaria 2 (Clasificación)**

*¿Existe la posibilidad de anticipar si una región entrará en un nivel de riesgo epidemiológico alto evaluando el comportamiento de sus casos en las últimas semanas?*

""")
    with c_q3:
        st.success("""**Pregunta Secundaria 3 (Clustering)**

*¿Se pueden agrupar los países en diferentes categorías de impacto según su tasa de letalidad y ritmo de propagación sin ser clasificados previamente?*

""")

    st.markdown("### 📋 Resumen Ejecutivo de lo Ejecutado en Cada Fase")
    fases_df = pd.DataFrame({
        "Fase del Proyecto": [
            "Fase 1: Formulación y EDA",
            "Fase 2: Limpieza e Ingeniería",
            "Fase 3: Modelado Avanzado y DL",
            "Fase 4: Despliegue y Analítica"
        ],
        "Transformación sobre los Datos": [
            "Auditoría de 72.090 filas y 8 variables. Detección de asimetría extrema (>12) y correlación de Spearman (0.902).",
            "Agregación de 72.090 filas a 51.030 registros nacionales limpios. Creación de 5 variables derivadas y partición estratificada sin Data Leakage.",
            "Entrenamiento de Ridge, Random Forest, Gradient Boosting, Red Neuronal MLP en PyTorch, Clustering K-Means y explicabilidad SHAP.",
            "Integración interactiva de los 189 países, simulador predictivo en vivo con modelos serializados y mapas de segmentación."
        ],
        "Hallazgo o Resultado Clave": [
            "69.66% de nulos en Province/State (MNAR estructural) y 5.24% en Recovered (MAR).",
            "Conservación del 100% de outliers epidemiológicos (picos reales) mediante RobustScaler.",
            "Random Forest supera a modelos lineales y redes densas en datos tabulares epidemiológicos.",
            "Sistema web reproducible listo para soporte a decisiones de salud pública."
        ]
    })
    st.dataframe(fases_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# PESTAÑA 2: LIMPIEZA DEL DATASET, LO QUE PREVALECIÓ Y COMPORTAMIENTO
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🧹 Auditoría de Limpieza, Transformación y Comportamiento de los Datos")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros Crudos Iniciales", f"{len(df_raw):,} filas", "Con sub-provincias")
    m2.metric("Registros Limpios Finales", f"{len(df_nacional):,} filas", "189 países x 270 días")
    m3.metric("Nulos Eliminados / Imputados", "54,000 celdas", "0% nulos residuales")
    m4.metric("Variables Totales", "8 → 14 columnas", "+6 variables de dominio")

    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        st.markdown("#### 🔍 Diagnóstico Antes vs. Después de la Limpieza")
        audit_table = pd.DataFrame({
            "Columna / Aspecto": ["Province/State", "Recovered", "Valores Negativos Diarios", "Outliers (> Q3 + 1.5 IQR)", "Variables Predictoras"],
            "Estado Original (Crudo)": ["69.66% Nulos (50.220 filas)", "5.24% Nulos (3.780 filas)", "Rectificaciones negativas", "18.9% (Picos de contagio)", "Solo acumulados monótonos"],
            "Tratamiento Aplicado (Fase 2)": ["Agregación por país (MNAR)", "Interpolación lineal temporal (MAR)", "Recorte inferior en 0 (clip)", "Conservados + RobustScaler", "Flujos diarios, MA7 y Letalidad"]
        })
        st.dataframe(audit_table, use_container_width=True, hide_index=True)
        
        st.info("""**💡 ¿Qué prevaleció en los datos?**

1. **Asimetría Positiva Extrema:** El **77.05%** de los días-país registraron baja mortalidad (≤ 4 muertes/día), mientras que el **22.95%** concentró las olas críticas.
2. **Concentración Geográfica:** Solo **3 países** (EE.UU., India y Brasil) acumularon más del **50%** de todos los contagios globales del período.""")

    with col_clean2:
        st.markdown("#### 📊 Distribución de Severidad Epidemiológica (Lo que Prevaleció)")
        dist_riesgo = df_nacional["Risk_Level"].value_counts().reset_index()
        dist_riesgo.columns = ["Nivel", "Conteo"]
        dist_riesgo["Etiqueta"] = dist_riesgo["Nivel"].map({
            0: "Riesgo Bajo / Moderado (≤ 4 muertes/día) - 77.05%",
            1: "Riesgo Sanitario Alto (> 4 muertes/día) - 22.95%"
        })
        fig_pie = px.pie(
            dist_riesgo, values="Conteo", names="Etiqueta", hole=0.45,
            title="Proporción de Días-País según Nivel de Alerta (51.030 Registros)",
            color_discrete_sequence=["#3498db", "#e74c3c"]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📈 Comportamiento Estadístico: Relación No Lineal entre Contagios y Decesos")
    muestra_scatter = df_nacional[df_nacional["Confirmed"] > 0].sample(min(3000, len(df_nacional)), random_state=42)
    fig_scat = px.scatter(
        muestra_scatter, x="Confirmed", y="Deaths", color="Risk_Level",
        log_x=True, log_y=True, opacity=0.6,
        hover_name="Country/Region", hover_data=["Date", "Fatality_Rate"],
        title="Dispersión Bi-Logarítmica: Casos Confirmados vs. Fallecimientos (Correlación Spearman = 0.902)",
        labels={"Confirmed": "Confirmados (Escala Log)", "Deaths": "Fallecidos (Escala Log)", "Risk_Level": "Alerta"}
    )
    st.plotly_chart(fig_scat, use_container_width=True)

# ------------------------------------------------------------------------------
# PESTAÑA 3: PANORAMA GLOBAL DE TODOS LOS PAÍSES (189 NACIONES)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🌍 Panorama Mundial Completo (189 Países) y Rankings Comparativos")
    
    # Mapa coroplético mundial
    resumen_paises = df_filtrado.groupby("Country/Region").agg({
        "Confirmed": "max",
        "Deaths": "max",
        "Recovered": "max",
        "Daily_Confirmed": "mean",
        "Fatality_Rate": "mean"
    }).reset_index()
    resumen_paises["Tasa_Recuperacion (%)"] = np.where(
        resumen_paises["Confirmed"] > 0, (resumen_paises["Recovered"] / resumen_paises["Confirmed"]) * 100, 0
    )

    fig_map = px.choropleth(
        resumen_paises,
        locations="Country/Region",
        locationmode="country names",
        color="Confirmed",
        hover_name="Country/Region",
        hover_data={"Deaths": ":,", "Recovered": ":,", "Fatality_Rate": ":.2f"},
        color_continuous_scale="YlOrRd",
        title="Mapa Global de Impacto Acumulado (Pasa el cursor sobre cualquier país)"
    )
    fig_map.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    c_rank1, c_rank2 = st.columns(2)
    with c_rank1:
        top_conf = resumen_paises.sort_values("Confirmed", ascending=False).head(15)
        fig_bar1 = px.bar(
            top_conf, x="Confirmed", y="Country/Region", orientation="h",
            color="Deaths", color_continuous_scale="Reds",
            title="Top 15 Países con Mayor Volumen de Contagios",
            labels={"Confirmed": "Casos Confirmados", "Country/Region": "País", "Deaths": "Fallecidos"}
        )
        fig_bar1.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_bar1, use_container_width=True)

    with c_rank2:
        # Evolución temporal de los Top N países del filtro
        top_names = resumen_paises.sort_values("Confirmed", ascending=False).head(top_n_chart)["Country/Region"].tolist()
        df_top_series = df_filtrado[df_filtrado["Country/Region"].isin(top_names)]
        fig_line_top = px.line(
            df_top_series, x="Date", y="Confirmed_MA7", color="Country/Region",
            title=f"Curvas de Velocidad de Contagio (MA7) - Top {len(top_names)} Países",
            labels={"Date": "Fecha", "Confirmed_MA7": "Nuevos Casos (Media 7d)", "Country/Region": "País"}
        )
        st.plotly_chart(fig_line_top, use_container_width=True)

    st.markdown("#### 📑 Tabla Exploratoria Completa de Todos los Países Filtrados")
    st.dataframe(
        resumen_paises.sort_values("Confirmed", ascending=False).style.format({
            "Confirmed": "{:,.0f}", "Deaths": "{:,.0f}", "Recovered": "{:,.0f}",
            "Daily_Confirmed": "{:,.1f}", "Fatality_Rate": "{:.2f}%", "Tasa_Recuperacion (%)": "{:.2f}%"
        }),
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# PESTAÑA 4: COMPARATIVA DE MODELOS Y SIMULADOR EN VIVO
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🤖 Evaluación Comparativa de Modelos (Fase 3) y Simulador en Vivo")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### 📉 Modelos de Regresión (Predicción de Muertes Diarias)")
        df_res_reg = pd.DataFrame({
            "Modelo": ["Baseline Regresión Lineal", "Ridge Regression (L2)", "Random Forest Regressor (Ganador)"],
            "R² Score": [0.6700, 0.6633, 0.9273],
            "RMSE": [72.83, 73.92, 34.34],
            "MAE": [16.32, 16.40, 5.57]
        })
        st.dataframe(df_res_reg, use_container_width=True, hide_index=True)
        
    with col_m2:
        st.markdown("#### 🚨 Modelos de Clasificación y Deep Learning (Riesgo Alto)")
        df_res_clf = pd.DataFrame({
            "Modelo": ["Baseline Logística", "Perceptrón Multicapa (MLP PyTorch)", "Gradient Boosting", "Random Forest Clf (Ganador)"],
            "Accuracy": [0.9058, 0.8868, 0.9482, 0.9446],
            "Recall (Sensibilidad)": [0.8687, 0.8526, 0.8796, 0.9381],
            "F1-Score": [0.7997, 0.7930, 0.8863, 0.8861],
            "AUC-ROC": [0.9593, 0.9739, 0.9881, 0.9872]
        })
        st.dataframe(df_res_clf, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🔮 Simulador Predictivo Interactivo en Tiempo Real")
    st.caption("Conectado directamente a los modelos Random Forest entrenados en la Fase 3. Modifique los valores o cargue un escenario real:")

    with st.form("form_prediccion"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_confirmed = st.number_input("Casos Confirmados Acumulados", min_value=0, value=250000, step=10000)
            in_recovered = st.number_input("Recuperados Acumulados", min_value=0, value=180000, step=10000)
            in_daily_conf = st.number_input("Nuevos Casos Hoy", min_value=0, value=4500, step=100)
        with c2:
            in_daily_rec = st.number_input("Nuevos Recuperados Hoy", min_value=0, value=3800, step=100)
            in_ma7 = st.number_input("Promedio Casos Últimos 7 Días (MA7)", min_value=0, value=4100, step=100)
            in_fatality = st.slider("Tasa de Letalidad Actual (%)", min_value=0.0, max_value=15.0, value=3.4, step=0.1)
        with c3:
            in_lat = st.number_input("Latitud Geográfica", min_value=-60.0, max_value=75.0, value=4.57)
            in_long = st.number_input("Longitud Geográfica", min_value=-140.0, max_value=180.0, value=-74.29)
        boton_predecir = st.form_submit_button("🚀 Predecir Mortalidad y Nivel de Riesgo")

    if boton_predecir:
        in_active = max(0, in_confirmed - in_recovered - (in_confirmed * (in_fatality / 100)))
        in_growth = np.clip((in_daily_conf / in_ma7) if in_ma7 > 0 else 1.0, 0.0, 10.0)
        input_data = np.array([[
            in_confirmed, in_recovered, in_daily_conf, in_daily_rec,
            in_fatality, in_ma7, in_active, in_growth, in_lat, in_long
        ]])
        input_scaled = scaler.transform(input_data)
        pred_muertes = rf_reg.predict(input_scaled)[0]
        pred_riesgo = rf_clf.predict(input_scaled)[0]
        prob_riesgo = rf_clf.predict_proba(input_scaled)[0][1] * 100

        r1, r2 = st.columns(2)
        with r1:
            st.metric("📉 Estimación de Fallecimientos Diarios (Regresión)", f"{int(round(pred_muertes))} decesos / día", "Margen MAE: ±5.5")
        with r2:
            if pred_riesgo == 1:
                st.error(f"🚨 **ALERTA SANITARIA: RIESGO ALTO** (Probabilidad: {prob_riesgo:.1f}%)")
            else:
                st.success(f"✅ **ESTADO CONTROLADO: RIESGO BAJO/MODERADO** (Probabilidad Alerta: {prob_riesgo:.1f}%)")

# ------------------------------------------------------------------------------
# PESTAÑA 5: SEGMENTACIÓN NO SUPERVISADA (CLUSTERING)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("🌐 Segmentación Territorial No Supervisada (K-Means + PCA)")
    st.write("Agrupación automática de los **189 países** sin etiquetas previas, validada mediante Método del Codo y Coeficiente de Silueta ($0.8827$ para $K=3$).")

    fig_clust = px.scatter(
        df_clusters, x="PCA_1", y="PCA_2", color="Perfil_Cluster", size="Confirmed",
        hover_name="Country/Region",
        hover_data={"Daily_Confirmed": ":.1f", "Daily_Deaths": ":.1f", "Fatality_Rate": ":.2f", "PCA_1": False, "PCA_2": False},
        title="Proyección Espacial PCA (99.87% Varianza Explicada) de los 3 Perfiles Epidemiológicos",
        color_discrete_sequence=["#2ecc71", "#e74c3c", "#f39c12"]
    )
    fig_clust.update_layout(height=480)
    st.plotly_chart(fig_clust, use_container_width=True)

    c_cl1, c_cl2 = st.columns([1.2, 1])
    with c_cl1:
        st.markdown("#### 📋 Características Promedio de Cada Clúster")
        resumen_clust = df_clusters.groupby("Perfil_Cluster")[["Daily_Confirmed", "Daily_Deaths", "Fatality_Rate", "Confirmed"]].mean().round(2)
        resumen_clust["Total_Paises"] = df_clusters["Perfil_Cluster"].value_counts()
        st.dataframe(resumen_clust, use_container_width=True)
    with c_cl2:
        st.markdown("#### 🔎 Consultar Países por Grupo")
        grupo_ver = st.selectbox("Ver integrantes del grupo:", sorted(df_clusters["Perfil_Cluster"].unique()))
        paises_grupo = df_clusters[df_clusters["Perfil_Cluster"] == grupo_ver]["Country/Region"].sort_values().tolist()
        st.write(f"**Total en este clúster:** {len(paises_grupo)} países")
        st.write(", ".join(paises_grupo))
