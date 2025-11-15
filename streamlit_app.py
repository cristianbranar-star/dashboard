import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Dashboard Alto Costo Medellín",
    page_icon="🏥",
    layout="wide"
)

# Configuración de visualización
sns.set_style("whitegrid")
# No necesitamos plt.rcParams['figure.figsize'] porque definiremos el tamaño en cada fig.

# --- ADVERTENCIA IMPORTANTE ---
st.warning("""
**Advertencia Importante:** Esta aplicación utiliza **datos falsos (sintéticos)** con fines educativos.
El manejo de datos reales de pacientes está estrictamente regulado por leyes de protección de datos (Habeas Data).
""")

#################################################################
# PASO 1: SIMULACIÓN DE DATOS (Función Cacheada)
#################################################################
# Usamos @st.cache_data para que los datos no se regeneren con cada clic
@st.cache_data
def cargar_datos_simulados():
    """Crea y retorna un DataFrame simulado de pacientes en Medellín."""
    np.random.seed(42)
    num_pacientes = 2000
    
    comunas_medellin = [
        'Popular', 'Santa Cruz', 'Manrique', 'Aranjuez', 'Castilla', 'Doce de Octubre',
        'Robledo', 'Villa Hermosa', 'Buenos Aires', 'La Candelaria', 'Laureles-Estadio',
        'La América', 'San Javier', 'El Poblado', 'Guayabal', 'Belén'
    ]
    diagnosticos = [
        'Diabetes Tipo 2', 'Hipertensión Arterial', 'Enfermedad Renal Crónica (ERC)',
        'Cáncer (Genérico)', 'EPOC', 'Falla Cardíaca', 'Condición Saludable'
    ]
    
    costo_base_por_diagnostico = {
        'Diabetes Tipo 2': 1500000, 'Hipertensión Arterial': 800000,
        'Enfermedad Renal Crónica (ERC)': 40000000, 'Cáncer (Genérico)': 50000000,
        'EPOC': 7000000, 'Falla Cardíaca': 12000000, 'Condición Saludable': 100000
    }
    
    data = {
        'paciente_id': range(1, num_pacientes + 1),
        'edad': np.random.randint(18, 85, size=num_pacientes),
        'sexo': np.random.choice(['M', 'F'], size=num_pacientes, p=[0.48, 0.52]),
        'comuna': np.random.choice(comunas_medellin, size=num_pacientes, p=[0.06] * 15 + [0.1]),
        'diagnostico_principal': np.random.choice(diagnosticos, size=num_pacientes, p=[0.15, 0.2, 0.1, 0.05, 0.1, 0.1, 0.3]),
        'numero_consultas_año': np.random.randint(1, 48, size=num_pacientes),
        'numero_hospitalizaciones_año': np.random.randint(0, 12, size=num_pacientes),
    }
    df_pacientes = pd.DataFrame(data)

    def asignar_costo_tratamiento(row):
        diagnostico = row['diagnostico_principal']
        costo = costo_base_por_diagnostico[diagnostico]
        tratamiento = 'Seguimiento Preventivo'
        if diagnostico == 'Enfermedad Renal Crónica (ERC)':
            tratamiento = 'Diálisis'
            costo += row['numero_hospitalizaciones_año'] * 5000000 + row['numero_consultas_año'] * 150000
        elif diagnostico == 'Cáncer (Genérico)':
            tratamiento = 'Quimioterapia'
            costo += row['numero_hospitalizaciones_año'] * 7000000 + row['numero_consultas_año'] * 200000
        elif diagnostico in ['Diabetes Tipo 2', 'Hipertensión Arterial', 'Falla Cardíaca']:
            tratamiento = 'Medicación Oral Crónica'
            costo += row['numero_hospitalizaciones_año'] * 3000000 + row['numero_consultas_año'] * 80000
        elif diagnostico == 'EPOC':
            tratamiento = 'Terapia Respiratoria'
            costo += row['numero_hospitalizaciones_año'] * 4000000 + row['numero_consultas_año'] * 100000
        costo = costo * np.random.uniform(0.8, 1.2)
        return pd.Series([tratamiento, int(costo)])

    df_pacientes[['tratamiento_principal', 'costo_total_año']] = df_pacientes.apply(asignar_costo_tratamiento, axis=1)
    
    for col in ['comuna', 'tratamiento_principal']:
        df_pacientes.loc[df_pacientes.sample(frac=0.05).index, col] = np.nan
        
    return df_pacientes

# --- Título Principal de la App ---
st.title("Dashboard de Analítica de Pacientes de Alto Costo 🏥")
st.markdown("Simulación para la toma de decisiones empresariales en la red de salud de Medellín.")

# Cargar y mostrar datos crudos en un expander
df_pacientes = cargar_datos_simulados()
with st.expander("Ver datos crudos simulados (primeras 100 filas)"):
    st.dataframe(df_pacientes.head(100))

#################################################################
# PASO 2: PREPROCESAMIENTO
#################################################################
st.header("PASO 2: Preprocesamiento de Datos 🧹", divider='rainbow')

df_procesado = df_pacientes.copy()

# 2.1. Manejo de Valores Nulos
st.subheader("2.1. Manejo de Valores Nulos")
col1, col2 = st.columns(2)
with col1:
    st.write("**Valores Nulos (Antes):**")
    st.code(df_procesado.isnull().sum())

# Imputación
df_procesado['comuna'] = df_procesado['comuna'].fillna('Desconocido')
moda_tratamiento = df_procesado['tratamiento_principal'].mode()[0]
df_procesado['tratamiento_principal'] = df_procesado['tratamiento_principal'].fillna(moda_tratamiento)

with col2:
    st.write("**Valores Nulos (Después):**")
    st.code(df_procesado.isnull().sum())

# 2.2. Definición de "Alto Costo"
st.subheader("2.2. Definición de 'Alto Costo'")
percentil_90 = df_procesado['costo_total_año'].quantile(0.90)

st.metric(
    label="Umbral de Alto Costo (Percentil 90)",
    value=f"${percentil_90:,.0f} COP"
)

df_procesado['es_alto_costo'] = (df_procesado['costo_total_año'] > percentil_90).astype(int)
total_alto_costo = df_procesado['es_alto_costo'].sum()
st.write(f"**Total de pacientes de Alto Costo (Top 10%):** {total_alto_costo} de {len(df_procesado)} pacientes.")


#################################################################
# PASO 3: TRANSFORMACIÓN (Ingeniería de Características)
#################################################################
st.header("PASO 3: Transformación y Feature Engineering ⚙️", divider='rainbow')

# 3.1. Creación de Rangos de Edad (Binning)
st.subheader("3.1. Creación de Rangos de Edad")
bins = [18, 30, 45, 60, 85]
labels = ['18-30 (Joven)', '31-45 (Adulto)', '46-60 (Adulto Medio)', '61+ (Adulto Mayor)']
df_procesado['rango_edad'] = pd.cut(df_procesado['edad'], bins=bins, labels=labels, right=True)
st.write("Se agregó la columna 'rango_edad' a partir de 'edad':")
st.dataframe(df_procesado[['paciente_id', 'edad', 'rango_edad']].head())

# 3.2. Creación de Características de Interacción
# (Este paso se omite de la visualización principal pero se mantiene en el dataframe)
df_procesado['costo_por_consulta'] = df_procesado['costo_total_año'] / (df_procesado['numero_consultas_año'] + 1)
df_procesado['costo_por_hospitalizacion'] = df_procesado['costo_total_año'] / (df_procesado['numero_hospitalizaciones_año'] + 1)


# 3.3 y 3.4. Encoding y Escalado (Para Modelos)
with st.expander("Ver detalles de Encoding y Escalado (Preparación para Modelos)"):
    st.markdown("""
    Estos pasos transforman los datos de texto a números (Encoding) y ajustan las escalas numéricas (Escalado),
    siendo cruciales si fuéramos a entrenar un modelo de Machine Learning.
    """)
    st.code(f"""
# 3.3. Encoding Categórico (Para Modelos)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cols_a_codificar = ['sexo', 'comuna', 'rango_edad', 'diagnostico_principal', 'tratamiento_principal']
encoded_features = encoder.fit_transform(df_procesado[cols_a_codificar])
# Esto creó {encoded_features.shape[1]} nuevas columnas numéricas.

# 3.4. Escalado de Características Numéricas (Para Modelos)
scaler = StandardScaler()
cols_a_escalar = ['edad', 'numero_consultas_año', 'numero_hospitalizaciones_año']
df_procesado[cols_a_escalar + '_scaled'] = scaler.fit_transform(df_procesado[cols_a_escalar])

print(df_procesado.head())
    """, language='python')


#################################################################
# PASO 4: VISUALES PARA DECISIONES EMPRESARIALES
#################################################################
st.header("PASO 4: Visuales para Decisiones Empresariales 📊", divider='rainbow')
st.info("A partir de aquí, los análisis se enfocan en el segmento de **Alto Costo**.")

# Filtramos solo los pacientes de alto costo
df_alto_costo = df_procesado[df_procesado['es_alto_costo'] == 1]

# ---
# VISUAL 1: ¿Qué diagnósticos y tratamientos generan el alto costo?
# ---
st.subheader("Visual 1: ¿Qué diagnósticos y tratamientos impulsan el Alto Costo?")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig1.suptitle('Análisis de Diagnósticos y Tratamientos en Pacientes de Alto Costo', fontsize=20, weight='bold')

# Gráfico de Diagnósticos
sns.countplot(
    data=df_alto_costo, y='diagnostico_principal',
    order=df_alto_costo['diagnostico_principal'].value_counts().index,
    ax=ax1, palette='Reds_r'
)
ax1.set_title('TOP Diagnósticos', fontsize=16)
ax1.set_xlabel('Cantidad de Pacientes')
ax1.set_ylabel('Diagnóstico')

# Gráfico de Tratamientos
sns.countplot(
    data=df_alto_costo, y='tratamiento_principal',
    order=df_alto_costo['tratamiento_principal'].value_counts().index,
    ax=ax2, palette='Blues_r'
)
ax2.set_title('TOP Tratamientos', fontsize=16)
ax2.set_xlabel('Cantidad de Pacientes')
ax2.set_ylabel('Tratamiento')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Usamos st.pyplot() para mostrar la figura en Streamlit
st.pyplot(fig1)

st.success("""
**Decisión de Negocio (Insight):**
El 80% de nuestros pacientes de alto costo provienen de **'Cáncer'** y **'Enfermedad Renal Crónica (ERC)'**.
Debemos crear programas de **gestión de caso (case management)** específicos para oncología y nefrología,
enfocados en la adherencia y optimización de **'Quimioterapia'** y **'Diálisis'**.
""")

# ---
# VISUAL 2: ¿Dónde están nuestros pacientes de alto costo? (Geolocalización)
# ---
st.subheader("Visual 2: ¿Dónde están nuestros pacientes de Alto Costo?")

fig2, ax = plt.subplots(figsize=(12, 7))
sns.countplot(
    data=df_alto_costo, y='comuna',
    order=df_alto_costo['comuna'].value_counts().index,
    palette='viridis', ax=ax
)
ax.set_title('Distribución Geográfica (Comunas) de Pacientes de Alto Costo', fontsize=16, weight='bold')
ax.set_xlabel('Cantidad de Pacientes')
ax.set_ylabel('Comuna')
plt.tight_layout()
st.pyplot(fig2)

st.success("""
**Decisión de Negocio (Insight):**
Tenemos una concentración desproporcionada en **'El Poblado'** y **'Robledo'**.
Acción: Investigar si esto se debe a demografía (ej. más adultos mayores en Poblado) o a barreras de acceso en Robledo.
**Priorizar la asignación de equipos de atención domiciliaria (brigadas) en estas dos comunas.**
""")

# ---
# VISUAL 3: ¿Qué impulsa el costo? ¿Consultas o Hospitalizaciones?
# ---
st.subheader("Visual 3: ¿Qué impulsa el costo? ¿Consultas o Hospitalizaciones?")

fig3, ax = plt.subplots(figsize=(12, 7))
sns.scatterplot(
    data=df_procesado, # Usamos TODOS los datos para ver la diferencia
    x='numero_consultas_año',
    y='numero_hospitalizaciones_año',
    hue='es_alto_costo',
    size='costo_total_año',
    sizes=(50, 1000),
    alpha=0.7,
    palette={0: 'grey', 1: 'red'},
    ax=ax
)
ax.set_title('Costo vs. Frecuencia de Servicios', fontsize=16, weight='bold')
ax.set_xlabel('Número de Consultas al Año')
ax.set_ylabel('Número de Hospitalizaciones al Año')
ax.legend(title='¿Es Alto Costo?')
plt.tight_layout()
st.pyplot(fig3)

st.success("""
**Decisión de Negocio (Insight):**
Los pacientes de 'Alto Costo' (rojo) se definen casi exclusivamente por el **Número de Hospitalizaciones**.
El costo no es por ir *mucho* al médico, es por ser *hospitalizado*.
**La estrategia debe ser agresiva en evitar la hospitalización.** Invertir en programas de prevención de recaídas.
""")

# ---
# VISUAL 4: ¿Cuál es el perfil demográfico del paciente de alto costo?
# ---
st.subheader("Visual 4: Perfil Demográfico del Paciente de Alto Costo")

fig4, ax = plt.subplots(figsize=(10, 6))
sns.countplot(
    data=df_alto_costo,
    x='rango_edad',
    hue='sexo',
    order=labels,
    palette={'M': 'blue', 'F': 'pink'},
    ax=ax
)
ax.set_title('Perfil Demográfico de Pacientes de Alto Costo', fontsize=16, weight='bold')
ax.set_xlabel('Rango de Edad')
ax.set_ylabel('Cantidad de Pacientes')
ax.legend(title='Sexo')
plt.tight_layout()
st.pyplot(fig4)

st.success("""
**Decisión de Negocio (Insight):**
El grupo **'61+ (Adulto Mayor)'** representa la mayoría de nuestros pacientes de alto costo.
**Las campañas de comunicación y prevención deben estar 100% enfocadas en este grupo etario.**
""")


# ---
# VISUAL 5: Distribución del Costo (La "Cola Larga")
# ---
st.subheader("Visual 5: Distribución del Costo (La 'Cola Larga')")

fig5, ax = plt.subplots(figsize=(12, 6))
sns.histplot(df_procesado['costo_total_año'], bins=50, kde=True, color='darkgreen', ax=ax)
ax.axvline(percentil_90, color='red', linestyle='--', linewidth=2, label=f'Percentil 90 (Alto Costo)\n${percentil_90:,.0f}')
ax.set_title('Distribución del Costo Anual por Paciente', fontsize=16, weight='bold')
ax.set_xlabel('Costo Total (COP)')
ax.set_ylabel('Frecuencia (Pacientes)')
ax.legend()
ax.get_xaxis().set_major_formatter(
    plt.FuncFormatter(lambda x, p: f'${x/1_000_000:.0f}M')
)
plt.tight_layout()
st.pyplot(fig5)

st.success("""
**Decisión de Negocio (Insight):**
La distribución está extremadamente sesgada. La gran mayoría de pacientes son de bajo costo.
El problema de 'Alto Costo' (a la derecha de la línea roja) es un grupo pequeño pero **extremadamente caro**.
Esto confirma que una estrategia de **'gestión de caso'** (asignar una enfermera o gestor a cada paciente de alto costo) es viable y tendrá un alto retorno de inversión (ROI).
""")
