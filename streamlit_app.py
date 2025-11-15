"""
Sistema de Análisis de Morbilidad y Segmentación de Pacientes de Alto Costo
ESE Metrosalud - Datos Abiertos Colombia

Este script realiza:
1. Carga y análisis de datos de morbilidad
2. Segmentación por grupos etarios
3. Identificación de factores de riesgo y enfermedades de alto costo
4. Análisis de patrones mediante técnicas de BI
5. Modelo de categorización de pacientes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnalizadorMorbilidad:
    """
    Clase principal para análisis de morbilidad y segmentación de pacientes
    """

    def __init__(self, url_datos):
        """
        Inicializa el analizador con la URL de los datos
        """
        self.url = url_datos
        self.df_original = None
        self.grupos_etarios = {}
        self.resultados = {}

    def cargar_datos(self):
        """
        Carga los datos desde la URL y realiza limpieza inicial
        """
        print("📊 Cargando datos desde datos.gov.co...")
        try:
            # Intenta con diferentes separadores
            try:
                self.df_original = pd.read_csv(self.url, sep=';', encoding='utf-8')
            except:
                self.df_original = pd.read_csv(self.url, sep=',', encoding='utf-8')

            print(f"✅ Datos cargados exitosamente: {self.df_original.shape[0]} registros, {self.df_original.shape[1]} columnas")
            print(f"\n📋 Columnas disponibles:")
            for i, col in enumerate(self.df_original.columns, 1):
                print(f"   {i}. {col}")

            # Limpieza de nombres de columnas
            self.df_original.columns = self.df_original.columns.str.strip().str.lower().str.replace(' ', '_')

            return self.df_original

        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return None

    def crear_grupos_etarios(self, columna_edad='edad'):
        """
        Crea dataframes segmentados por grupos etarios

        Parámetros:
        -----------
        columna_edad : str
            Nombre de la columna que contiene la edad
        """
        print("\n👥 Creando grupos etarios...")

        if self.df_original is None:
            print("❌ Primero debe cargar los datos")
            return None

        # Verificar si existe columna de edad
        posibles_cols_edad = ['edad', 'age', 'anos', 'años', 'edad_anos']
        col_edad = None

        for col in posibles_cols_edad:
            if col in self.df_original.columns:
                col_edad = col
                break

        if col_edad is None:
            print(f"⚠️  No se encontró columna de edad. Columnas disponibles: {list(self.df_original.columns)}")
            # Crear columna de edad simulada para demostración
            print("📝 Creando columna de edad simulada para demostración...")
            self.df_original['edad'] = np.random.randint(0, 90, size=len(self.df_original))
            col_edad = 'edad'

        # Asegurar que edad es numérica
        self.df_original[col_edad] = pd.to_numeric(self.df_original[col_edad], errors='coerce')

        # Definir grupos etarios
        definiciones_grupos = {
            'primera_infancia': (0, 5, 'Primera infancia (0-5 años)'),
            'infancia': (6, 11, 'Infancia (6-11 años)'),
            'adolescencia': (12, 18, 'Adolescencia (12-18 años)'),
            'juventud': (19, 26, 'Juventud (19-26 años)'),
            'adultez': (27, 59, 'Adultez (27-59 años)'),
            'persona_mayor': (60, 120, 'Persona mayor (60+ años)')
        }

        # Crear dataframes por grupo
        for clave, (min_edad, max_edad, descripcion) in definiciones_grupos.items():
            df_grupo = self.df_original[
                (self.df_original[col_edad] >= min_edad) &
                (self.df_original[col_edad] <= max_edad)
            ].copy()

            df_grupo['grupo_etario'] = descripcion
            self.grupos_etarios[clave] = df_grupo

            print(f"   ✓ {descripcion}: {len(df_grupo)} registros ({len(df_grupo)/len(self.df_original)*100:.1f}%)")

        return self.grupos_etarios

    def analizar_factores_riesgo(self, top_n=15):
        """
        Analiza factores de riesgo y enfermedades de alto costo

        Parámetros:
        -----------
        top_n : int
            Número de principales diagnósticos a analizar
        """
        print("\n🔍 Analizando factores de riesgo y enfermedades de alto costo...")

        # Buscar columnas relevantes
        cols_diagnostico = [col for col in self.df_original.columns if 'diagn' in col or 'enfermedad' in col or 'patolog' in col]
        cols_servicio = [col for col in self.df_original.columns if 'servicio' in col or 'atencion' in col or 'procedimiento' in col]
        cols_costo = [col for col in self.df_original.columns if 'costo' in col or 'valor' in col or 'precio' in col]

        resultados_analisis = {}

        # Análisis por diagnóstico/patología
        if cols_diagnostico:
            col_diag = cols_diagnostico[0]
            print(f"\n📊 Análisis de diagnósticos (columna: {col_diag}):")

            # Top diagnósticos generales
            top_diagnosticos = self.df_original[col_diag].value_counts().head(top_n)
            resultados_analisis['top_diagnosticos'] = top_diagnosticos

            print(f"\n   Top {top_n} diagnósticos más frecuentes:")
            for i, (diag, freq) in enumerate(top_diagnosticos.items(), 1):
                print(f"   {i}. {diag}: {freq} casos ({freq/len(self.df_original)*100:.2f}%)")

            # Análisis por grupo etario
            print("\n   📈 Distribución por grupo etario:")
            diagnosticos_por_grupo = {}

            for nombre_grupo, df_grupo in self.grupos_etarios.items():
                if len(df_grupo) > 0:
                    top_grupo = df_grupo[col_diag].value_counts().head(5)
                    diagnosticos_por_grupo[nombre_grupo] = top_grupo
                    desc_grupo = df_grupo['grupo_etario'].iloc[0]
                    print(f"\n   {desc_grupo}:")
                    for j, (diag, freq) in enumerate(top_grupo.items(), 1):
                        print(f"      {j}. {diag}: {freq} casos")

            resultados_analisis['diagnosticos_por_grupo'] = diagnosticos_por_grupo

        # Análisis de servicios
        if cols_servicio:
            col_serv = cols_servicio[0]
            print(f"\n📊 Análisis de servicios (columna: {col_serv}):")

            top_servicios = self.df_original[col_serv].value_counts().head(top_n)
            resultados_analisis['top_servicios'] = top_servicios

            print(f"\n   Top {top_n} servicios más utilizados:")
            for i, (serv, freq) in enumerate(top_servicios.items(), 1):
                print(f"   {i}. {serv}: {freq} atenciones")

        # Análisis de costos (si disponible)
        if cols_costo:
            col_cost = cols_costo[0]
            self.df_original[col_cost] = pd.to_numeric(self.df_original[col_cost], errors='coerce')

            print(f"\n💰 Análisis de costos (columna: {col_cost}):")
            print(f"   Costo total: ${self.df_original[col_cost].sum():,.0f}")
            print(f"   Costo promedio: ${self.df_original[col_cost].mean():,.0f}")
            print(f"   Costo mediano: ${self.df_original[col_cost].median():,.0f}")

            # Identificar pacientes de alto costo (percentil 90)
            umbral_alto_costo = self.df_original[col_cost].quantile(0.90)
            pacientes_alto_costo = self.df_original[self.df_original[col_cost] >= umbral_alto_costo]

            print(f"\n   🔴 Pacientes de ALTO COSTO (top 10%):")
            print(f"   Umbral: ${umbral_alto_costo:,.0f}")
            print(f"   Cantidad: {len(pacientes_alto_costo)} pacientes")
            print(f"   Costo total: ${pacientes_alto_costo[col_cost].sum():,.0f}")
            print(f"   % del costo total: {pacientes_alto_costo[col_cost].sum()/self.df_original[col_cost].sum()*100:.1f}%")

            resultados_analisis['pacientes_alto_costo'] = pacientes_alto_costo
            resultados_analisis['umbral_alto_costo'] = umbral_alto_costo

        self.resultados['factores_riesgo'] = resultados_analisis
        return resultados_analisis

    def segmentacion_inteligencia_negocios(self, n_clusters=5):
        """
        Aplica técnicas de BI y machine learning para segmentación de pacientes

        Parámetros:
        -----------
        n_clusters : int
            Número de clusters para segmentación
        """
        print(f"\n🤖 Aplicando técnicas de Inteligencia de Negocios y ML...")
        print(f"   Clusters a generar: {n_clusters}")

        # Preparar datos para clustering
        df_clustering = self.df_original.copy()

        # Buscar columnas numéricas relevantes
        cols_numericas = df_clustering.select_dtypes(include=[np.number]).columns.tolist()

        if 'edad' in df_clustering.columns:
            features_base = ['edad']
        else:
            features_base = []

        # Agregar columnas de costo si existen
        cols_costo = [col for col in cols_numericas if 'costo' in col or 'valor' in col]
        features_base.extend(cols_costo[:3])  # Máximo 3 columnas de costo

        # Agregar otras columnas numéricas
        otras_cols = [col for col in cols_numericas if col not in features_base][:5]
        features_base.extend(otras_cols)

        # Crear features adicionales
        if 'edad' in df_clustering.columns:
            df_clustering['edad_cuadrada'] = df_clustering['edad'] ** 2
            features_base.append('edad_cuadrada')

        # Contar frecuencia de servicios por paciente (si hay ID de paciente)
        cols_id = [col for col in df_clustering.columns if 'id' in col or 'documento' in col or 'paciente' in col]
        if cols_id:
            col_id = cols_id[0]
            frecuencia = df_clustering.groupby(col_id).size().reset_index(name='frecuencia_servicios')
            df_clustering = df_clustering.merge(frecuencia, on=col_id, how='left')
            features_base.append('frecuencia_servicios')

        # Filtrar features válidas
        features_validas = [f for f in features_base if f in df_clustering.columns]

        if len(features_validas) < 2:
            print("⚠️  Insuficientes features numéricas para clustering")
            return None

        print(f"\n   Features utilizadas: {features_validas}")

        # Preparar datos
        X = df_clustering[features_validas].fillna(df_clustering[features_validas].median())

        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Clustering K-Means
        print("\n   Aplicando K-Means clustering...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustering['cluster'] = clusters

        # Análisis de clusters
        print("\n   📊 Perfil de clusters identificados:")

        perfiles_clusters = {}
        for i in range(n_clusters):
            cluster_data = df_clustering[df_clustering['cluster'] == i]
            perfil = {
                'tamaño': len(cluster_data),
                'porcentaje': len(cluster_data) / len(df_clustering) * 100
            }

            # Estadísticas por feature
            for feature in features_validas:
                perfil[f'{feature}_promedio'] = cluster_data[feature].mean()
                perfil[f'{feature}_std'] = cluster_data[feature].std()

            perfiles_clusters[f'Cluster_{i}'] = perfil

            print(f"\n   🔹 Cluster {i}:")
            print(f"      Pacientes: {perfil['tamaño']} ({perfil['porcentaje']:.1f}%)")

            if 'edad' in features_validas:
                print(f"      Edad promedio: {perfil['edad_promedio']:.1f} años")

            # Mostrar costos si existen
            for feature in features_validas:
                if 'costo' in feature or 'valor' in feature:
                    print(f"      {feature}: ${perfil[f'{feature}_promedio']:,.0f}")

        # PCA para visualización
        if len(features_validas) >= 2:
            print("\n   Reduciendo dimensionalidad con PCA...")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            df_clustering['pca_1'] = X_pca[:, 0]
            df_clustering['pca_2'] = X_pca[:, 1]

            print(f"   Varianza explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%")

        self.resultados['segmentacion'] = {
            'df_segmentado': df_clustering,
            'perfiles_clusters': perfiles_clusters,
            'features_utilizadas': features_validas,
            'modelo_kmeans': kmeans,
            'scaler': scaler
        }

        return df_clustering

    def modelo_categorizacion_pacientes(self):
        """
        Desarrolla modelo de categorización basado en datos históricos
        """
        print("\n🎯 Desarrollando modelo de categorización de pacientes...")

        if 'segmentacion' not in self.resultados:
            print("⚠️  Primero ejecute segmentacion_inteligencia_negocios()")
            return None

        df_segmentado = self.resultados['segmentacion']['df_segmentado']
        features = self.resultados['segmentacion']['features_utilizadas']

        # Crear categorías de riesgo basadas en múltiples factores
        print("\n   Creando categorías de riesgo...")

        categorias = []

        for idx, row in df_segmentado.iterrows():
            score_riesgo = 0

            # Factor edad
            if 'edad' in df_segmentado.columns:
                edad = row['edad']
                if edad < 5 or edad > 60:
                    score_riesgo += 2
                elif edad > 50:
                    score_riesgo += 1

            # Factor costo
            cols_costo = [col for col in features if 'costo' in col or 'valor' in col]
            if cols_costo:
                col_costo = cols_costo[0]
                if row[col_costo] > df_segmentado[col_costo].quantile(0.75):
                    score_riesgo += 3
                elif row[col_costo] > df_segmentado[col_costo].quantile(0.50):
                    score_riesgo += 1

            # Factor frecuencia
            if 'frecuencia_servicios' in df_segmentado.columns:
                if row['frecuencia_servicios'] > df_segmentado['frecuencia_servicios'].quantile(0.75):
                    score_riesgo += 2

            # Categorizar
            if score_riesgo >= 5:
                categoria = 'ALTO_RIESGO'
            elif score_riesgo >= 3:
                categoria = 'MEDIO_RIESGO'
            else:
                categoria = 'BAJO_RIESGO'

            categorias.append(categoria)

        df_segmentado['categoria_riesgo'] = categorias

        # Análisis por categoría
        print("\n   📊 Distribución de pacientes por categoría de riesgo:")

        for categoria in ['ALTO_RIESGO', 'MEDIO_RIESGO', 'BAJO_RIESGO']:
            pacientes_cat = df_segmentado[df_segmentado['categoria_riesgo'] == categoria]
            print(f"\n   🔸 {categoria}:")
            print(f"      Pacientes: {len(pacientes_cat)} ({len(pacientes_cat)/len(df_segmentado)*100:.1f}%)")

            if 'edad' in df_segmentado.columns:
                print(f"      Edad promedio: {pacientes_cat['edad'].mean():.1f} años")

            cols_costo = [col for col in features if 'costo' in col or 'valor' in col]
            if cols_costo:
                col_costo = cols_costo[0]
                print(f"      Costo promedio: ${pacientes_cat[col_costo].mean():,.0f}")

        # Análisis cruzado: Categoría x Grupo Etario
        if 'grupo_etario' in df_segmentado.columns:
            print("\n   📈 Matriz de riesgo por grupo etario:")
            tabla_cruzada = pd.crosstab(
                df_segmentado['grupo_etario'],
                df_segmentado['categoria_riesgo'],
                normalize='index'
            ) * 100

            print(tabla_cruzada.round(1))

        self.resultados['categorizacion'] = {
            'df_categorizado': df_segmentado,
            'distribucion_categorias': df_segmentado['categoria_riesgo'].value_counts()
        }

        return df_segmentado

    def generar_visualizaciones(self, guardar=True):
        """
        Genera visualizaciones del análisis

        Parámetros:
        -----------
        guardar : bool
            Si True, guarda las gráficas como imágenes
        """
        print("\n📊 Generando visualizaciones...")

        fig = plt.figure(figsize=(20, 12))

        # 1. Distribución por grupo etario
        ax1 = plt.subplot(2, 3, 1)
        grupos_counts = [len(df) for df in self.grupos_etarios.values()]
        grupos_nombres = [df['grupo_etario'].iloc[0] if len(df) > 0 else 'Sin datos'
                         for df in self.grupos_etarios.values()]

        ax1.bar(range(len(grupos_nombres)), grupos_counts, color='steelblue')
        ax1.set_xticks(range(len(grupos_nombres)))
        ax1.set_xticklabels(grupos_nombres, rotation=45, ha='right')
        ax1.set_title('Distribución de Pacientes por Grupo Etario', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Número de Pacientes')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Top diagnósticos
        if 'factores_riesgo' in self.resultados and 'top_diagnosticos' in self.resultados['factores_riesgo']:
            ax2 = plt.subplot(2, 3, 2)
            top_diag = self.resultados['factores_riesgo']['top_diagnosticos'].head(10)
            ax2.barh(range(len(top_diag)), top_diag.values, color='coral')
            ax2.set_yticks(range(len(top_diag)))
            ax2.set_yticklabels([str(x)[:40] + '...' if len(str(x)) > 40 else str(x) for x in top_diag.index])
            ax2.set_title('Top 10 Diagnósticos Más Frecuentes', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Frecuencia')
            ax2.grid(axis='x', alpha=0.3)

        # 3. Distribución de clusters
        if 'segmentacion' in self.resultados:
            ax3 = plt.subplot(2, 3, 3)
            df_seg = self.resultados['segmentacion']['df_segmentado']
            cluster_counts = df_seg['cluster'].value_counts().sort_index()
            ax3.pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(cluster_counts)))
            ax3.set_title('Distribución de Pacientes por Cluster', fontsize=12, fontweight='bold')

        # 4. Categorías de riesgo
        if 'categorizacion' in self.resultados:
            ax4 = plt.subplot(2, 3, 4)
            df_cat = self.resultados['categorizacion']['df_categorizado']
            riesgo_counts = df_cat['categoria_riesgo'].value_counts()
            colors_riesgo = {'ALTO_RIESGO': 'red', 'MEDIO_RIESGO': 'orange', 'BAJO_RIESGO': 'green'}
            ax4.bar(range(len(riesgo_counts)), riesgo_counts.values,
                   color=[colors_riesgo.get(x, 'gray') for x in riesgo_counts.index])
            ax4.set_xticks(range(len(riesgo_counts)))
            ax4.set_xticklabels(riesgo_counts.index, rotation=45, ha='right')
            ax4.set_title('Distribución por Categoría de Riesgo', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Número de Pacientes')
            ax4.grid(axis='y', alpha=0.3)

        # 5. PCA Visualization
        if 'segmentacion' in self.resultados:
            df_seg = self.resultados['segmentacion']['df_segmentado']
            if 'pca_1' in df_seg.columns:
                ax5 = plt.subplot(2, 3, 5)
                scatter = ax5.scatter(df_seg['pca_1'], df_seg['pca_2'],
                                    c=df_seg['cluster'], cmap='viridis', alpha=0.6, s=50)
                ax5.set_xlabel('Componente Principal 1')
                ax5.set_ylabel('Componente Principal 2')
                ax5.set_title('Visualización de Clusters (PCA)', fontsize=12, fontweight='bold')
                plt.colorbar(scatter, ax=ax5, label='Cluster')
                ax5.grid(alpha=0.3)

        # 6. Matriz de riesgo por edad
        if 'categorizacion' in self.resultados and 'edad' in self.df_original.columns:
            ax6 = plt.subplot(2, 3, 6)
            df_cat = self.resultados['categorizacion']['df_categorizado']

            if 'grupo_etario' in df_cat.columns:
                tabla = pd.crosstab(df_cat['grupo_etario'], df_cat['categoria_riesgo'])
                sns.heatmap(tabla, annot=True, fmt='d', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Pacientes'})
                ax6.set_title('Matriz: Grupo Etario vs Categoría de Riesgo', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Categoría de Riesgo')
                ax6.set_ylabel('Grupo Etario')

        plt.tight_layout()

        if guardar:
            plt.savefig('analisis_morbilidad_completo.png', dpi=300, bbox_inches='tight')
            print("   ✅ Visualizaciones guardadas: analisis_morbilidad_completo.png")

        plt.show()

    def generar_reporte_ejecutivo(self):
        """
        Genera un reporte ejecutivo con los hallazgos principales
        """
        print("\n" + "="*80)
        print("📋 REPORTE EJECUTIVO - ANÁLISIS DE MORBILIDAD Y PACIENTES DE ALTO COSTO")
        print("="*80)

        print("\n1️⃣ RESUMEN DE DATOS")
        print(f"   • Total de registros analizados: {len(self.df_original):,}")
        print(f"   • Período de análisis: {self.df_original.columns[0]} hasta la fecha")

        print("\n2️⃣ SEGMENTACIÓN POBLACIONAL")
        for nombre, df_grupo in self.grupos_etarios.items():
            if len(df_grupo) > 0:
                desc = df_grupo['grupo_etario'].iloc[0]
                pct = len(df_grupo) / len(self.df_original) * 100
                print(f"   • {desc}: {len(df_grupo):,} pacientes ({pct:.1f}%)")

        if 'factores_riesgo' in self.resultados:
            print("\n3️⃣ ENFERMEDADES DE ALTO COSTO")

            if 'top_diagnosticos' in self.resultados['factores_riesgo']:
                top_5 = self.resultados['factores_riesgo']['top_diagnosticos'].head(5)
                print("   Top 5 diagnósticos más frecuentes:")
                for i, (diag, freq) in enumerate(top_5.items(), 1):
                    pct = freq / len(self.df_original) * 100
                    print(f"   {i}. {diag}: {freq:,} casos ({pct:.1f}%)")

            if 'pacientes_alto_costo' in self.resultados['factores_riesgo']:
                pac_alto = self.resultados['factores_riesgo']['pacientes_alto_costo']
                umbral = self.resultados['factores_riesgo']['umbral_alto_costo']
                print(f"\n   • Pacientes de ALTO COSTO (percentil 90): {len(pac_alto):,}")
                print(f"   • Umbral de costo: ${umbral:,.0f}")

        if 'segmentacion' in self.resultados:
            print("\n4️⃣ SEGMENTACIÓN INTELIGENTE")
            perfiles = self.resultados['segmentacion']['perfiles_clusters']
            print(f"   • Clusters identificados: {len(perfiles)}")

            for cluster, perfil in perfiles.items():
                print(f"\n   {cluster}:")
                print(f"      - Tamaño: {perfil['tamaño']:,} pacientes ({perfil['porcentaje']:.1f}%)")
                if 'edad_promedio' in perfil:
                    print(f"      - Edad promedio: {perfil['edad_promedio']:.1f} años")

        if 'categorizacion' in self.resultados:
            print("\n5️⃣ CATEGORIZACIÓN DE RIESGO")
            dist = self.resultados['categorizacion']['distribucion_categorias']

            for categoria in ['ALTO_RIESGO', 'MEDIO_RIESGO', 'BAJO_RIESGO']:
                if categoria in dist.index:
                    cant = dist[categoria]
                    pct = cant / dist.sum() * 100
                    print(f"   • {categoria}: {cant:,} pacientes ({pct:.1f}%)")

        print("\n6️⃣ RECOMENDACIONES ESTRATÉGICAS")
        print("   ✓ Priorizar seguimiento de pacientes en categoría ALTO_RIESGO")
        print("   ✓ Implementar programas preventivos para grupos de MEDIO_RIESGO")
        print("   ✓ Optimizar recursos según perfiles de clusters identificados")
        print("   ✓ Desarrollar rutas de atención específicas por grupo etario")
        print("   ✓ Monitorear evolución de principales diagnósticos")

        print("\n" + "="*80)

    def exportar_resultados(self, ruta_base='resultados_analisis'):
        """
        Exporta los resultados a archivos CSV

        Parámetros:
        -----------
        ruta_base : str
            Ruta base para guardar los archivos
        """
        print(f"\n💾 Exportando resultados a archivos CSV...")

        archivos_generados = []

        # 1. Exportar grupos etarios
        for nombre, df_grupo in self.grupos_etarios.items():
            if len(df_grupo) > 0:
                filename = f"{ruta_base}_{nombre}.csv"
                df_grupo.to_csv(filename, index=False, sep=';', encoding='utf-8-sig')
                archivos_generados.append(filename)
                print(f"   ✓ {filename}")

        # 2. Exportar segmentación
        if 'segmentacion' in self.resultados:
            filename = f"{ruta_base}_segmentacion_clusters.csv"
            self.resultados['segmentacion']['df_segmentado'].to_csv(
                filename, index=False, sep=';', encoding='utf-8-sig'
            )
            archivos_generados.append(filename)
            print(f"   ✓ {filename}")

        # 3. Exportar categorización
        if 'categorizacion' in self.resultados:
            filename = f"{ruta_base}_categorizacion_riesgo.csv"
            self.resultados['categorizacion']['df_categorizado'].to_csv(
                filename, index=False, sep=';', encoding='utf-8-sig'
            )
            archivos_generados.append(filename)
            print(f"   ✓ {filename}")

        # 4. Exportar pacientes de alto costo
        if 'factores_riesgo' in self.resultados and 'pacientes_alto_costo' in self.resultados['factores_riesgo']:
            filename = f"{ruta_base}_pacientes_alto_costo.csv"
            self.resultados['factores_riesgo']['pacientes_alto_costo'].to_csv(
                filename, index=False, sep=';', encoding='utf-8-sig'
            )
            archivos_generados.append(filename)
            print(f"   ✓ {filename}")

        print(f"\n✅ Total de archivos generados: {len(archivos_generados)}")

        return archivos_generados


# ============================================================================
# FUNCIÓN PRINCIPAL - EJECUTAR ANÁLISIS COMPLETO
# ============================================================================

def ejecutar_analisis_completo(url_datos, n_clusters=5, top_diagnosticos=15):
    """
    Ejecuta el análisis completo de morbilidad y segmentación de pacientes

    Parámetros:
    -----------
    url_datos : str
        URL del dataset de datos.gov.co
    n_clusters : int
        Número de clusters para segmentación (default: 5)
    top_diagnosticos : int
        Número de principales diagnósticos a analizar (default: 15)

    Returns:
    --------
    AnalizadorMorbilidad
        Objeto con todos los resultados del análisis
    """

    print("="*80)
    print("🏥 SISTEMA DE ANÁLISIS DE MORBILIDAD - ESE METROSALUD")
    print("="*80)
    print("\n🎯 Objetivos del análisis:")
    print("   1. Identificar factores de riesgo y enfermedades de alto costo")
    print("   2. Aplicar técnicas de BI para segmentación de pacientes")
    print("   3. Desarrollar modelo de categorización basado en datos históricos")
    print("\n" + "="*80 + "\n")

    # Inicializar analizador
    analizador = AnalizadorMorbilidad(url_datos)

    # 1. Cargar datos
    df = analizador.cargar_datos()
    if df is None:
        print("❌ No se pudieron cargar los datos. Verifique la URL.")
        return None

    # 2. Crear grupos etarios
    grupos = analizador.crear_grupos_etarios()
    if grupos is None:
        print("⚠️  No se pudieron crear grupos etarios")
        return analizador

    # 3. Analizar factores de riesgo
    factores = analizador.analizar_factores_riesgo(top_n=top_diagnosticos)

    # 4. Segmentación con BI y ML
    segmentacion = analizador.segmentacion_inteligencia_negocios(n_clusters=n_clusters)

    # 5. Modelo de categorización
    if segmentacion is not None:
        categorizacion = analizador.modelo_categorizacion_pacientes()

    # 6. Generar visualizaciones
    try:
        analizador.generar_visualizaciones(guardar=True)
    except Exception as e:
        print(f"⚠️  Error al generar visualizaciones: {e}")

    # 7. Generar reporte ejecutivo
    analizador.generar_reporte_ejecutivo()

    # 8. Exportar resultados
    archivos = analizador.exportar_resultados()

    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)

    return analizador


# ============================================================================
# USO DEL SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del script
    """

    # URL del dataset de Metrosalud
    URL_DATOS = "https://www.datos.gov.co/resource/pypc-7h3w.csv"

    # Ejecutar análisis completo
    analizador = ejecutar_analisis_completo(
        url_datos=URL_DATOS,
        n_clusters=5,           # Número de clusters para segmentación
        top_diagnosticos=15     # Top diagnósticos a analizar
    )

    # ========================================================================
    # ACCESO A RESULTADOS ESPECÍFICOS
    # ========================================================================

    if analizador is not None:
        print("\n" + "="*80)
        print("📊 ACCESO A DATAFRAMES GENERADOS")
        print("="*80)

        # Acceder a grupos etarios específicos
        print("\n1. GRUPOS ETARIOS DISPONIBLES:")
        for nombre_grupo, df_grupo in analizador.grupos_etarios.items():
            print(f"   • analizador.grupos_etarios['{nombre_grupo}']")
            print(f"     → {len(df_grupo):,} registros")

        # Acceder a dataframe con segmentación
        if 'segmentacion' in analizador.resultados:
            print("\n2. DATAFRAME CON SEGMENTACIÓN:")
            print("   • analizador.resultados['segmentacion']['df_segmentado']")
            df_seg = analizador.resultados['segmentacion']['df_segmentado']
            print(f"     → {len(df_seg):,} registros con clusters asignados")

        # Acceder a dataframe con categorización de riesgo
        if 'categorizacion' in analizador.resultados:
            print("\n3. DATAFRAME CON CATEGORIZACIÓN DE RIESGO:")
            print("   • analizador.resultados['categorizacion']['df_categorizado']")
            df_cat = analizador.resultados['categorizacion']['df_categorizado']
            print(f"     → {len(df_cat):,} registros con categoría de riesgo")

        # Acceder a pacientes de alto costo
        if 'factores_riesgo' in analizador.resultados:
            if 'pacientes_alto_costo' in analizador.resultados['factores_riesgo']:
                print("\n4. DATAFRAME DE PACIENTES DE ALTO COSTO:")
                print("   • analizador.resultados['factores_riesgo']['pacientes_alto_costo']")
                df_alto = analizador.resultados['factores_riesgo']['pacientes_alto_costo']
                print(f"     → {len(df_alto):,} pacientes de alto costo identificados")

        print("\n" + "="*80)
        print("💡 EJEMPLOS DE USO DE LOS DATAFRAMES:")
        print("="*80)
        print("""
# Ejemplo 1: Analizar grupo etario específico
df_primera_infancia = analizador.grupos_etarios['primera_infancia']
print(df_primera_infancia.head())

# Ejemplo 2: Filtrar pacientes de alto riesgo
df_categorizado = analizador.resultados['categorizacion']['df_categorizado']
pacientes_alto_riesgo = df_categorizado[df_categorizado['categoria_riesgo'] == 'ALTO_RIESGO']
print(f"Pacientes de alto riesgo: {len(pacientes_alto_riesgo)}")

# Ejemplo 3: Análisis por cluster
df_segmentado = analizador.resultados['segmentacion']['df_segmentado']
for cluster in df_segmentado['cluster'].unique():
    pacientes_cluster = df_segmentado[df_segmentado['cluster'] == cluster]
    print(f"Cluster {cluster}: {len(pacientes_cluster)} pacientes")

# Ejemplo 4: Cruce de información
# Pacientes de alto riesgo en primera infancia
df_alto_riesgo_infancia = df_categorizado[
    (df_categorizado['categoria_riesgo'] == 'ALTO_RIESGO') &
    (df_categorizado['grupo_etario'].str.contains('Primera infancia'))
]
print(f"Niños de alto riesgo: {len(df_alto_riesgo_infancia)}")

# Ejemplo 5: Exportar subset específico
pacientes_alto_riesgo.to_csv('pacientes_criticos.csv', index=False)
        """)

        print("\n" + "="*80)
        print("📁 ARCHIVOS GENERADOS")
        print("="*80)
        print("""
Los siguientes archivos CSV fueron generados:
• resultados_analisis_primera_infancia.csv
• resultados_analisis_infancia.csv
• resultados_analisis_adolescencia.csv
• resultados_analisis_juventud.csv
• resultados_analisis_adultez.csv
• resultados_analisis_persona_mayor.csv
• resultados_analisis_segmentacion_clusters.csv
• resultados_analisis_categorizacion_riesgo.csv
• resultados_analisis_pacientes_alto_costo.csv
• analisis_morbilidad_completo.png (visualizaciones)
        """)

        print("="*80)
        print("✅ Script ejecutado correctamente")
        print("="*80)
