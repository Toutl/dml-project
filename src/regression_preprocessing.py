"""
Módulo de Preprocesamiento para Predicción de Redshift
======================================================

Este módulo contiene todas las funciones de preprocesamiento para el dataset
de clasificación estelar SDSS17, enfocado en la predicción de redshift.

Autor: Isaac Vazquez Sandoval
Email: isaac.vazqsand@gmail.com
Año: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath='data/regression/redshift.csv'):
    """
    Carga el dataset de redshift desde un archivo CSV.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con los datos crudos
    """
    df = pd.read_csv(filepath)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def analyze_basic_stats(df):
    """
    Extrae y muestra estadísticas básicas del dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    dict
        Diccionario con estadísticas básicas
    """
    stats = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'unique_counts': df.nunique().to_dict(),
        'numeric_stats': df.describe().to_dict()
    }
    
    print("\n=== ESTADÍSTICAS BÁSICAS ===")
    print(f"Filas: {stats['shape'][0]}")
    print(f"Columnas: {stats['shape'][1]}")
    print(f"\nColumnas: {stats['columns']}")
    print(f"\nDatos nulos por columna:")
    for col, nulls in stats['null_counts'].items():
        if nulls > 0:
            print(f"  {col}: {nulls}")
    if sum(stats['null_counts'].values()) == 0:
        print("  No hay valores nulos")
    
    return stats


def analyze_skewness(df):
    """
    Calcula el sesgo (skewness) de las columnas numéricas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    pd.Series
        Serie con el sesgo de cada columna numérica
    """
    skewness = df.select_dtypes(include=[np.number]).skew().round(3)
    
    print("\n=== ANÁLISIS DE SESGO ===")
    print(skewness.sort_values(ascending=False))
    print("\nInterpretación:")
    print("  |sesgo| < 0.5: Distribución simétrica")
    print("  0.5 < |sesgo| < 1: Sesgo moderado")
    print("  |sesgo| > 1: Sesgo alto")
    
    return skewness


def detect_invalid_values(df, filters=['u', 'g', 'z']):
    """
    Detecta valores inválidos (-9999) en los filtros fotométricos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a analizar
    filters : list
        Lista de columnas de filtros a verificar
        
    Returns:
    --------
    dict
        Diccionario con conteos de valores inválidos por filtro
    """
    invalid_counts = {}
    
    print("\n=== DETECCIÓN DE VALORES INVÁLIDOS ===")
    for filter_name in filters:
        if filter_name in df.columns:
            count = (df[filter_name] == -9999).sum()
            invalid_counts[filter_name] = count
            print(f"Filtro {filter_name}: {count} valores = -9999")
    
    return invalid_counts


def analyze_redshift(df):
    """
    Analiza la distribución de la variable objetivo (redshift).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con la columna 'redshift'
        
    Returns:
    --------
    dict
        Diccionario con estadísticas de redshift
    """
    redshift_stats = {
        'min': df['redshift'].min(),
        'max': df['redshift'].max(),
        'mean': df['redshift'].mean(),
        'median': df['redshift'].median(),
        'mode': df['redshift'].mode()[0],
        'std': df['redshift'].std(),
        'unique_count': df['redshift'].nunique(),
        'zero_count': (df['redshift'] == 0).sum(),
        'negative_count': (df['redshift'] < 0).sum(),
        'positive_count': (df['redshift'] > 0).sum()
    }
    
    print("\n=== ANÁLISIS DE REDSHIFT ===")
    print(f"Rango: [{redshift_stats['min']:.6f}, {redshift_stats['max']:.6f}]")
    print(f"Media: {redshift_stats['mean']:.6f}")
    print(f"Mediana: {redshift_stats['median']:.6f}")
    print(f"Moda: {redshift_stats['mode']:.6f}")
    print(f"Desviación estándar: {redshift_stats['std']:.6f}")
    print(f"\nDistribución:")
    print(f"  Redshift = 0: {redshift_stats['zero_count']} ({redshift_stats['zero_count']/len(df)*100:.2f}%)")
    print(f"  Redshift < 0: {redshift_stats['negative_count']} ({redshift_stats['negative_count']/len(df)*100:.2f}%)")
    print(f"  Redshift > 0: {redshift_stats['positive_count']} ({redshift_stats['positive_count']/len(df)*100:.2f}%)")
    
    return redshift_stats


def analyze_stars_redshift(df):
    """
    Analiza específicamente el redshift de las estrellas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columnas 'class' y 'redshift'
        
    Returns:
    --------
    dict
        Estadísticas de redshift para estrellas
    """
    stars = df[df['class'] == 'STAR']
    stars_above_0 = stars[stars['redshift'] > 0]
    
    stars_stats = {
        'total': len(stars),
        'above_zero': len(stars_above_0),
        'at_zero': len(stars[stars['redshift'] == 0]),
        'below_zero': len(stars[stars['redshift'] < 0]),
        'min_above_zero': stars_above_0['redshift'].min() if len(stars_above_0) > 0 else None,
        'max_above_zero': stars_above_0['redshift'].max() if len(stars_above_0) > 0 else None,
        'mean_above_zero': stars_above_0['redshift'].mean() if len(stars_above_0) > 0 else None,
        'median_above_zero': stars_above_0['redshift'].median() if len(stars_above_0) > 0 else None
    }
    
    print("\n=== ANÁLISIS DE ESTRELLAS ===")
    print(f"Total de estrellas: {stars_stats['total']}")
    print(f"Estrellas con redshift > 0: {stars_stats['above_zero']} ({stars_stats['above_zero']/stars_stats['total']*100:.2f}%)")
    print(f"Estrellas con redshift = 0: {stars_stats['at_zero']}")
    print(f"Estrellas con redshift < 0: {stars_stats['below_zero']}")
    
    if stars_stats['above_zero'] > 0:
        print(f"\nEstadísticas para estrellas con redshift > 0:")
        print(f"  Mínimo: {stars_stats['min_above_zero']:.6f}")
        print(f"  Máximo: {stars_stats['max_above_zero']:.6f}")
        print(f"  Media: {stars_stats['mean_above_zero']:.6f}")
        print(f"  Mediana: {stars_stats['median_above_zero']:.6f}")
    
    return stars_stats


def clean_invalid_filters(df, filters=['u', 'g', 'z'], invalid_value=-9999):
    """
    Elimina filas con valores inválidos en los filtros fotométricos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a limpiar
    filters : list
        Lista de columnas de filtros
    invalid_value : int/float
        Valor considerado inválido
        
    Returns:
    --------
    pd.DataFrame
        DataFrame limpio
    """
    df_clean = df.copy()
    rows_before = len(df_clean)
    
    # Crear máscara para filtrar
    mask = pd.Series([True] * len(df_clean))
    for filter_name in filters:
        if filter_name in df_clean.columns:
            mask &= (df_clean[filter_name] != invalid_value)
    
    df_clean = df_clean[mask]
    rows_removed = rows_before - len(df_clean)
    
    print("\n=== LIMPIEZA DE VALORES INVÁLIDOS ===")
    print(f"Filas eliminadas: {rows_removed}")
    print(f"Filas restantes: {len(df_clean)}")
    
    return df_clean


def drop_irrelevant_columns(df):
    """
    Elimina columnas que no son relevantes para la predicción de redshift.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame a limpiar
        
    Returns:
    --------
    pd.DataFrame
        DataFrame sin columnas irrelevantes
    """
    columns_to_drop = [
        'obj_ID',      # ID único del objeto
        'spec_obj_ID', # ID espectroscópico
        'run_ID',      # ID de corrida
        'rerun_ID',    # ID de reprocesamiento (sin varianza)
        'cam_col',     # Columna de cámara (técnico)
        'field_ID',    # ID de campo
        'plate',       # ID de placa
        'MJD',         # Fecha juliana modificada
        'fiber_ID'     # ID de fibra
    ]
    
    df_clean = df.copy()
    columns_dropped = []
    
    for col in columns_to_drop:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            columns_dropped.append(col)
    
    print("\n=== ELIMINACIÓN DE COLUMNAS IRRELEVANTES ===")
    print(f"Columnas eliminadas ({len(columns_dropped)}):")
    for col in columns_dropped:
        print(f"  - {col}")
    print(f"\nColumnas restantes ({len(df_clean.columns)}):")
    print(f"  {list(df_clean.columns)}")
    
    return df_clean


def encode_class_variable(df):
    """
    Codifica la variable categórica 'class' a valores numéricos.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con la columna 'class'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con 'class' codificada
    dict
        Diccionario de mapeo usado
    """
    df_clean = df.copy()
    
    # GALAXY=0, STAR=1, QSO=2
    class_mapping = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
    
    print("\n=== ENCODING DE VARIABLE CATEGÓRICA ===")
    print("Valores antes de codificar:")
    print(df_clean['class'].value_counts())
    
    df_clean['class'] = df_clean['class'].map(class_mapping)
    
    print("\nMapeo aplicado:")
    for key, value in class_mapping.items():
        print(f"  {key} -> {value}")
    
    print("\nValores después de codificar:")
    print(df_clean['class'].value_counts().sort_index())
    
    return df_clean, class_mapping


def compute_correlation_matrix(df):
    """
    Calcula la matriz de correlación para variables numéricas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con variables numéricas
        
    Returns:
    --------
    pd.DataFrame
        Matriz de correlación
    """
    correlation = df.corr()
    
    print("\n=== MATRIZ DE CORRELACIÓN ===")
    print("\nCorrelación con redshift (variable objetivo):")
    if 'redshift' in correlation.columns:
        redshift_corr = correlation['redshift'].sort_values(ascending=False)
        print(redshift_corr)
    
    return correlation


def split_data(df, target_column='redshift', test_size=0.2, random_state=42):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    target_column : str
        Nombre de la columna objetivo
    test_size : float
        Proporción de datos para prueba
    random_state : int
        Semilla para reproducibilidad
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print("\n=== DIVISIÓN DE DATOS ===")
    print(f"Total de muestras: {len(df)}")
    print(f"\nDatos de entrenamiento:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"\nDatos de prueba:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    print(f"\nPorcentaje de entrenamiento: {len(X_train)/len(X)*100:.2f}%")
    print(f"Porcentaje de prueba: {len(X_test)/len(X)*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Escala las características usando StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Datos de entrenamiento
    X_test : pd.DataFrame
        Datos de prueba
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de nuevo a DataFrame para mantener nombres de columnas
    X_train_scaled = pd.DataFrame(
        X_train_scaled, 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled, 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    print("\n=== ESCALADO DE CARACTERÍSTICAS ===")
    print("StandardScaler aplicado")
    print("\nEstadísticas después del escalado (entrenamiento):")
    print(X_train_scaled.describe().loc[['mean', 'std']].round(4))
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath='data/regression/redshift.csv', 
                        test_size=0.2, 
                        random_state=42,
                        verbose=True):
    """
    Pipeline completo de preprocesamiento.
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV
    test_size : float
        Proporción de datos para prueba
    random_state : int
        Semilla para reproducibilidad
    verbose : bool
        Si True, imprime información detallada
        
    Returns:
    --------
    dict
        Diccionario con todos los datos procesados:
        - 'X_train_scaled': Características de entrenamiento escaladas
        - 'X_test_scaled': Características de prueba escaladas
        - 'y_train': Variable objetivo de entrenamiento
        - 'y_test': Variable objetivo de prueba
        - 'scaler': Objeto StandardScaler ajustado
        - 'class_mapping': Mapeo de clases
        - 'feature_names': Nombres de las características
    """
    if not verbose:
        import warnings
        warnings.filterwarnings('ignore')
    
    print("="*60)
    print("PIPELINE DE PREPROCESAMIENTO - PREDICCIÓN DE REDSHIFT")
    print("="*60)
    
    # 1. Cargar datos
    df = load_data(filepath)
    
    # 2. Análisis exploratorio
    if verbose:
        analyze_basic_stats(df)
        analyze_skewness(df)
        detect_invalid_values(df)
        analyze_redshift(df)
        analyze_stars_redshift(df)
    
    # 3. Limpieza
    df_clean = clean_invalid_filters(df)
    df_clean = drop_irrelevant_columns(df_clean)
    df_clean, class_mapping = encode_class_variable(df_clean)
    
    # 4. Análisis de correlación
    if verbose:
        compute_correlation_matrix(df_clean)
    
    # 5. División de datos
    X_train, X_test, y_train, y_test = split_data(
        df_clean, 
        target_column='redshift',
        test_size=test_size,
        random_state=random_state
    )
    
    # 6. Escalado
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n" + "="*60)
    print("PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    print(f"\nDatos listos para modelado:")
    print(f"  - {len(X_train_scaled)} muestras de entrenamiento")
    print(f"  - {len(X_test_scaled)} muestras de prueba")
    print(f"  - {len(X_train_scaled.columns)} características")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'class_mapping': class_mapping,
        'feature_names': list(X_train.columns)
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Ejecutar pipeline completo
    data = preprocess_pipeline(
        filepath='data/regression/redshift.csv',
        test_size=0.2,
        random_state=42,
        verbose=True
    )
    
    # Acceder a los datos procesados
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("\n¡Listo para entrenar modelos!")
    print(f"Características disponibles: {data['feature_names']}")