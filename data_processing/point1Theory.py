import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Cargar datos
dataset_general = pd.read_csv('./assets/dataset_punto1.csv')

# Imprimir estadísticas de Línea B para verificar los datos

# -----------------------------------------------------------
def goodness_of_fit(series, nombre='Serie', alpha=0.05):
    """
    Aplica pruebas KS para Normal, Exponencial y Weibull.
    Devuelve (mejor, resultados_dict) donde:
      - mejor: nombre de la distribución con p más alta (o 'Ninguna...')
      - resultados_dict: {'Normal': p1, 'Exponencial': p2, 'Weibull': p3}
    """
    resultados = {}
    
    # Normal
    mu, sigma = series.mean(), series.std(ddof=1)
    ks_norm = stats.kstest(series, 'norm', args=(mu, sigma))
    resultados['Normal'] = ks_norm.pvalue
    
    # Exponencial (floc=0)
    loc_exp, scale_exp = stats.expon.fit(series, floc=0)
    ks_exp = stats.kstest(series, 'expon', args=(loc_exp, scale_exp))
    resultados['Exponencial'] = ks_exp.pvalue
    
    # Weibull mínima (sin fijar loc)
    c_w, loc_w, scale_w = stats.weibull_min.fit(series)  # Removido floc=0
    ks_weib = stats.kstest(series, 'weibull_min', args=(c_w, loc_w, scale_w))
    resultados['Weibull'] = ks_weib.pvalue
    
    # Elegir la mejor distribución
    mejor = max(resultados, key=resultados.get)
    if resultados[mejor] < alpha:
        mejor = 'Ninguna (todas p < α)'
    else:
        None
    
    return mejor, resultados

# Ejecutar para Línea A
mejor_A, dict_A = goodness_of_fit(dataset_general['Línea A (s)'], 'Línea A')

# Ejecutar para Línea B
mejor_B, dict_B = goodness_of_fit(dataset_general['Línea B (s)'], 'Línea B')

# Crear el DataFrame de pandas
df_pvalues = pd.DataFrame([dict_A, dict_B], index=['Línea A', 'Línea B'])
df_pvalues['Mejor ajuste'] = [mejor_A, mejor_B]
