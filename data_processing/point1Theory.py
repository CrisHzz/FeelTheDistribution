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

def probability_find():
    # Parámetros empíricos
    mu_A, sigma_A = dataset_general['Línea A (s)'].mean(), dataset_general['Línea A (s)'].std(ddof=1)
    mu_B, sigma_B = dataset_general['Línea B (s)'].mean(), dataset_general['Línea B (s)'].std(ddof=1)

    # c‑1) Probabilidad de exceder 120 s usando distribución normal
    p_A = 1 - stats.norm.cdf(120, loc=mu_A, scale=sigma_A)
    p_B = 1 - stats.norm.cdf(120, loc=mu_B, scale=sigma_B)

    # c‑2) Tiempo estándar para cumplir 90% usando distribución normal
    t90_A = stats.norm.ppf(0.90, loc=mu_A, scale=sigma_A)
    t90_B = stats.norm.ppf(0.90, loc=mu_B, scale=sigma_B)

    # Mostrar también qué distribución se identificó como mejor ajuste
    result_text = f"Línea A (distribución usada: Normal, mejor ajuste: {mejor_A}) →  P(T>120) = {p_A:.2%}   |   t_90 = {t90_A:.2f} s\n"
    result_text += f"Línea B (distribución usada: Normal, mejor ajuste: {mejor_B}) →  P(T>120) = {p_B:.2%}   |   t_90 = {t90_B:.2f} s"
    
    return result_text



probability_time_extra = probability_find()


def ic95_media(series):
    n = len(series)
    mu = series.mean()
    s = series.std(ddof=1)
    ic = stats.t.interval(0.95, df=n-1, loc=mu, scale=s/np.sqrt(n))
    return ic

def get_confidence_intervals():
    ic_A = ic95_media(dataset_general['Línea A (s)'])
    ic_B = ic95_media(dataset_general['Línea B (s)'])
    
    result = f"IC 95 % μ_A: {ic_A[0]:.2f} – {ic_A[1]:.2f}\n"
    result += f"IC 95 % μ_B: {ic_B[0]:.2f} – {ic_B[1]:.2f}"
    
    return result

confidence_intervals_95 = get_confidence_intervals()


