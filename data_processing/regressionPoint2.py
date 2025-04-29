import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

dataset_general = pd.read_excel('./media/dataset_punto2.xlsx')

def cof_correlation(dataset_general):
    correlation = np.corrcoef(dataset_general['velocidad_produccion'], dataset_general['consumo_energia'])
    text = f'El coeficiente de correlacion entre las 2 variables es de {np.round(correlation[0,1],3)},\npor lo tanto hay CORRELACION LINEAL ALTA.'
    return text

def recomendar_modelo_regresion(df, var_x, var_y):
    X = df[[var_x]].values
    y = df[var_y].values

    modelos = {
        'Regresión Lineal': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }

    resultados = {}

    texto_resultados = "\n--- Evaluación de modelos (R² con cross-validation) ---\n\n"
    for nombre, modelo in modelos.items():
        scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
        promedio = np.mean(scores)
        resultados[nombre] = promedio
        texto_resultados += f"{nombre}: R² promedio = {promedio:.4f}\n"

    mejor_modelo = max(resultados, key=resultados.get)
    texto_resultados += "\n--- Resultados ---\n"
    texto_resultados += f"\n✅ Modelo recomendado: {mejor_modelo}\n"
    texto_resultados += f"Valores de R² promedio:\n{resultados}\n"

    return texto_resultados

best_model = recomendar_modelo_regresion(dataset_general,'velocidad_produccion','consumo_energia')

corre_variables = cof_correlation(dataset_general)

#Creando la regression lineal ,datos de entreno y testeo ,estandarizacion para un mejor rendimiento

modelo_estandarizador = StandardScaler()

dataset_estandarizado = modelo_estandarizador.fit_transform(dataset_general.drop(columns='jornada'))


dataset_estandarizado = pd.DataFrame(columns=['velocidad_produccion','consumo_energia'],data=dataset_estandarizado)

dataset_estandarizado_short= dataset_estandarizado.head(10)