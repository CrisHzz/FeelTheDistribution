import pandas as pd
import numpy as np
import matplotlib as plt

def generate_dataset1():
    np.random.seed(42)

    media_a = 120  
    desviacion_a = 10  

    media_b = 110  
    desviacion_b = 12  

    n_observaciones = 100

    tiempos_a = np.random.normal(loc=media_a, scale=desviacion_a, size=n_observaciones)
    tiempos_b = np.random.normal(loc=media_b, scale=desviacion_b, size=n_observaciones)

    df = pd.DataFrame({
        'Línea A (s)': np.round(tiempos_a, 2),
        'Línea B (s)': np.round(tiempos_b, 2)
    })

    df.to_csv('./assets/dataset_punto1.csv', index=False)



generate_dataset1()

dataset_general = pd.read_csv('./assets/dataset_punto1.csv')

dataset_general = dataset_general.head(10)

