import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # Usar el backend Agg que no requiere interfaz gráfica
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#Generar datasets
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


dataset_general_short = dataset_general.head(10) 


dataset_general = dataset_general

#Funciones para obtener las metricas

def get_mtc(dataset):
    
    columns = ['Linea A (Media)','Linea B (Media)','Linea A (mediana)','Linea B (mediana)','Linea A (Moda)','Linea B (Moda)']
    
    data = []

    data.append(dataset['Línea A (s)'].mean())
    data.append(dataset['Línea B (s)'].mean())
    data.append(dataset['Línea A (s)'].median())
    data.append(dataset['Línea B (s)'].median())
    data.append(dataset['Línea A (s)'].mode()[0])
    data.append(dataset['Línea B (s)'].mode()[0])


    data = np.array(data).reshape(1,-1) #Redimensionar para quedar en el mismo tamaño de las columnas
    
    dataset = pd.DataFrame(data=data, columns=columns)

    return dataset


dataset_mtc = get_mtc(dataset_general)

def get_dispersion_metrics(dataset):

    columns = ['Línea A (STD)','Línea B (STD)','Línea A (VAR)','Línea B (VAR)','Línea A (Range)','Línea B (Range)','Línea A (Cof_Var)','Línea B (Cof_Var)',]

    data = []

    data.append(dataset['Línea A (s)'].std())
    data.append(dataset['Línea B (s)'].std())
    data.append(dataset['Línea A (s)'].var())
    data.append(dataset['Línea B (s)'].var())
    data.append(dataset['Línea A (s)'].max() - dataset['Línea A (s)'].min())
    data.append(dataset['Línea B (s)'].max() - dataset['Línea B (s)'].min())
    data.append(dataset['Línea A (s)'].std() / dataset['Línea A (s)'].mean())
    data.append(dataset['Línea B (s)'].std() / dataset['Línea B (s)'].mean())

    data = np.array(data).reshape(1,-1)

    dataset = pd.DataFrame(columns=columns,data=data)

    return dataset


dataset_dispersion = get_dispersion_metrics(dataset_general)

def get_form_metrics(dataset):
    
    columns = ['Línea A (Asimetria)','Línea B (Asimetria)','Línea A (Curtosis)','Línea B (Curtosis)']

    data = []

    data.append(dataset['Línea A (s)'].skew())
    data.append(dataset['Línea B (s)'].skew())
    data.append(dataset['Línea A (s)'].kurtosis())
    data.append(dataset['Línea B (s)'].kurtosis())

    
    data = np.array(data).reshape(1,-1)

    dataset = pd.DataFrame(columns=columns,data=data)

    return dataset


dataset_form = get_form_metrics(dataset_general)

#Graficos

figHist, ax_hist = plt.subplots()

def create_hist():
    #Creamos el diagrama el historigrama
    ax_hist.hist([dataset_general['Línea A (s)'], dataset_general['Línea B (s)']], #El primer argumento es la lista de los 2
                bins=20, alpha=0.7, label=['Línea A', 'Línea B'],
                color=['tomato', 'orange']) #Aqui lo mismo

    # Etiquetas y texto
    ax_hist.set_xlabel('Tiempo (s)')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title('Histograma de Tiempos - Líneas A y B')
    ax_hist.legend()
    # Return the figure instead of showing it
    return figHist


histogram = create_hist()

# Save the histogram figure as an image file
histogram_path = './assets/histogram.png'
histogram.savefig(histogram_path)


figBox, ax_box_plot = plt.subplots()

def create_box_plots():
    #Creamos el boxplot
    ax_box_plot.boxplot([dataset_general['Línea A (s)'], dataset_general['Línea B (s)']],
                        vert=True,
                        tick_labels=['Linea A', 'Linea B'],  #Lista simple ,Usar en vez de labels tick_labels es mas nuevo
                        boxprops=dict(facecolor='gold'),
                        medianprops=dict(color='red'),
                        patch_artist=True)  

    ax_box_plot.set_title('Diagrama de cajas de las lineas')

    return figBox

boxPlot = create_box_plots()

# Save the box plot figure as an image file
box_plot_path = './assets/box_plot.png'
boxPlot.savefig(box_plot_path)

figStem, ax_stem = plt.subplots(figsize=(15, 8))


