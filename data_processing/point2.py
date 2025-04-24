import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_2 = pd.read_excel('./media/dataset_punto2.xlsx')


dataset_2_short = dataset_2.head(10)

def get_mtc(dataset):
    
    columns = ['velocidad_produccion (Media)','consumo_energia (Media)','velocidad_produccion (mediana)','consumo_energia (mediana)','velocidad_produccion (Moda)','consumo_energia (Moda)']
    
    data = []

    data.append(dataset['velocidad_produccion'].mean())
    data.append(dataset['consumo_energia'].mean())
    data.append(dataset['velocidad_produccion'].median())
    data.append(dataset['consumo_energia'].median())
    data.append(dataset['velocidad_produccion'].mode()[0])
    data.append(dataset['consumo_energia'].mode()[0])


    data = np.array(data).reshape(1,-1) #Redimensionar para quedar en el mismo tamaño de las columnas
    
    dataset = pd.DataFrame(data=data, columns=columns)

    return dataset

dataset_mtc2 = get_mtc(dataset_2)

def get_dispersion_metrics(dataset):

    columns = ['velocidad_produccion (STD)','consumo_energia (STD)','velocidad_produccion (VAR)','consumo_energia (VAR)','velocidad_produccion (Range)','consumo_energia (Range)','velocidad_produccion (Cof_Var)','consumo_energia (Cof_Var)',]

    data = []

    data.append(dataset['velocidad_produccion'].std())
    data.append(dataset['consumo_energia'].std())
    data.append(dataset['velocidad_produccion'].var())
    data.append(dataset['consumo_energia'].var())
    data.append(dataset['velocidad_produccion'].max() - dataset['velocidad_produccion'].min())
    data.append(dataset['consumo_energia'].max() - dataset['consumo_energia'].min())
    data.append(dataset['velocidad_produccion'].std() / dataset['velocidad_produccion'].mean())
    data.append(dataset['consumo_energia'].std() / dataset['consumo_energia'].mean())

    data = np.array(data).reshape(1,-1)

    dataset = pd.DataFrame(columns=columns,data=data)

    return dataset


dataset_dispersion2 = get_dispersion_metrics(dataset_2)

def get_form_metrics(dataset):
    
    columns = ['velocidad_produccion (Asimetria)','consumo_energia (Asimetria)','velocidad_produccion (Curtosis)','consumo_energia (Curtosis)']

    data = []

    data.append(dataset['velocidad_produccion'].skew())
    data.append(dataset['consumo_energia'].skew())
    data.append(dataset['velocidad_produccion'].kurtosis())
    data.append(dataset['consumo_energia'].kurtosis())

    
    data = np.array(data).reshape(1,-1)

    dataset = pd.DataFrame(columns=columns,data=data)

    return dataset


dataset_form2 = get_form_metrics(dataset_2)

fig_hist2, ax_hist = plt.subplots()


def create_hist():
    #Creamos el diagrama el historigrama
    ax_hist.hist([dataset_2['velocidad_produccion'], dataset_2['consumo_energia']], #El primer argumento es la lista de los 2
                bins=20, alpha=0.7, label=['Velocidad de produccion', 'consumo de energia'],
                color=['khaki', 'yellowgreen']) #Aqui lo mismo

    # Etiquetas y texto
    ax_hist.set_xlabel('Valores')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title('Histograma de Velocidad de produccion y Consumo de energia')
    ax_hist.legend()

    return fig_hist2

histogram = create_hist()


histogram_path = './assets/histogram2.png'

histogram.savefig(histogram_path)

fig_scatter, ax_scatter = plt.subplots()


def create_scatter(dataset):
    #Creamos la instancia

    #Se crea el grafico de dispersion
    ax_scatter.scatter(dataset['velocidad_produccion'],dataset['consumo_energia'],
                    alpha=0.7, 
                    label='Velocidad de produccion vs consumo de energia',s=100,color='yellowgreen')

    ax_scatter.set_xlabel('Velocidad de produccion')
    ax_scatter.set_ylabel("Consumo de energia")
    
    ax_scatter.set_title("Grafico de dispersion")
    ax_scatter.legend()
    return fig_scatter

scatter =  create_scatter(dataset_2)
scatter_path = './assets/scatter.png'
scatter.savefig(scatter_path)






    