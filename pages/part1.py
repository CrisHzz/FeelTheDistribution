import reflex as rx
from rxconfig import config
from data_processing.point1 import dataset_general, dataset_general_short , dataset_mtc , dataset_dispersion, dataset_form
from data_processing.point1Theory import df_pvalues , probability_time_extra, confidence_intervals_95, hypothesis_test_results

class State(rx.State):
    """The app state."""
    ...

def part1() -> rx.Component:
    return rx.container(
        rx.color_mode.button(
            position="top-right", 
            class_name="bg-white text-black hover:bg-gray-200 transition-colors px-4 py-2 rounded-lg border-6 border-white-400 shadow-md"
        ),
        rx.vstack(
            rx.box(
                rx.hstack(
                    rx.heading(
                        "FeelTheDistribution",
                        size="8",
                        class_name="text-white font-bold pl-4",
                    ),
                    rx.hstack(
                        rx.link(
                            "Part 1", 
                            href="/part1", 
                            class_name="text-white hover:text-purple-300 transition-colors px-4 py-2 rounded-lg hover:bg-purple-900/30"
                        ),
                        rx.link(
                            "Part 2", 
                            href="/part2", 
                            class_name="text-white hover:text-purple-300 transition-colors px-4 py-2 rounded-lg hover:bg-purple-900/30"
                        ),
                        rx.link(
                            "Part 3", 
                            href="/part3", 
                            class_name="text-white hover:text-purple-300 transition-colors px-4 py-2 rounded-lg hover:bg-purple-900/30"
                        ),
                        rx.link(
                            "Bibliografias", 
                            href="/bibliografias", 
                            class_name="text-white hover:text-purple-300 transition-colors px-4 py-2 rounded-lg hover:bg-purple-900/30"
                        ),
                        
                        spacing="8",
                    ),
                    width="100%",
                    justify="between",
                    align="center",
                    padding="6",
                    class_name="bg-gradient-to-r from-purple-900/50 to-black/50 backdrop-blur-sm rounded-2xl shadow-2xl border border-purple-500/20",
                ),
                class_name="w-full mb-8",
            ),
            rx.box(
                rx.vstack(
                    # Dataset Parte #1
                    rx.heading(
                        "Dataset Parte #1",
                        size="7",
                        class_name="text-white mb-4",
                    ),
                    rx.heading(
                        "Creación del conjunto de datos",
                        size="5",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Se crea un dataset usando las librerías de numpy y pandas, estos nos generarán los datasets para este apartado. La columna Línea A (s) representa el tiempo que demora completar una operación en esta línea de operación para un producto, lo mismo con la columna Línea B (s). Siendo un total de 100 registros por cada línea de producción.",
                        class_name="text-white mb-8",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in dataset_general.columns
                                ]
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    *[
                                        rx.table.cell(
                                            str(value),
                                            class_name="text-white"
                                        )
                                        for value in row
                                    ]
                                )
                                for row in dataset_general_short.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black mb-8 w-full",
                    ),
                    
                    # Medidas de tendencia central, dispersión y forma
                    rx.heading(
                        "Medidas de tendencia central, dispersión y forma",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Existen varios tipos de medidas que nos sirven para darle sentido a valor a los datos sea desde el apartado basico como entender el promedio, como se desvian sus datos del centro y hasta que forma y comportamiento tienen ellos",
                        class_name="text-white mb-4",
                    ),
                    rx.heading(
                        "Medidas de tendencia central",
                        size="5",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Las medidas de tendencia central nos ayudan a entender el valor central o típico de nuestros datos. La media nos da el promedio, la mediana el valor central y la moda el valor más frecuente.",
                        class_name="text-white mb-4",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in dataset_mtc.columns
                                ]
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    *[
                                        rx.table.cell(
                                            str(value),
                                            class_name="text-white"
                                        )
                                        for value in row
                                    ]
                                )
                                for row in dataset_mtc.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black mb-8 w-full",
                    ),
                    rx.heading(
                        "Medidas de dispersión",
                        size="5",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Las medidas de dispersión nos indican qué tan dispersos están los datos alrededor de su valor central. La desviación estándar, varianza y rango nos ayudan a entender la variabilidad de los datos.",
                        class_name="text-white mb-4",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in dataset_dispersion.columns
                                ]
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    *[
                                        rx.table.cell(
                                            str(value),
                                            class_name="text-white"
                                        )
                                        for value in row
                                    ]
                                )
                                for row in dataset_dispersion.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black mb-8 w-full",
                    ),
                    rx.heading(
                        "Medidas de forma",
                        size="5",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Las medidas de forma nos ayudan a entender la distribución de los datos. La asimetría nos indica si la distribución está sesgada hacia la izquierda o derecha, mientras que la curtosis nos dice qué tan puntiaguda o plana es la distribución.",
                        class_name="text-white mb-4",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in dataset_form.columns
                                ]
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    *[
                                        rx.table.cell(
                                            str(value),
                                            class_name="text-white"
                                        )
                                        for value in row
                                    ]
                                )
                                for row in dataset_form.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black mb-8 w-full",
                    ),
                    
                    # Gráficos
                    rx.heading(
                        "Graficos: Histograma, boxplot y tallos de hojas",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Entender como se ven nuestros datos graficamente es parte fundamental en un analisis estadistico, aqui encontraremos el significado de las figuras y colores que le dan sentido a la estadistica",
                        class_name="text-white mb-4",
                    ),
                    rx.heading(
                        "Grafico de histograma",
                        size="6",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Un histograma es una representación gráfica de la distribución de un conjunto de datos. Se utiliza para mostrar la frecuencia de los datos en intervalos específicos, lo que permite visualizar la forma de la distribución.",
                        class_name="text-white mb-4",
                    ),
                    rx.image(
                        src='/histogram.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 w-full",
                    ),
                    rx.heading(
                        "Grafico de boxplot",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "Un boxplot es una representación gráfica que muestra la distribución de un conjunto de datos a través de sus cuartiles. Permite identificar la mediana, los cuartiles y valores de tipo outlier.",
                        class_name="text-white mb-4",
                    ),
                    rx.image(
                        src='/box_plot.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 w-full",
                    ),
                    rx.heading(
                        "Grafico de tallos y hojas",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El gráfico de tallos y hojas es un tipo de gráfico que permite ver la distribución de los datos manteniendo su orden original y su valor exacto, lo que lo hace especialmente útil para análisis exploratorios. A diferencia de otros gráficos como los histogramas, este conserva los valores individuales, permitiendo una inspección más precisa. Además, facilita la identificación de valores atípicos (outliers), la detección de la moda, y proporciona una visión clara de la simetría o asimetría en la distribución.",
                        class_name="text-white mb-4"
                    ),
                    rx.image(
                        src='/stem_comparison_tables.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 w-full",
                    ),
                    
                    # Identificar la distribución
                    rx.heading(
                        "Indentificar la distribucion indicada",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Para identificar la distribución de los datos, se aplican pruebas de bondad de ajuste. Estas pruebas comparan la distribución observada con varias distribuciones teóricas para determinar cuál se ajusta mejor a los datos. En este caso, se aplicaron las pruebas de Kolmogorov-Smirnov (KS) para las distribuciones Normal, Exponencial y Weibull.",
                        class_name="text-white mb-4",
                    ),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in df_pvalues.columns
                                ]
                            )
                        ),
                        rx.table.body(
                            *[
                                rx.table.row(
                                    *[
                                        rx.table.cell(
                                            str(value),
                                            class_name="text-white"
                                        )
                                        for value in row
                                    ]
                                )
                                for row in df_pvalues.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black mb-8 w-full",
                    ),
                    rx.text(
                        "En la tabla anterior, se presentan los resultados de las pruebas de bondad de ajuste para las distribuciones Normal, Exponencial y Weibull. La columna 'Mejor ajuste' indica cuál distribución se ajusta mejor a los datos de cada línea de producción. Si el valor p es menor que el nivel de significancia (α = 0.05), se rechaza la hipótesis nula de que los datos siguen esa distribución.",
                        class_name="text-white mb-4 font-bold"
                    ),
                    rx.image(
                        src='/Línea_A_dist_fit.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 w-full",
                    ),
                    rx.text(
                        "Se muestra el gráfico de la Línea A con la distribución Normal superpuesta. La línea azul representa la distribución Normal ajustada a los datos de la Línea A. Se observa que la distribución Normal se ajusta bastante bien a los datos.",
                        class_name="text-white mb-4"
                    ),
                    rx.image(
                        src='/Línea_B_dist_fit.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 w-full",
                    ),
                    rx.text(
                        "En el gráfico de la Línea B con la distribución Normal y weibull superpuesta. La línea azul representa la distribución Normal ajustada a los datos de la Línea B.",
                        class_name="text-white mb-4"
                    ),
                    rx.heading(
                        "Utilizando la distribucion indicada, encontrar la probabilidad de encontrar un tiempo extra",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Calcule la probabilidad de que un ciclo exceda el tiempo estándar establecido en 120 segundos Determine el tiempo máximo que debe establecerse como estándar para garantizar que el 90% de los ciclos se completen dentro de ese tiempo ",
                        size="5",
                        class_name="text-white mb-4",

                    ),
                    rx.code_block(
                        probability_time_extra,
                        language="python",
                        class_name="text-white mb-4",
                    ),
                    
                    rx.heading(
                        "Interpretación de resultados",
                        size="6",
                        class_name="text-white mb-4 mt-4",
                    ),
                    rx.text(
                        "Los resultados del análisis probabilístico muestran lo siguiente:",
                        class_name="text-white mb-2",
                    ),
                    rx.unordered_list(
                        rx.list_item(
                            rx.text(
                                rx.text("Línea A: ", as_="span", class_name="font-bold"), 
                                "La probabilidad de que un ciclo exceda el tiempo estándar de 120 segundos es del 45.45%. Esto significa que casi la mitad de los ciclos de producción en esta línea sobrepasarán el tiempo establecido, lo cual indica una eficiencia comprometida."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                rx.text("Línea A (tiempo estándar para 90%): ", as_="span", class_name="font-bold"), 
                                "Para garantizar que el 90% de los ciclos se completen dentro del tiempo estándar, este debería establecerse en 130.60 segundos, lo que representa un incremento de 10.6 segundos respecto al valor actual."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                rx.text("Línea B: ", as_="span", class_name="font-bold"), 
                                "La probabilidad de que un ciclo exceda los 120 segundos es considerablemente menor, solo del 19.75%. Esto muestra que la Línea B es más eficiente y consistente en términos de tiempos de producción."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                rx.text("Línea B (tiempo estándar para 90%): ", as_="span", class_name="font-bold"), 
                                "Para garantizar que el 90% de los ciclos se completen a tiempo en esta línea, el estándar debería establecerse en 124.93 segundos, un incremento mucho menor de solo 4.93 segundos."
                            )
                        ),
                        class_name="text-white mb-4 pl-6",
                    ),
                    rx.text(
                        "Con base en estos resultados, podemos concluir que la Línea B presenta un mejor desempeño operativo en términos de tiempos de ciclo. La menor variabilidad y el tiempo promedio más bajo en la Línea B sugieren que podría ser beneficioso analizar qué factores contribuyen a su mejor rendimiento para potencialmente implementarlos también en la Línea A y mejorar su eficiencia.",
                        class_name="text-white mb-4 font-bold",
                    ),
                    rx.heading(
                        "Intervalos de confianza al 95%",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Los intervalos de confianza al 95% para las medias de las líneas A y B son los siguientes:",
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        confidence_intervals_95,
                        language="python",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Esto significa que estamos un 95% seguros de que la media real del tiempo de producción para la Línea A está entre 117.16 y 120.76 segundos, mientras que para la Línea B está entre 108.00 y 112.54 segundos.",
                        class_name="text-white mb-4",
                    ),
                    
                    # Prueba de hipótesis
                    rx.heading(
                        "Prueba de hipótesis para comparar tiempos medios",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.text(
                        "Para determinar si existe una diferencia estadísticamente significativa entre los tiempos medios de ciclo de ambas líneas de producción, se realizó una prueba t de Student para muestras independientes (también conocida como prueba t de Welch, ya que las varianzas son diferentes).",
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        hypothesis_test_results,
                        language="python",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Esta prueba nos permite tomar decisiones basadas en evidencia estadística sobre si las diferencias observadas en los tiempos de ciclo son significativas o podrían deberse al azar. Los resultados tienen importantes implicaciones para la gestión del proceso, ya que nos ayudan a identificar si una línea de producción es significativamente más eficiente que la otra.",
                        class_name="text-white mb-4",
                    ),
                    
                    # Comparación con artículo científico
                    rx.heading(
                        "Comparación con artículo científico",
                        size="7",
                        class_name="text-white mb-4 mt-8",
                    ),
                    rx.image(
                        src='/article_image.png',  # Asegúrate de guardar tu imagen con este nombre en la carpeta /assets
                        height="300px",
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-6 object-contain",
                    ),
                    rx.text(
                        "Se realizó una comparación de las técnicas estadísticas utilizadas en este análisis con las del artículo 'Design the Abnormal Object Detection System Using Template Matching and Subtract Background Algorithm', obtenido a través de la base de datos Springer de la universidad.",
                        class_name="text-white mb-4",
                    ),
                    rx.heading(
                        "Similitudes metodológicas",
                        size="6",
                        class_name="text-white mb-2",
                    ),
                    rx.unordered_list(
                        rx.list_item(
                            rx.text(
                                "Uso de distribuciones estadísticas para modelar el comportamiento de los datos: En nuestro análisis aplicamos pruebas de bondad de ajuste para identificar la distribución que mejor se adapta a los datos (Normal, Exponencial, Weibull), mientras que en el artículo se utilizan distribuciones estadísticas para modelar patrones normales y anómalos en la detección de objetos."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                "Aplicación de pruebas de hipótesis para la toma de decisiones: Nuestro análisis utiliza la prueba t para determinar si hay diferencia significativa entre líneas de producción; similarmente, el algoritmo de detección de objetos anómalos emplea pruebas estadísticas para decidir si un objeto se desvía significativamente del patrón esperado."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                "Análisis de variabilidad mediante medidas de dispersión: En ambos casos se utilizan medidas como la desviación estándar y la varianza para cuantificar y analizar la variabilidad de los datos, ya sea en tiempos de ciclo o en características de imágenes."
                            )
                        ),
                        class_name="text-white mb-4 pl-6",
                    ),
                    rx.heading(
                        "Diferencias metodológicas",
                        size="6",
                        class_name="text-white mb-2",
                    ),
                    rx.unordered_list(
                        rx.list_item(
                            rx.text(
                                "Dominio de aplicación: Nuestro análisis se centra en datos temporales (tiempos de ciclo en procesos productivos), mientras que el artículo trabaja con datos espaciales y visuales (detección de objetos en imágenes)."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                "Técnicas específicas: El artículo implementa algoritmos especializados como Template Matching y Subtract Background que no tienen equivalente directo en nuestro análisis estadístico tradicional de procesos industriales."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                "Métricas de evaluación: En nuestro caso utilizamos p-valores e intervalos de confianza para evaluar los resultados, mientras que el sistema de detección descrito en el artículo emplea métricas como precisión, recall y tasas de falsos positivos/negativos típicas del campo de visión por computadora."
                            )
                        ),
                        rx.list_item(
                            rx.text(
                                "Enfoque de modelado: Nuestro enfoque es paramétrico y basado en distribuciones teóricas conocidas, mientras que el artículo utiliza un enfoque híbrido que combina técnicas estadísticas con procesamiento de imágenes y aprendizaje por patrones."
                            )
                        ),
                        class_name="text-white mb-4 pl-6",
                    ),
                    rx.text(
                        "Esta comparación ilustra cómo los principios estadísticos se aplican en diversos campos, aunque con adaptaciones específicas según el dominio de aplicación. Tanto nuestro análisis de procesos industriales como el sistema de detección de anomalías comparten fundamentos estadísticos, pero implementados de manera diferente según los objetivos específicos.",
                        class_name="text-white mb-4",
                    ),
                    
                    align="start",
                    spacing="4",
                    width="100%",
                ),
                class_name="p-8 rounded-2xl w-full bg-gradient-to-br from-black to-purple-800 shadow-lg overflow-x-auto",
            ),
            align="center",
            justify="center",
            spacing="8",
            width="100%",
        ),
        class_name="min-h-screen bg-gradient-to-br from-black to-purple-800 p-8",
    )

app = rx.App()
app.add_page(part1)
