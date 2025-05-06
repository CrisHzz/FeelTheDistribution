import reflex as rx
from rxconfig import config
from data_processing.point1 import dataset_general, dataset_general_short , dataset_mtc , dataset_dispersion, dataset_form
from data_processing.point1Theory import df_pvalues

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
