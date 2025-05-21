import reflex as rx
from rxconfig import config
from data_processing.point2 import dataset_2_short, dataset_dispersion2, dataset_dispersion2, dataset_form2
from data_processing.regressionPoint2 import corre_variables, best_model, dataset_estandarizado_short, regression_instancia, model_message, regression_data_model, model_validation, model_optimization

class State(rx.State):
    """The app state."""
    ...

# Convertir el objeto regression_instancia a texto antes de pasarlo a rx.code_block
regression_instancia_text = str(regression_instancia)

def part2() -> rx.Component:
    return rx.container(
        rx.color_mode.button(
            position="top-right", 
            class_name="bg-yellow-500 text-black hover:bg-yellow-600 transition-colors px-4 py-2 rounded-lg"
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
                rx.heading(
                    "Dataset Parte #2",
                    size="8",
                    class_name="text-white mb-4",
                ),

                rx.text(
                    """Caso de estudio: Eficiencia Energética en Procesos Industriales

                    Una planta industrial está interesada en entender y optimizar el consumo energético de uno de sus procesos 
                    principales. Se sospecha que el consumo de energía (kWh) está relacionado con la velocidad de producción 
                    (unidades/hora). Se han recolectado datos de 50 jornadas de producción, registrando ambas variables.""",
                    class_name="text-white mb-4",
                    style={"white-space": "pre-line"},
                ),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            *[
                                rx.table.column_header_cell(
                                    col,
                                    class_name="text-white"
                                )
                                for col in dataset_2_short.columns
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
                            for row in dataset_2_short.values
                        ]
                    ),
                    variant="surface",
                    class_name="bg-black",
                ),
                class_name="p-8 rounded-2xl w-full bg-gradient-to-br from-black to-purple-800 shadow-lg overflow-x-auto",
            ),
            rx.box(
                rx.heading(
                    "Medidas de tendencia central, dispersión y forma",
                    size="8",
                    class_name="text-white mb-4",
                ),
                rx.heading(
                    "Medidas de tendencia central",
                    size="6",
                    class_name="text-white mb-4",
                ),
                rx.text(
                    "Existen varios tipos de medidas que nos sirven para darle sentido a valor a los datos sea desde el apartado basico como entender el promedio, como se desvian sus datos del centro y hasta que forma y comportamiento tienen ellos",
                    style={"white-space": "pre-line"},
                    class_name="text-white mb-4",),
                    rx.table.root(
                        rx.table.header(
                            rx.table.row(
                                *[
                                    rx.table.column_header_cell(
                                        col,
                                        class_name="text-white"
                                    )
                                    for col in dataset_dispersion2.columns
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
                                for row in dataset_dispersion2.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black",
                    ),
                    rx.heading(
                        "Medidas de dispersión",
                        size="6",
                        class_name="text-white mb-4 pt-4",

                    ),
                    rx.text(
                        "Las medidas de dispersión nos indican qué tan dispersos están los datos alrededor de su valor central. La desviación estándar, varianza y rango nos ayudan a entender la variabilidad de los datos.",
                        style={"white-space": "pre-line"},
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
                                    for col in dataset_dispersion2.columns
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
                                for row in dataset_dispersion2.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black",
                    ),
                    rx.heading(
                        "Medidas de forma",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "Las medidas de forma nos indican la asimetría y la forma de la distribución de los datos. La asimetría nos dice si los datos están sesgados hacia un lado, mientras que la curtosis nos indica la 'altura' de la distribución.",
                        style={"white-space": "pre-line"},
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
                                    for col in dataset_form2.columns
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
                                for row in dataset_form2.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black",
                    ),
                    rx.heading(
                        "Grafico de histograma",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El histograma es una representación gráfica de los datos a traves del tiempo y poder evidenciar como es la distribucion de los datos",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.image(
                        src='/histogram2.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto",
                    ),

                    rx.heading(
                        "Grafico de dispersion",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El grafico de dispersion nos permite ver por encima la distribucion de los datos , datos atipicos y encontrar que tipo de correlacion existe entre las variables que se estan analizando",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.image(
                        src='/scatter.png',
                        class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto",
                    ),
                    rx.heading(
                        "Encontrar la coeficiente de correlacion entre las variables",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El coeficiente de correlación es una medida estadística que indica la fuerza y dirección de la relación lineal entre dos variables. Varía entre -1 y 1, donde 1 indica una correlación positiva perfecta, -1 una correlación negativa perfecta y 0 ninguna correlación.",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        corre_variables,
                        language="python",
                        show_line_numbers=True,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    rx.heading( 
                        "Recomendación de modelo de regresión",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "Existen varios tipos de modelos de regresion que nos pueden servir para desarollar este proyecto , todo depende del tipo de datos que se tienen , la cantidad de variables , el peso de estas para la prediccion entre muchas otras caracteristicas. En este caso a pesar de plantear 3 tipos de modelo de regresion , la regresion lineal es la que mejor se adapta a los datos y el comportamiento de estos, ridge y lasso nos darian el mismo rendimiento pues las condiciones para estos no son las indicadas y el R2 nos da igual en los 3 casos",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        best_model,
                        language="python",
                        show_line_numbers=True,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    rx.heading(
                        "Estandarizacion del dataset",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "La estandarización es un proceso que transforma los datos para que tengan una media de 0 y una desviación estándar de 1. Esto es útil para comparar variables que están en diferentes escalas o unidades. ayuda bastante a la hora de alimentar nuestro modelo o regresion haciendo los datos mas pequeños y disminuir el tiempo de entreno",
                        style={"white-space": "pre-line"},
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
                                    for col in dataset_estandarizado_short.columns
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
                                for row in dataset_estandarizado_short.values
                            ]
                        ),
                        variant="surface",
                        class_name="bg-black",
                    ),
                    rx.heading(
                        "Implementacion de regresion lineal",
                        size="7",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "Ahora ya encontramos el tipo de regresion adecuada para el conjunto de datos , gracias a ello podemos crear una regresion lineal que permitira hacer una prediccion dada una variable indepediente",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",

                    ),
                    rx.markdown(
                        "$$Y= WX + b $$",
                    
               
                    ),
                    rx.text(
                        "Donde W son los pesos y b el sesgo de la regresion lineal",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        regression_instancia_text,
                        language="python",
                        show_line_numbers=True,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    rx.heading(
                        "Entreno del modelo , pesos y sesgos de este",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "Vamos a entrenar nuestro modelo utilizando aproximadamente un 70% para los datos de entreno y un 30% para los datos de testeo, esto nos permitira ver si el modelo es capaz de predecir los datos que no ha visto antes , ademas de visualizar el peso y sesgo del modelo",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.code_block(
                        model_message,
                        language="python",
                        show_line_numbers=True,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    rx.markdown(
                        "$$ Y = 0.8283X -0.0237 $$"
                    ),
                    rx.text(
                        "Evaluacion del rendimiento de la regresion lineal",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),

                    rx.text(
                        "Existen varias metricas para evaluar nuestro rendimiento del modelo , en este caso usaremos el error cuadratico medio , el coeficiente de determinacion y la raiz del error cuadratico medio",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),

                    rx.code_block(
                        regression_data_model,
                        language="python",
                        show_line_numbers=True,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    
                    # Nueva sección para validación del modelo (punto d)
                    rx.heading(
                        "Validación del Modelo de Regresión",
                        size="7",
                        class_name="text-white mb-4 pt-6",
                    ),
                    rx.text(
                        "La validación del modelo es un paso crucial para determinar si un modelo de regresión lineal es adecuado para los datos. Esta validación incluye verificar los supuestos del modelo, realizar pruebas de hipótesis para los parámetros, construir intervalos de confianza y analizar los residuos.",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    
                    rx.code_block(
                        model_validation,
                        language="markdown",
                        show_line_numbers=False,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    
                    rx.heading(
                        "Análisis Visual de Residuos",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El análisis visual de los residuos es fundamental para verificar los supuestos del modelo de regresión lineal. A continuación se presentan gráficos que nos ayudan a evaluar dichos supuestos:",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    
                    rx.cond(
                        model_validation.__contains__("NOTA"),
                        rx.text(
                            "Para visualizar los gráficos de residuos, instale las dependencias indicadas arriba y ejecute la aplicación nuevamente.",
                            class_name="text-white text-center p-4 bg-red-900/50 rounded-xl mb-6"
                        ),
                        rx.vstack(
                            rx.heading(
                                "Residuos vs Valores Ajustados",
                                size="5",
                                class_name="text-white text-center font-bold mt-6"
                            ),
                            rx.text(
                                "Este gráfico permite visualizar si los residuos se distribuyen de manera aleatoria alrededor de cero, lo que indicaría que el modelo es apropiado. Patrones en este gráfico pueden revelar problemas con el modelo.",
                                class_name="text-white mb-4 text-center"
                            ),
                            rx.image(
                                src='/residuos_vs_ajustados.png',
                                height="500px",
                                class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 object-contain",
                            ),
                            
                            rx.heading(
                                "Gráfico Q-Q de Residuos",
                                size="5",
                                class_name="text-white text-center font-bold mt-6"
                            ),
                            rx.text(
                                "El gráfico Q-Q compara los cuantiles de los residuos con los cuantiles teóricos de una distribución normal. Si los puntos siguen aproximadamente la línea diagonal, los residuos siguen una distribución normal.",
                                class_name="text-white mb-4 text-center"
                            ),
                            rx.image(
                                src='/qq_residuos.png',
                                height="500px",
                                class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 object-contain",
                            ),
                            
                            rx.heading(
                                "Histograma de Residuos",
                                size="5",
                                class_name="text-white text-center font-bold mt-6"
                            ),
                            rx.text(
                                "El histograma muestra la distribución de los residuos. Idealmente, debería seguir una forma aproximadamente normal y centrada en cero, lo que indicaría que los errores del modelo son aleatorios.",
                                class_name="text-white mb-4 text-center"
                            ),
                            rx.image(
                                src='/histograma_residuos.png',
                                height="500px",
                                class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 object-contain",
                            ),
                            width="100%",
                            spacing="4",
                            class_name="mb-6",
                        )
                    ),
                    
                    rx.text(
                        "En base a esta validación, podemos concluir si el modelo de regresión lineal es adecuado para predecir el consumo de energía en función de la velocidad de producción, y cuáles son sus limitaciones y fortalezas.",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    
                    # Nueva sección para el punto e: Utilización del modelo
                    rx.heading(
                        "Aplicación y Optimización del Modelo",
                        size="7",
                        class_name="text-white mb-4 pt-6",
                    ),
                    rx.text(
                        "En esta sección aplicaremos el modelo para realizar predicciones específicas, construir intervalos de predicción, optimizar la eficiencia energética y proporcionar recomendaciones concretas para la empresa.",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    
                    rx.code_block(
                        model_optimization,
                        language="markdown",
                        show_line_numbers=False,
                        class_name="bg-black text-white p-4 rounded-lg my-4 w-full overflow-auto",
                    ),
                    
                    rx.heading(
                        "Visualización de la Eficiencia Energética",
                        size="6",
                        class_name="text-white mb-4 pt-4",
                    ),
                    rx.text(
                        "El siguiente gráfico muestra la relación entre la velocidad de producción y la eficiencia energética (consumo por unidad producida). El punto marcado en rojo representa la velocidad óptima que minimiza el consumo energético por unidad.",
                        style={"white-space": "pre-line"},
                        class_name="text-white mb-4",
                    ),
                    rx.image(
                        src='/eficiencia_vs_velocidad.png',
                        height="600px",
                        class_name="p-4 rounded-2xl bg-black shadow-lg mb-8 object-contain",
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
app.add_page(part2)

#e
