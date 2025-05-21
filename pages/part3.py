import reflex as rx
from rxconfig import config
from data_processing.point3Titanic import dataset_original_titanic
from data_processing.point3Titanic import dataset_cleaned_titanic
class State(rx.State):
    """The app state."""
    ...

def part3() -> rx.Component:
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
                    "Part 3 - Aplicando la estadistica en la vida real",
                    size="8",
                    class_name="text-white mb-4",
                ),
                rx.text(
                    "Aplicando estadística descriptiva e inferencial a problemas reales",
                    class_name="text-white text-xl mb-6",
                    style={"white-space": "pre-line"},
                ),
                
                # Contenido de los dos problemas planteados
                rx.vstack(
                    rx.box(
                        rx.heading(
                            "Problemas planteados",
                            size="6",
                            class_name="text-purple-300 mb-4",
                        ),
                        rx.text(
                            "Tenemos 2 problemas planteados que involucran la estadística descriptiva e inferencial, "
                            "que son útiles y dan significado de lo importante que es esta ciencia hoy en día.",
                            class_name="text-gray-200 mb-6",
                            style={"white-space": "pre-line"},
                        ),
                        class_name="w-full",
                    ),
                    
                    # Problema 1: Identificador de infecciones en las hojas de plantas
                    rx.box(
                        rx.heading(
                            "Problema 1: Identificador de infecciones en hojas de plantas",
                            size="5",
                            class_name="text-purple-200 mb-3",
                        ),
                        rx.text(
                            "El primero se trata de un identificador de infecciones en las hojas de plantas, "
                            "mostrando que en las hojas existen 3 tipos:",
                            class_name="text-gray-200 mb-2",
                        ),
                        rx.unordered_list(
                            rx.list_item("Infección tipo frijol con manchas con frijoles que contaminan a la planta"),
                            rx.list_item("Infección tipo angular que sus figuras tienen una forma más característica"),
                            rx.list_item("Plantas sanas que no tienen ninguna enfermedad alguna"),
                            class_name="text-gray-200 mb-4 pl-6",
                        ),
                        rx.text(
                            "Este problema ayudaría a que los agricultores primerizos puedan identificar "
                            "cómo están sus plantas y cómo controlar las plagas.",
                            class_name="text-gray-200 mb-3",
                        ),
                        rx.text(
                            "Sin embargo, no tiene un análisis estadístico tan grande que sirva como fruto "
                            "de juntar varios conocimientos del curso.",
                            class_name="text-gray-200 mb-3",
                        ),
                        rx.box(
                            rx.text(
                                "Si deseas ver cómo se realizó, se usó Redes neuronales y clasificación binaria",
                                class_name="text-gray-200",
                            ),
                            rx.link(
                                "Ver implementación",
                                href="https://github.com/CrisHzz/CureMyLeaf", 
                                class_name="text-blue-300 hover:text-blue-200 transition-colors",
                            ),
                            class_name="p-4 bg-purple-900/30 rounded-xl border border-purple-500/30 mb-8",
                        ),
                        class_name="w-full p-6 bg-gradient-to-br from-purple-900/40 to-black/40 rounded-2xl shadow-lg border border-purple-500/20 mb-6",
                    ),
                    
                    # Problema 2: El problema del Titanic
                    rx.box(
                        rx.heading(
                            "Campo 2: Ciencia de datos y análisis (El problema del Titanic)",
                            size="5",
                            class_name="text-purple-200 mb-3",
                        ),
                        rx.text(
                            "Nuestro segundo campo es la ciencia de datos y el análisis de ello. La ingeniería de sistemas y "
                            "todas estas ciencias computacionales han dado oportunidad a la revolución actual de la inteligencia "
                            "artificial y la implementación de modelos matemáticos y estadísticos para la solución de problemas actuales. "
                            "Con este campo queremos enfocarnos en problemas que utilicen estas ciencias que ya mencionamos y toda la "
                            "parte computacional moderna.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "Gracias a ello planteamos la siguiente aplicación de todo lo anterior y se trata del problema del Titanic. "
                            "Nos plantean un dataset roto y real con muchos valores vacíos, campos faltantes y bastantes datos inexistentes "
                            "que provienen del incidente real del barco RMS TITANIC ocurrido el 14 de abril de 1912 en un recorrido desde "
                            "Reino Unido a Estados Unidos. La distorsión del tiempo, las personas fallecidas sin reconocer y las condiciones "
                            "estadísticas de ese tiempo generaron la pérdida de muchos datos y el significado del estado base del dataset.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "La misión es limpiarlo, eliminar variables no relevantes, identificar factores de supervivencia y crear "
                            "una solución predictiva para poder conocer según datos ingresados si una persona llega a sobrevivir o no.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Temas y conceptos aplicados en la solucion:",
                            size="6",
                            class_name="text-purple-200 mt-4 mb-2",
                        ),
                        rx.unordered_list(
                            rx.list_item("Estadística descriptiva"),
                            rx.list_item("Representaciones gráficas"),
                            rx.list_item("Métodos de optimización"),
                            rx.list_item("Normalización y estandarización de datos"),
                            rx.list_item("Funciones matemáticas (Función Sigmoid)"),
                            rx.list_item("Validación cruzada"),
                            rx.list_item("Métodos de entrenamiento"),
                            rx.list_item("Matrices de confusión"),
                            class_name="text-gray-200 mb-4 pl-6",
                        ),
                        rx.heading(
                            "Ciencias y campos fundamentales de la ciencia de datos:",
                            size="6",
                            class_name="text-purple-200 mb-2",
                        ),
                        rx.unordered_list(
                            rx.list_item("Matemáticas"),
                            rx.list_item("Estadística"),
                            rx.list_item("Cálculo"),
                            rx.list_item("Álgebra lineal"),
                            rx.list_item("Álgebra tensorial"),
                            rx.list_item("Física"),
                            rx.list_item("Trigonometría"),
                            rx.list_item("Geometría analítica"),
                            rx.list_item("Paradigmas matemáticos de programación"),
                            rx.list_item("Teoría de grafos"),
                            class_name="text-gray-200 mb-4 pl-6",
                        ),
                        rx.box(
                            rx.text(
                                "Este segundo campo sera nuestra elección y desarrollaremos como problema principal",
                                class_name="text-white font-semibold",
                            ),
                            class_name="p-4 bg-purple-700/40 rounded-xl border border-purple-500/30 mb-4 text-center",
                        ),
                        rx.heading(
                            "Obtencion de los datos",
                            size="6",
                            class_name="text-purple-200 mb-2",
                        ),
                        rx.text (
                            "Los datos fueron obtenidos de Kaggle. Kaggle es una aplicación-comunidad que reúne todo lo enfocado en el análisis de datos, desde datasets, modelos de machine learning y fundamentos matemáticos y estadísticos para resolver problemas. El dataset original es el siguiente:",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    *[
                                        rx.table.column_header_cell(
                                            col,
                                            class_name="text-white"
                                        )
                                        for col in dataset_original_titanic.columns
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
                                    for row in dataset_original_titanic.values
                                ]
                            ),

                            variant="surface",
                            class_name="bg-black mb-8 w-full",
                        ),
                        rx.heading(
                            "Analisis estadistico",
                            size="8",
                            class_name="text-purple-200 mb-2 font-bold",
                        ),
                        rx.text(
                            "Antes de comenzar con el análisis estadístico, voy a recordar que el proceso de limpieza fue muy largo y, por lo tanto, para no agobiar con tantos procesos, voy a resaltar lo más importante y los temas requeridos por la entrega.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.box(
                            rx.text(
                                "Si deseas ver el proceso completo que se hizo , ingresa al siguiente link (El desarollo esta en ingles)",
                                class_name="text-gray-200",
                            ),
                            rx.link(
                                "Link a Github",
                                href='https://github.com/CrisHzz/TitanicSurvivors/blob/main/DataCleaning/DatasetCleaning.ipynb'
                            ),
                            class_name="text-blue-300 hover:text-blue-200 transition-colors mb-8",
                        ),

                        rx.heading(
                            "Limpiar Nan Values", 
                            size="6",
                            class_name="text-purple-200 mb-2",
                        ),
                        rx.text(
                            "El primer paso es eliminar todos los NaN (Not a Number) que son los valores vacíos. y generan muchos problemas a la hora de realizar análisis estadísticos.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.image(
                            src='/datasetnan.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.heading(
                            "Eliminacion de variables (Sibsp y parch)",
                            size="6",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "Este conjunto de datos contiene datos innecesarios y es fundamental eliminar algunas columnas que no son útiles para el modelo. Sin embargo, otras columnas requieren un análisis para determinar si podemos prescindir de ellas. En este caso, vamos a analizar 2 variables para eliminar: (SibSp) y (Parch), que representan la cantidad de hermanos o cónyuges a bordo y la cantidad de padres o hijos a bordo respectivamente, para evidenciar si son relevantes para la supervivencia.",
                            class_name="text-gray-200 mb-4"
                        ),
                        rx.image(
                            src='/sibsp.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto",
                        ),
                        rx.image(
                            src='/parch.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.text(
                            'Podemos concluir que la estructura familiar tuvo una influencia moderada en las probabilidades de supervivencia. Las personas que viajaban con un número pequeño de familiares (1-3) mostraron tasas de supervivencia más altas que aquellas que viajaban solas o con familias numerosas.\n\nSin embargo, esta variable por sí sola no es determinante. Las amplias barras de error en los gráficos indican una gran variabilidad, sugiriendo que otros factores como el género, la clase social y la edad probablemente jugaron un papel más decisivo',
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Costo del pasaje para la supervivencia",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "El costo del pasaje es una variable que puede influir en la supervivencia. En este caso, vamos a analizar su impacto en la supervivencia de los pasajeros.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.image(
                            src='/fare.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.text(
                            "El análisis muestra una clara correlación entre el precio del boleto y la supervivencia. Los datos revelan:",
                            class_name="text-gray-200 mb-2",
                        ),
                        rx.unordered_list(
                            [
                                rx.list_item(
                                    rx.text(
                                        "Tarifas bajas (£0 - £7.9): Tasa de supervivencia del 15% - 20%",
                                        class_name="text-gray-200"
                                    )
                                ),
                                rx.list_item(
                                    rx.text(
                                        "Tarifas medias (£14 - £27): Tasa de supervivencia del 40% - 45%",
                                        class_name="text-gray-200"
                                    )
                                ),
                                rx.list_item(
                                    rx.text(
                                        "Tarifas altas (£78 - £512): Tasa de supervivencia del 70% - 75%",
                                        class_name="text-gray-200"
                                    )
                                ),
                            ],
                            class_name="mb-4"
                        ),
                        rx.text(
                            "Los 'bins' son simplemente divisiones del rango de tarifas en 10 grupos iguales para facilitar el análisis visual. "
                            "Esta agrupación permite ver claramente cómo la tasa de supervivencia aumenta con el precio del boleto.\n\n"
                            "Esta variable es un fuerte predictor de supervivencia, probablemente porque refleja la clase social del pasajero, "
                            "que determinaba la ubicación del camarote (más cercano a la cubierta en primera clase) y el acceso prioritario a los botes salvavidas.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Edades faltantes en la media y llenado de valores en las cabinas y botes salvavidas",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "El conjunto de datos presenta valores faltantes en tres variables importantes: edad, cabina y bote salvavidas. Para manejar estos datos faltantes:\n\n"
                            "Para la edad, utilizamos la media como método de imputación. Esta decisión se basa en que la media es un estimador robusto que representa el valor típico de la edad de los pasajeros, minimizando el impacto en las distribuciones estadísticas y manteniendo la estructura general de los datos de edad.\n\n"
                            "En el caso de las cabinas (que son identificadores alfanuméricos que indican la ubicación del pasajero en el barco), creamos categorías especiales para aquellos registros sin asignación. Esto nos permite mantener la integridad del análisis sin perder información sobre los pasajeros sin cabina asignada.\n\n"
                            "De manera similar, para los botes salvavidas, asignamos identificadores especiales a aquellos registros sin información, permitiendo incluir estos casos en el análisis general de la distribución y uso de los botes salvavidas.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.image(
                            src='/meanclean.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.heading(
                            "Compresion de graficas y analisis de supervivencia",
                            size="7",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "En esta sección, vamos a analizar gráficamente y a través de herramientas estadísticas diversos factores que pueden determinar la supervivencia de una persona en el Titanic. Este análisis nos permitirá comprender mejor las variables que influyeron en las probabilidades de sobrevivir al desastre.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Distribución de edades en el barco y en los embarques",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.image(
                            src='/cakeage.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.text(
                            "Este grafico de pastel muestra la poblacion del barco diferenciacion sus edades",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Intervalo de confianza para encontrar la edad promedio",
                            size="4",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.image(
                            src='/confidentage.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.vstack(
                            rx.text(
                                "Tras analizar la edad promedio de los pasajeros del Titanic, podemos concluir que:",
                                class_name="text-gray-200",
                            ),
                            rx.ordered_list(
                                rx.list_item("La edad media de los pasajeros se encuentra entre 35.6 y 39.1 años, con un 95% de confianza."),
                                rx.list_item("Este intervalo relativamente estrecho nos indica que tenemos una buena estimación de la edad promedio real."),
                                rx.list_item("El análisis confirma que la mayoría de los pasajeros eran adultos de mediana edad, lo que coincide con el perfil demográfico de los viajeros transatlánticos de la época."),
                                rx.list_item("Tanto el método paramétrico (t de Student) como el no paramétrico (bootstrap) arrojan resultados similares, validando la robustez de nuestra estimación."),
                                class_name="text-gray-200 pl-6",
                            ),
                            rx.text(
                                "Este hallazgo es valioso para caracterizar correctamente a la población que viajaba en el Titanic y puede ayudar a contextualizar otros análisis demográficos y de supervivencia.",
                                class_name="text-gray-200",
                            ),
                            spacing="4",
                            class_name="mb-4",
                        ),

                        rx.heading(
                            "Mujeres vs Hombres , Quien sobrevive?",
                            size="4",
                            class_name="text-purple-200 mb-4",
                        ),

                        rx.image(
                            src='/survivors.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",

                        ),
                        rx.text(
                            "El análisis revela que las mujeres tenían una tasa de supervivencia significativamente más alta que los hombres. "
                            "Esto sugiere que las mujeres recibieron un trato preferencial en la evacuación, De aqui sale famosa frase Mujeres y niños primero.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Puertos de embarque",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "El RMS Titanic zarpó de Southampton, Inglaterra, y realizó escalas en Cherburgo, Francia, y Queenstown (actualmente Cobh), Irlanda ,vamos a analizar la distribucion de personas en los puertos de embarque ",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.image(
                            src='/embark.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.heading(
                            "Prueba de hipotesis: El dataset es lo suficientemente grande para crear un modelo predictivo con precision del 70%?",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),

                        rx.text(
                            "A la hora de entrenar nuestro modelo, vamos a plantear una prueba de hipótesis que nos permitirá determinar si la cantidad de datos es suficiente para obtener una precisión esperada del 70%. Para esto establecemos:",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "Hipótesis nula (H₀): La cantidad de datos no es suficiente para alcanzar un 70% de precisión.",
                            class_name="text-gray-200 font-bold mb-2",
                        ),
                        rx.text(
                            "Hipótesis alternativa (H₁): La cantidad de datos es suficiente para alcanzar un 70% de precisión.",
                            class_name="text-gray-200 font-bold mb-4",
                        ),
                        rx.text(
                            "Nivel de significancia (α): 0.05",
                            class_name="text-gray-200 font-bold mb-2",
                        ),
                        rx.text(
                            "Tamaño de la muestra (n): 351",
                            class_name="text-gray-200 font-bold mb-4",
                        ),
                        rx.image(
                            src='/hypothesis.png',
                            class_name="p-4 rounded-2xl bg-black shadow-lg overflow-x-auto mb-4",
                        ),
                        rx.text(
                            "Teóricamente, el tamaño de la muestra no es lo suficientemente grande para obtener esa precisión esperada en terminos teoricos; "
                            "sin embargo, empíricamente y computacionalmente, obtenemos una precisión del 76%.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.heading(
                            "Dataset limpio",
                            size="7",
                            class_name="text-purple-200 mb-4",
                        ),
            
                        rx.text(
                            "Después de todo este largo proceso, obtuvimos nuestro dataset limpiando variables, justificando decisiones y obteniendo resultados. De un total de aproximadamente 1309 filas y 12 columnas, quedamos con 292 filas y 10 columnas. Realmente, es mucha la cantidad de datos e información que se perdió eliminando los valores vacíos, incoherentes y NaN, demostrando que este problema del Titanic, al haber ocurrido hace más de 100 años, perdió mucha información y es un milagro que en esos tiempos se pudiera haber obtenido al menos algunos datos de las personas.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "De las variables obtenidas, algunas serán descartadas para la predicción, ya sea por su uso para otros propósitos o porque no sirven para alimentar el modelo. Específicamente, descartaremos Name, Cabin, boat y survived (esta última tiene un uso para la clasificación, mas no para la predicción).",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    *[
                                        rx.table.column_header_cell(
                                            col,
                                            class_name="text-white"
                                        )
                                        for col in dataset_cleaned_titanic.columns
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
                                    for row in dataset_cleaned_titanic.values
                                ]
                            ),

                            variant="surface",
                            class_name="bg-black mb-8 w-full",
                        ),
                        rx.heading(
                            "Nuestro modelo",
                            size="6",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "Para nuestro modelo, utilizamos una regresión logística, también conocida como función sigmoid. Esta técnica nos permite crear una clasificación de tipo binaria para obtener un resultado basado en una o más variables independientes. Las variables pasan por esta función y retornan un valor entre 0 y 1, donde los valores cercanos a 1 indican alta probabilidad de supervivencia, mientras que los valores cercanos a 0 indican baja probabilidad de supervivencia.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "En nuestro análisis, interpretamos los valores mayores a 0.6 como una supervivencia aceptable, siendo 1 una supervivencia casi perfecta. Por otro lado, los valores menores a 0.6 nos indican una probabilidad de supervivencia muy baja, casi nula. Esta función sigmoid nos permite transformar múltiples variables predictoras en una probabilidad clara y fácil de interpretar.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.markdown(
                            """
### Función Logística (Sigmoid)

$$P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + ... + \\beta_n x_n)}}$$

Donde:
- $P(y=1|x)$ representa la probabilidad de supervivencia
- $\\beta_0$ es el término de sesgo o intercepto
- $\\beta_1, \\beta_2, ..., \\beta_n$ son los coeficientes del modelo
- $x_1, x_2, ..., x_n$ son las variables independientes (edad, género, tarifa, etc.)
- $e$ es la base del logaritmo natural (aproximadamente 2.71828)

Esta función transforma cualquier valor de entrada en un rango entre 0 y 1, ideal para representar probabilidades.
            """,
                            class_name="text-gray-200 bg-purple-900/20 p-4 rounded-xl border border-purple-500/30 mb-4 overflow-x-auto",
                        ),
                        rx.heading(
                            "Titanic Survivors SandBox",
                            size="8",
                            class_name="text-purple-200 mb-4",
                        ),
                        rx.text(
                            "Es hora de probar nuestro modelo. Ingresa a esta página para explorar la caja de arena del proyecto, donde podrás hacer una predicción de supervivencia llenando la siguiente información:",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.vstack(
                            rx.unordered_list(
                                rx.list_item("Age: Edad de la persona", class_name="text-gray-200 font-bold"),
                                rx.list_item("Gender: El género de la persona", class_name="text-gray-200 font-bold"), 
                                rx.list_item("Fare: El costo del pasaje", class_name="text-gray-200 font-bold"),
                                rx.list_item("Embarked: El puerto de embarque", class_name="text-gray-200 font-bold"),
                                spacing="2",
                            ),
                            class_name="mb-4",
                            align="start",
                        ),
                        rx.text(
                            "¿Podrías haber sobrevivido al Titanic? ¡Pruébalo ahora!",
                            class_name="text-purple-200 font-bold mb-4",
                        ),
                        rx.box(
                            rx.text(
                                "NOTA: Al entrar por primera vez, debes esperar un tiempo mientras carga la página. "
                                "Luego de eso, recarga la página y una vez que aparezca arriba que el estado del modelo "
                                "está en verde, podrás hacer las pruebas. Si no usas la página en 15 minutos, se volverá "
                                "a apagar y deberás volver a esperar.",
                                class_name="text-gray-200",
                            ),
                            rx.link(
                                "Ver implementación",
                                href="https://titanicjsw.onrender.com/", 
                                class_name="text-blue-300 hover:text-blue-200 transition-colors",
                            ),
                            class_name="p-4 bg-purple-900/30 rounded-xl border border-purple-500/30 mb-8",
                        ),

                        class_name="w-full p-6 bg-gradient-to-br from-purple-900/40 to-black/40 rounded-2xl shadow-lg border border-purple-500/20 mb-6",
                    ),
                    class_name="w-full",
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
app.add_page(part3)