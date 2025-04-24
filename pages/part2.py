import reflex as rx
from rxconfig import config
from data_processing.point2 import dataset_2, dataset_2_short, dataset_dispersion2, dataset_dispersion2, dataset_form2

class State(rx.State):
    """The app state."""
    ...

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