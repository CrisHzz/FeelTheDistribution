import reflex as rx
from rxconfig import config
from data_processing.point1 import dataset_general , dataset_mtc , dataset_dispersion, dataset_form

class State(rx.State):
    """The app state."""
    ...

def part1() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.box(
                rx.hstack(
                    rx.heading(
                        "FeelTheDistribution",
                        size="7",
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
                            for row in dataset_general.values
                        ]
                    ),
                    variant="surface",
                    class_name="bg-black",
                ),
                class_name="p-8 rounded-2xl w-full bg-gradient-to-br from-black to-purple-800 shadow-lg overflow-x-auto",
            ),
            rx.box(
                rx.vstack(
                    rx.heading(
                        "Medidas de tendencia central, dispersión y forma",
                        size="7",
                        class_name="text-white mb-4",
                    ),
                    rx.text(
                        "Existen varios tipos de medidas que nos sirven para darle sentido a valor a los datos sea desde la parte basico como entender el promedio, como se desvian sus datos del centro y hasta que forma y comportamiento tienen ellos",
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
                        class_name="bg-black mb-8",
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
                        class_name="bg-black mb-8",
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
                        class_name="bg-black",
                    ),
                    align="start",
                    spacing="4",
                ),
                width="100%",
                padding="6",
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
