import reflex as rx
from rxconfig import config

from data_processing.point1 import dataset_general

class State(rx.State):
    """The app state."""
    ...

def part1() -> rx.Component:
    return rx.container(
        rx.color_mode.button(position="top-right"),
        rx.hstack(
            rx.heading(
                "FeelTheDistribution",
                size="7",
                class_name="text-white",
            ),
            rx.hstack(
                rx.link("Part 1", href="/part1", class_name="text-white"),
                rx.link("Part 2", href="/part2", class_name="text-white"),
                rx.link("Part 3", href="/part3", class_name="text-white"),
                spacing="8",
            ),
            width="100%",
            justify="between",
            align="center",
            padding="4",
            class_name="bg-black bg-opacity-50",
        ),
        rx.vstack(
            rx.box(
                rx.heading(
                    "Dataset Punto 1", 
                    size="7",
                    class_name="text-white mb-4",
                ),
                rx.heading(
                    "Creación del conjunto de datos",
                    size="5",
                    class_name="text-white mb-4",
                ),
                rx.text(
                    "Se crea un dataset usando las librerías de numpy y pandas, estos nos generaran los datasets para este apartado. La columna Línea A (s) representa el tiempo demora completar una operación en esta linea de operación para un producto lo mismo con la columna Línea B (s). Siendo un total de 100 registros por cada linea de producción.",
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
            align="center",
            justify="center",
            height="100vh",
            spacing="8",
            width="100%",
        ),
        class_name="min-h-screen bg-gradient-to-br from-black to-purple-800 p-8",
    )

app = rx.App()
app.add_page(part1)
