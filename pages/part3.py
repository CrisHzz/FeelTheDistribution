import reflex as rx
from rxconfig import config

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
                    "The Titanic Problem",
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
                            "Problema 2: El problema del Titanic",
                            size="5",
                            class_name="text-purple-200 mb-3",
                        ),
                        rx.text(
                            "El segundo problema es el del Titanic. Nos plantean un dataset roto con muchos valores vacíos, "
                            "variables que no son relevantes y muchos valores desconocidos o nulos. El problema es poder "
                            "limpiarlo y seleccionarlo para así crear un modelo que permita identificar quién sobreviviría "
                            "en ese desastre según varios parámetros.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.text(
                            "Esto aplica mucha estadística demostrando las variables relevantes, el relleno de datos "
                            "y las proyecciones de supervivencia de las personas.",
                            class_name="text-gray-200 mb-4",
                        ),
                        rx.box(
                            rx.text(
                                "Será nuestra elección y desarrollaremos como problema principal",
                                class_name="text-white font-semibold",
                            ),
                            class_name="p-4 bg-purple-700/40 rounded-xl border border-purple-500/30 mb-4 text-center",
                        ),
                    ),
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