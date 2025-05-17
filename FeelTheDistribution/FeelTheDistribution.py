import reflex as rx
from rxconfig import config
from pages.part1 import part1
from pages.part2 import part2
from pages.part3 import part3
from pages.bibliografias import bibliografias

class State(rx.State):
    """The app state."""
    pass

def index() -> rx.Component:
    # HTML de la animación con Tailwind CSS
    animation_html = """
    <div class="flex justify-center items-center h-96 bg-transparent p-4 w-full max-w-[800px] mx-auto">
        <div class="flex justify-center items-end h-full w-full">
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 2px; --delay: 0s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 12px; --delay: -0.25s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 40px; --delay: -0.5s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 98px; --delay: -0.75s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 168px; --delay: -1.0s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 200px; --delay: -1.25s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 168px; --delay: -1.5s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 98px; --delay: -1.75s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 40px; --delay: -2.0s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 12px; --delay: -2.25s"></div>
            <div class="wave-point w-3 mx-1.5 bg-white bg-opacity-80 h-0" style="--height: 2px; --delay: -2.5s"></div>
        </div>
    </div>
    """

    # Estilos CSS para la animación y efectos adicionales
    animation_styles = """
    <style>
        @keyframes wave {
            0% { height: 0; }
            50% { height: var(--height); }
            100% { height: 0; }
        }
        .wave-point {
            animation: wave 2.5s infinite cubic-bezier(0.4, 0, 0.6, 1);
            animation-delay: var(--delay);
        }
        .bg-pattern {
            background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.1) 1px, transparent 0);
            background-size: 40px 40px;
        }
        .button-glow {
            box-shadow: 0 0 15px rgba(147, 51, 234, 0.2);
        }
        .button-glow:hover {
            box-shadow: 0 0 25px rgba(147, 51, 234, 0.4);
        }
    </style>
    """

    return rx.container(
        rx.box(
            class_name="absolute inset-0 bg-pattern z-[-1]",
        ),
        rx.vstack(
            rx.heading(
                "Feel The Distribution",
                size="9",
                class_name="text-white mb-2 text-center w-full font-bold tracking-wider drop-shadow-lg",
            ),
            rx.text(
                "By Cristian Hernandez - Jonathan Garcia",
                size="4",
                class_name="text-white/80 mb-8 text-center w-full italic tracking-wide",
            ),
            rx.html(animation_styles + animation_html),
            rx.hstack(
                rx.button(
                    "Part 1",
                    size="4",
                    variant="soft",
                    on_click=lambda: rx.redirect("/part1"),
                    class_name="button-glow bg-white/10 hover:bg-white/20 transition-all duration-300 px-8 py-3 rounded-xl backdrop-blur-md border border-white/10 hover:border-white/20 text-white font-medium tracking-wide",
                ),
                rx.button(
                    "Part 2",
                    size="4",
                    variant="soft",
                    on_click=lambda: rx.redirect("/part2"),
                    class_name="button-glow bg-white/10 hover:bg-white/20 transition-all duration-300 px-8 py-3 rounded-xl backdrop-blur-md border border-white/10 hover:border-white/20 text-white font-medium tracking-wide",
                ),
                rx.button(
                    "Part 3",
                    size="4",
                    variant="soft",
                    on_click=lambda: rx.redirect("/part3"),
                    class_name="button-glow bg-white/10 hover:bg-white/20 transition-all duration-300 px-8 py-3 rounded-xl backdrop-blur-md border border-white/10 hover:border-white/20 text-white font-medium tracking-wide",
                ),
                rx.button(
                    "Bibliografías",
                    size="4",
                    variant="soft",
                    on_click=lambda: rx.redirect("/bibliografias"),
                    class_name="button-glow bg-white/10 hover:bg-white/20 transition-all duration-300 px-8 py-3 rounded-xl backdrop-blur-md border border-white/10 hover:border-white/20 text-white font-medium tracking-wide",
                ),
                spacing="6",
                class_name="mt-12",
            ),
            align="center",
            justify="center",
            height="100vh",
            class_name="w-full",
        ),
        class_name="min-h-screen bg-gradient-to-br from-purple via-purple-600 to-purple-500 flex justify-center relative overflow-hidden",
    )

app = rx.App()

app.add_page(index)
app.add_page(part1, route="/part1")
app.add_page(part2, route="/part2")
app.add_page(part3, route="/part3")
app.add_page(bibliografias, route="/bibliografias")