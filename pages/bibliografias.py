import reflex as rx
from rxconfig import config

class State(rx.State):
    """The app state."""
    ...

def bibliografias() -> rx.Component:
    return rx.container(
        rx.heading("Bibliografia", size="7"),
        rx.text("Bibliografias del proyecto y creditos"),
        padding="2em",
        class_name="min-h-screen bg-gradient-to-br from-black to-purple-800"
    )

app = rx.App()
app.add_page(bibliografias)