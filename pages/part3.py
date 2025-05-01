import reflex as rx
from rxconfig import config

class State(rx.State):
    """The app state."""
    ...

def part3() -> rx.Component:
    return rx.container(
        rx.heading("Part 3", size="7"),
        rx.text("Part 3 The titanic problem (In development)"),
        padding="2em",
        class_name="min-h-screen bg-gradient-to-br from-black to-purple-800"
    )

app = rx.App()
app.add_page(part3)