import reflex as rx
from rxconfig import config
from pages.part1 import part1
from pages.part2 import part2
#from pages.part3 import part3


class State(rx.State):
    """The app state."""
    ...

def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading(
                "Feel The Distribution",
                size="9",
                class_name="text-white mb-8",
            ),
            rx.hstack(
                rx.button("Part 1", size="4", variant="soft", on_click=lambda: rx.redirect("/part1")),
                rx.button("Part 2", size="4", variant="soft", on_click=lambda: rx.redirect("/part2")),
                #rx.button("Part 3", size="4", variant="soft", on_click=lambda: rx.redirect("/part3")),
                spacing="4",
            ),
            align="center",
            justify="center",
            height="100vh",
        ),
        class_name="min-h-screen bg-gradient-to-br from-black to-purple-800",
    )

app = rx.App()

app.add_page(index)
app.add_page(part1, route="/part1")
app.add_page(part2, route="/part2")
#app.add_page(part3, route="/part3")