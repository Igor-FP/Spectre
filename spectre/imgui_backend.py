"""pygame backend for Dear ImGui with a complete keyboard map.

`imgui_bundle.python_backends.pygame_backend.PygameRenderer` only forwards
navigation and modifier keys, so `imgui.is_key_pressed(imgui.Key.f)` never fires
and application shortcuts are impossible.  This subclass maps the whole
keyboard and still feeds printable characters to text fields.
"""

from __future__ import annotations

import pygame
from imgui_bundle import imgui
from imgui_bundle.python_backends.pygame_backend import PygameRenderer


def _build_extra_key_map() -> dict:
    """pygame key code -> imgui.Key for everything the base map leaves out."""
    mapping = {}

    def add(pygame_name: str, imgui_name: str) -> None:
        code = getattr(pygame, pygame_name, None)
        key = getattr(imgui.Key, imgui_name, None)
        if code is not None and key is not None:
            mapping[code] = key

    for letter in "abcdefghijklmnopqrstuvwxyz":
        add(f"K_{letter}", letter)
    for digit in range(10):
        add(f"K_{digit}", f"_{digit}")
    for n in range(1, 25):
        add(f"K_F{n}", f"f{n}")
    for digit in range(10):
        add(f"K_KP{digit}", f"keypad{digit}")
        add(f"K_KP_{digit}", f"keypad{digit}")

    for pygame_name, imgui_name in (
        ("K_SPACE", "space"),
        ("K_MINUS", "minus"),
        ("K_EQUALS", "equal"),
        ("K_LEFTBRACKET", "left_bracket"),
        ("K_RIGHTBRACKET", "right_bracket"),
        ("K_BACKSLASH", "backslash"),
        ("K_SEMICOLON", "semicolon"),
        ("K_QUOTE", "apostrophe"),
        ("K_BACKQUOTE", "grave_accent"),
        ("K_COMMA", "comma"),
        ("K_PERIOD", "period"),
        ("K_SLASH", "slash"),
        ("K_CAPSLOCK", "caps_lock"),
        ("K_SCROLLLOCK", "scroll_lock"),
        ("K_NUMLOCK", "num_lock"),
        ("K_NUMLOCKCLEAR", "num_lock"),
        ("K_PRINTSCREEN", "print_screen"),
        ("K_PRINT", "print_screen"),
        ("K_PAUSE", "pause"),
        ("K_MENU", "menu"),
        ("K_KP_PERIOD", "keypad_decimal"),
        ("K_KP_DIVIDE", "keypad_divide"),
        ("K_KP_MULTIPLY", "keypad_multiply"),
        ("K_KP_MINUS", "keypad_subtract"),
        ("K_KP_PLUS", "keypad_add"),
        ("K_KP_EQUALS", "keypad_equal"),
    ):
        add(pygame_name, imgui_name)

    return mapping


class SpectreRenderer(PygameRenderer):
    """PygameRenderer with the full keyboard mapped."""

    #: Wheel notches per event; 1.0 keeps `io.mouse_wheel` in natural units.
    wheel_scale = 1.0

    def _map_keys(self) -> None:
        super()._map_keys()
        self.key_map.update(_build_extra_key_map())

    def process_event(self, event) -> bool:
        """Feed one pygame event to ImGui.

        Reimplemented rather than extended: the base class suppresses text input
        for any key present in `key_map`, which would break text fields now that
        letters are mapped.
        """
        io = self.io

        if event.type == pygame.MOUSEMOTION:
            io.add_mouse_pos_event(event.pos[0], event.pos[1])
            return True

        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
            # pygame also reports the wheel as buttons 4/5; ImGui has its own
            # wheel event, so ignore those here.
            if event.button <= 3:
                io.add_mouse_button_event(event.button - 1, event.type == pygame.MOUSEBUTTONDOWN)
            return True

        if event.type == pygame.MOUSEWHEEL:
            io.add_mouse_wheel_event(event.x * self.wheel_scale, event.y * self.wheel_scale)
            return True

        if event.type in (pygame.KEYDOWN, pygame.KEYUP):
            down = event.type == pygame.KEYDOWN
            key = self.key_map.get(event.key)
            if key is not None:
                io.add_key_event(key, down=down)
            modifier = self.modifier_map.get(event.key)
            if modifier is not None:
                io.add_key_event(modifier, down=down)
            if down:
                for char in event.unicode:
                    code = ord(char)
                    # Skip control characters: Enter/Tab/Backspace/Esc arrive as
                    # key events, and inserting them as text confuses ImGui.
                    if 0x20 <= code < 0x10000 and code != 0x7F:
                        io.add_input_character(code)
            return True

        if event.type == pygame.WINDOWFOCUSLOST:
            io.add_focus_event(False)
            return True

        if event.type == pygame.WINDOWFOCUSGAINED:
            io.add_focus_event(True)
            return True

        if event.type == pygame.VIDEORESIZE:
            surface = pygame.display.get_surface()
            # pygame does not resize the existing surface for us.
            pygame.display.set_mode((event.w, event.h), flags=surface.get_flags())
            self._update_textures()  # the font atlas texture is now invalid
            io.display_size = event.size
            return True

        return False
