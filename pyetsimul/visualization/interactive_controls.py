"""Centralized keyboard controls for interactive plots."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyetsimul.types import Point3D, Position3D

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent

    from pyetsimul.core import Eye


class InteractiveControls:
    """Centralized keyboard controls for interactive eye tracking plots."""

    def __init__(
        self,
        eyes: list["Eye"],
        target_point: Point3D,
        step_size: float = 2.5e-3,
        initial_eye_positions: list[Position3D] | None = None,
        initial_target_position: Point3D | None = None,
        custom_handlers: dict[str, Callable] | None = None,
    ) -> None:
        """Initialize interactive controls.

        Args:
            eyes: List of Eye objects to control (IJKL moves all together).
            target_point: Target point for gaze.
            step_size: Movement step size in meters.
            initial_eye_positions: Initial positions for each eye (defaults to current).
            initial_target_position: Initial target position (defaults to current).
            custom_handlers: Additional key handlers.

        """
        self.eyes = eyes
        self.target_point = target_point

        self.step_size = step_size
        self.initial_eye_positions = initial_eye_positions or [
            Position3D(eye.position.x, eye.position.y, eye.position.z) for eye in eyes
        ]
        self.initial_target_position = initial_target_position or Point3D(
            target_point.x, target_point.y, target_point.z
        )
        self.custom_handlers = custom_handlers or {}
        self._update_callback: Callable | None = None

    def set_update_callback(self, callback: Callable) -> None:
        """Set callback function to call after position changes."""
        self._update_callback = callback

    def handle_key_press(self, event: "KeyEvent") -> bool:
        """Handle keyboard input and return True if key was handled."""
        standard_actions = {
            # Reset
            " ": self.reset_positions,
            # Target movement
            "up": lambda: self._move_target(0, 0, self.step_size),
            "Up": lambda: self._move_target(0, 0, self.step_size),
            "↑": lambda: self._move_target(0, 0, self.step_size),
            "down": lambda: self._move_target(0, 0, -self.step_size),
            "Down": lambda: self._move_target(0, 0, -self.step_size),
            "↓": lambda: self._move_target(0, 0, -self.step_size),
            "left": lambda: self._move_target(-self.step_size, 0, 0),
            "Left": lambda: self._move_target(-self.step_size, 0, 0),
            "←": lambda: self._move_target(-self.step_size, 0, 0),
            "right": lambda: self._move_target(self.step_size, 0, 0),
            "Right": lambda: self._move_target(self.step_size, 0, 0),
            "→": lambda: self._move_target(self.step_size, 0, 0),
            # Eye movement (all eyes together)
            "j": lambda: self._move_eyes(-self.step_size, 0, 0),
            "l": lambda: self._move_eyes(self.step_size, 0, 0),
            "i": lambda: self._move_eyes(0, 0, self.step_size),
            "k": lambda: self._move_eyes(0, 0, -self.step_size),
            ".": lambda: self._move_eyes(0, -self.step_size, 0),
            ",": lambda: self._move_eyes(0, self.step_size, 0),
            # Special keys
            "escape": InteractiveControls._handle_escape,
        }

        # Check custom handlers first
        if event.key in self.custom_handlers:
            result = self.custom_handlers[event.key](event, self)
            if result is not False:
                self._trigger_update()
                return True

        # Check standard actions
        if event.key in standard_actions:
            standard_actions[event.key]()
            self._trigger_update()
            return True

        return False

    def _move_target(self, dx: float, dy: float, dz: float) -> None:
        """Move target by the specified amounts."""
        self.target_point = Point3D(self.target_point.x + dx, self.target_point.y + dy, self.target_point.z + dz)

    def _move_eyes(self, dx: float, dy: float, dz: float) -> None:
        """Move all eyes by the specified amounts (simulates head movement)."""
        for eye in self.eyes:
            eye.trans[0, 3] += dx
            eye.trans[1, 3] += dy
            eye.trans[2, 3] += dz
            eye.position = Position3D(eye.trans[0, 3], eye.trans[1, 3], eye.trans[2, 3])

    def reset_positions(self) -> None:
        """Reset all eyes and target to initial positions."""
        for eye, initial_pos in zip(self.eyes, self.initial_eye_positions, strict=True):
            eye.trans[0, 3] = initial_pos.x
            eye.trans[1, 3] = initial_pos.y
            eye.trans[2, 3] = initial_pos.z
            eye.position = Position3D(eye.trans[0, 3], eye.trans[1, 3], eye.trans[2, 3])

        self.target_point = Point3D(
            self.initial_target_position.x, self.initial_target_position.y, self.initial_target_position.z
        )

    @staticmethod
    def _handle_escape() -> None:
        """Handle escape key press - close the current figure."""
        plt.close("all")

    def _trigger_update(self) -> None:
        """Trigger update callback if set."""
        if self._update_callback:
            self._update_callback()

    @staticmethod
    def print_controls(include_reset: bool = True, additional_controls: dict[str, str] | None = None) -> None:
        """Print standardized control instructions."""
        print("CONTROLS:")
        print("Target Movement (Arrow keys):")
        print("  ↑/↓: Move target up/down")
        print("  ←/→: Move target left/right")
        print()
        print("Eye Movement (I/K/J/L/./):")
        print("  I/K: Move eye up/down")
        print("  J/L: Move eye left/right")
        print("  ./,: Move eye closer/farther from camera")
        if include_reset:
            print("Reset (Space): Reset eye and target to initial positions")
        if additional_controls:
            for description, keys in additional_controls.items():
                print(f"{description}: {keys}")
