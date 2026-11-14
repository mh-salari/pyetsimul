"""Centralized keyboard controls for interactive plots."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from pyetsimul.log import info
from pyetsimul.types import Point3D, Position3D

if TYPE_CHECKING:
    from matplotlib.backend_bases import KeyEvent

    from pyetsimul.core import Eye


_WORLD_AXES = ("x", "y", "z")


class InteractiveControls:
    """Centralized keyboard controls for interactive eye tracking plots."""

    def __init__(
        self,
        eyes: list["Eye"],
        target_point: Point3D,
        step_size: float = 2.5,
        initial_eye_positions: list[Position3D] | None = None,
        initial_target_position: Point3D | None = None,
        custom_handlers: dict[str, Callable] | None = None,
        screen_axes: tuple[str, str] = ("x", "z"),
    ) -> None:
        """Initialize interactive controls.

        Args:
            eyes: List of Eye objects to control (IJKL moves all together).
            target_point: Target point for gaze.
            step_size: Movement step size in mm.
            initial_eye_positions: Initial positions for each eye (defaults to where each eye is placed).
            initial_target_position: Initial target position (defaults to current).
            custom_handlers: Additional key handlers.
            screen_axes: World axes of the screen plane, horizontal then vertical ("x", "y" or "z" each). The
                arrow keys move the target along them, J/L and I/K move the eyes along them, and ./, move
                the eyes along the third axis.

        Raises:
            ValueError: If ``screen_axes`` are not two different world axes.

        """
        if len(set(screen_axes)) != len(_WORLD_AXES) - 1 or not set(screen_axes) <= set(_WORLD_AXES):
            raise ValueError(f"screen_axes must be two different axes out of {_WORLD_AXES}, got {screen_axes}")
        depth_axis = next(axis for axis in _WORLD_AXES if axis not in screen_axes)
        # Unit steps along the screen's horizontal and vertical axes and along the axis through the screen.
        self._right, self._up, self._away = (
            tuple(float(axis == name) for axis in _WORLD_AXES) for name in (*screen_axes, depth_axis)
        )

        self.eyes = eyes
        self.target_point = target_point

        self.step_size = step_size
        self.initial_eye_positions = (
            [eye.placement for eye in eyes] if initial_eye_positions is None else initial_eye_positions
        )
        self.initial_target_position = (
            Point3D(target_point.x, target_point.y, target_point.z)
            if initial_target_position is None
            else initial_target_position
        )
        self.custom_handlers = {} if custom_handlers is None else custom_handlers
        self._update_callback: Callable | None = None

    def set_update_callback(self, callback: Callable) -> None:
        """Set callback function to call after position changes."""
        self._update_callback = callback

    def handle_key_press(self, event: "KeyEvent") -> bool:
        """Handle keyboard input and return True if key was handled."""
        standard_actions = {
            # Reset
            " ": self.reset_positions,
            # Target movement, in the screen plane
            "up": lambda: self._move_target(*self._step(self._up, 1)),
            "Up": lambda: self._move_target(*self._step(self._up, 1)),
            "↑": lambda: self._move_target(*self._step(self._up, 1)),
            "down": lambda: self._move_target(*self._step(self._up, -1)),
            "Down": lambda: self._move_target(*self._step(self._up, -1)),
            "↓": lambda: self._move_target(*self._step(self._up, -1)),
            "left": lambda: self._move_target(*self._step(self._right, -1)),
            "Left": lambda: self._move_target(*self._step(self._right, -1)),
            "←": lambda: self._move_target(*self._step(self._right, -1)),
            "right": lambda: self._move_target(*self._step(self._right, 1)),
            "Right": lambda: self._move_target(*self._step(self._right, 1)),
            "→": lambda: self._move_target(*self._step(self._right, 1)),
            # Eye movement (all eyes together)
            "j": lambda: self._move_eyes(*self._step(self._right, -1)),
            "l": lambda: self._move_eyes(*self._step(self._right, 1)),
            "i": lambda: self._move_eyes(*self._step(self._up, 1)),
            "k": lambda: self._move_eyes(*self._step(self._up, -1)),
            ".": lambda: self._move_eyes(*self._step(self._away, -1)),
            ",": lambda: self._move_eyes(*self._step(self._away, 1)),
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

    def _step(self, direction: tuple[float, float, float], sign: int) -> tuple[float, float, float]:
        """One step of ``step_size`` mm along a unit direction, forward (1) or back (-1)."""
        return tuple(sign * self.step_size * component for component in direction)

    def _move_target(self, dx: float, dy: float, dz: float) -> None:
        """Move target by the specified amounts."""
        self.target_point = Point3D(self.target_point.x + dx, self.target_point.y + dy, self.target_point.z + dz)

    def _move_eyes(self, dx: float, dy: float, dz: float) -> None:
        """Move all eyes by the specified amounts (simulates head movement)."""
        for eye in self.eyes:
            # Stepped from the placement: the current position also holds the shift of the last look_at.
            placement = eye.placement
            eye.position = Position3D(placement.x + dx, placement.y + dy, placement.z + dz)

    def reset_positions(self) -> None:
        """Reset all eyes and target to initial positions."""
        for eye, initial_pos in zip(self.eyes, self.initial_eye_positions, strict=True):
            eye.position = initial_pos

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
        info("CONTROLS:")
        info("Target Movement (Arrow keys):")
        info("  ↑/↓: Move target up/down")
        info("  ←/→: Move target left/right")
        info()
        info("Eye Movement (I/K/J/L/./):")
        info("  I/K: Move eye up/down")
        info("  J/L: Move eye left/right")
        info("  ./,: Move eye closer/farther from camera")
        if include_reset:
            info("Reset (Space): Reset eye and target to initial positions")
        if additional_controls:
            for description, keys in additional_controls.items():
                info(f"{description}: {keys}")
