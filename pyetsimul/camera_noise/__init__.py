"""Camera noise simulation for eye tracking experiments."""

from .glint_noise import apply_glint_noise, GlintNoiseConfig

__all__ = [
    "apply_glint_noise",
    "GlintNoiseConfig",
]
