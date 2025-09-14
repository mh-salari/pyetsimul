"""Camera noise simulation for eye tracking experiments."""

from .glint_noise import GlintNoiseConfig, apply_glint_noise

__all__ = [
    "GlintNoiseConfig",
    "apply_glint_noise",
]
