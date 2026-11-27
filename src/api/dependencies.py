"""Lazy-load classifiers for API routes."""

from dynamic_inference import DEFAULT_METHOD, load_dynamic_bundle
from static_inference import STATIC_MODEL_PATH, load_static_model

_static_model = None
_dynamic_bundle = None


def get_static_model():
    """Return the cached static RF, loading it on first use."""
    global _static_model
    if _static_model is None:
        if not STATIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing {STATIC_MODEL_PATH}. Train or install a classifier first."
            )
        _static_model = load_static_model()
    return _static_model


def get_dynamic_bundle():
    """Return the cached J/Z model bundle, loading it on first use."""
    global _dynamic_bundle
    if _dynamic_bundle is None:
        _dynamic_bundle = load_dynamic_bundle(DEFAULT_METHOD, jz_only=True)
    return _dynamic_bundle
