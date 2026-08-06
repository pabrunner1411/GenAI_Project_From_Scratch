"""Feature extraction package from Part 2."""

from .pipeline import (
    FeatureExtractionError,
    extract_document_features,
    load_example_data,
)

__all__ = [
    "FeatureExtractionError",
    "extract_document_features",
    "load_example_data",
]
