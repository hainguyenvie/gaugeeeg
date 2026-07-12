"""GaugeEEG: reference-shift robustness for EEG representations."""

from .referencing import common_average, reference_matrix, single_reference

__all__ = ["common_average", "reference_matrix", "single_reference"]
__version__ = "0.1.0"
