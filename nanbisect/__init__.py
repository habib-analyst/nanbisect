"""nanbisect — git bisect for your training data.

When a retrained model gets worse, which data slice caused it? nanbisect
binary-searches over ordered training-data chunks (data commits): train on a
prefix, measure a behavior metric on a probe set, halve the range, repeat —
until the culprit chunk is isolated.
"""

__version__ = "0.1.0"
__all__ = ["Bisector", "BisectResult", "BisectStep"]

from .core import Bisector, BisectResult, BisectStep
