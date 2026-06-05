"""Deprecated alias for :mod:`thermochain` (renamed). Import ``thermochain`` instead."""
import sys
import warnings

import thermochain as _tc
from thermochain import *  # noqa: F401,F403  (mirrors thermochain.__all__)
from thermochain import Mooring, io, pipeline, plot  # noqa: F401

for _sub in ("io", "pipeline", "plot"):  # make `import thermodrift.io` resolve
    sys.modules[f"thermodrift.{_sub}"] = getattr(_tc, _sub)

warnings.warn(
    "thermodrift was renamed thermochain; update imports to `import thermochain`",
    DeprecationWarning,
    stacklevel=2,
)
