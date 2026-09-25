"""OversightArena (so-arena): principled experiments on scalable-oversight mechanisms."""

from so_arena.config import configure, settings
from so_arena.core import *  # noqa: F401,F403
from so_arena.core import __all__ as _core_all
from so_arena import models  # noqa: E402

__version__ = "0.1.0"
__all__ = ["configure", "settings", "models", "__version__", *_core_all]
