"""Domains (settings): items + verifiers + tools + ground-truth scorers.

Importing this package registers every built-in domain whose optional dependencies are installed.
"""

import importlib
import logging

from so_arena.domains.base import Domain, get_domain, list_domains, register_domain
from so_arena.domains import synthetic  # noqa: F401

_OPTIONAL = ("chess", "sql", "code", "repo", "firm", "forecasting", "qa", "lean", "controlarena")
for _mod in _OPTIONAL:
    try:
        importlib.import_module(f"so_arena.domains.{_mod}")
    except ModuleNotFoundError as e:  # missing optional dependency or module not present
        logging.getLogger("so_arena").debug("domain %s unavailable: %s", _mod, e)

__all__ = ["Domain", "get_domain", "list_domains", "register_domain"]
