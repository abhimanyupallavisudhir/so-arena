"""Integrations with the Inspect / ControlArena ecosystem.

- :mod:`.inspect` — run any OversightArena experiment as an Inspect ``Task`` (``inspect eval``,
  model roles, Inspect View) and load the logs back as :class:`~oversight_arena.Results`.
- :mod:`.control_arena` — ControlArena settings as OversightArena domains, reward rules on
  ControlArena eval logs (IC analysis of control protocols used as training signals), and
  OversightArena mechanisms as ControlArena micro-protocols. Requires ``control-arena``.

Submodules are imported lazily so that optional dependencies stay optional.
"""
