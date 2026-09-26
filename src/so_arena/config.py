"""Global library settings (cache, dry-run simulation, concurrency)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    cache_dir: Path | None = None
    simulate: bool = False
    concurrency: int = 16


settings = Settings()


def configure(
    *,
    cache_dir: str | Path | None = None,
    simulate: bool | None = None,
    concurrency: int | None = None,
) -> Settings:
    """Set global options.

    Args:
        cache_dir: directory for the on-disk response cache of real model calls.
        simulate: if True, every real model is replaced by a SimulatedModel (a dry run that
            estimates token usage and cost without calling any API).
        concurrency: default maximum number of concurrent episodes.
    """
    from so_arena.models import base

    if cache_dir is not None:
        settings.cache_dir = Path(cache_dir)
    if simulate is not None:
        settings.simulate = simulate
    if concurrency is not None:
        settings.concurrency = concurrency
    base._MODEL_CACHE.clear()
    return settings
