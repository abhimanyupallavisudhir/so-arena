"""Global library settings (cache, dry-run simulation, concurrency)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Settings(BaseModel):
    cache_dir: Path | None = None
    simulate: bool = False
    concurrency: int = 16
    # provider calls: retries after a failed attempt (None: the backend's default, which for Inspect is
    # unlimited) and a per-attempt timeout in seconds; a call that keeps failing then raises instead of stalling
    max_retries: int | None = None
    attempt_timeout: int | None = None
    # concurrent requests per model (Inspect's default is 10, which throttles large concurrent runs)
    max_connections: int | None = None


settings = Settings()


def configure(
    *,
    cache_dir: str | Path | None = None,
    simulate: bool | None = None,
    concurrency: int | None = None,
    max_retries: int | None = None,
    attempt_timeout: int | None = None,
    max_connections: int | None = None,
) -> Settings:
    """Set global options.

    Args:
        cache_dir: directory for the on-disk response cache of real model calls.
        simulate: if True, every real model is replaced by a SimulatedModel (a dry run that
            estimates token usage and cost without calling any API).
        concurrency: default maximum number of concurrent episodes.
        max_retries: retries of a failed provider call (default: unlimited, Inspect's default).
        attempt_timeout: seconds after which a provider call attempt is abandoned and retried.
        max_connections: concurrent requests per model (Inspect's default is 10).
    """
    from so_arena.models import base

    if cache_dir is not None:
        settings.cache_dir = Path(cache_dir)
    if simulate is not None:
        settings.simulate = simulate
    if concurrency is not None:
        settings.concurrency = concurrency
    if max_retries is not None:
        settings.max_retries = max_retries
    if attempt_timeout is not None:
        settings.attempt_timeout = attempt_timeout
    if max_connections is not None:
        settings.max_connections = max_connections
    base._MODEL_CACHE.clear()
    return settings
