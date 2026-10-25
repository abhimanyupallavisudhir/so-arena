"""Domains (≈ ControlArena settings): task sources with their tools, verifiers, information
access, environments and ground truth. Import specific domains from their modules (some need
optional dependencies, e.g. ``python-chess``); list them with ``oversight-arena list domain``."""

from .base import Domain, Environment, TaskListDomain, default_render

__all__ = ["Domain", "Environment", "TaskListDomain", "default_render"]
