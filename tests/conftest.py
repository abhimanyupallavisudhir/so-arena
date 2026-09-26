"""Test-suite settings."""

import os

from so_arena.core import sandbox

# Where no sandbox backend works, the suite still exercises agent commands and untrusted code, unsandboxed
# (the sandbox's own tests are skipped there); elsewhere every such command runs sandboxed, as in use.
if not sandbox.available():
    os.environ.setdefault(sandbox.ALLOW_ENV, "1")
