"""Backward-compatible module.

The homework now uses supervisor.py as the top-level coordinator.
This file is kept so older imports do not break.
"""

from supervisor import supervisor_agent as agent  # noqa: F401
from supervisor import new_thread_id as new_session  # noqa: F401
