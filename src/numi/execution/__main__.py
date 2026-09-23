"""Windows-safe direct launcher for the Execution Service.

Two separate problems, both Windows-only:

1. `uvicorn numi.execution.api.app:app` (the Dockerfile/Makefile
   invocation) starts its event loop via `asyncio.run()` *before* it
   imports the app module, so the policy fix in `execution/api/app.py`
   lands too late to help psycopg's async connections.
2. Even called before that, `asyncio.set_event_loop_policy(...)` alone
   isn't enough: uvicorn >= 0.36 ignores the global policy and passes its
   own `loop_factory` to `asyncio.run()`, which for `loop="asyncio"`/`"auto"`
   explicitly returns `asyncio.ProactorEventLoop` on Windows
   (`uvicorn.loops.asyncio.asyncio_loop_factory`) — that factory wins over
   any policy. Passing `loop="none"` makes uvicorn pass no factory at all,
   falling back to whatever the current policy says — which is where
   setting the policy first actually takes effect.

Run `python -m numi.execution` instead of the raw uvicorn CLI on Windows.
Linux/macOS containers are unaffected either way (ProactorEventLoop is a
Windows-only default), so the Dockerfile keeps using the plain `uvicorn
module:app` form.
"""

from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402 — must follow the policy fix above

from numi.common.config import get_settings  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "numi.execution.api.app:app",
        host="0.0.0.0",
        port=settings.execution_port,
        log_level=settings.log_level.lower(),
        loop="none" if sys.platform == "win32" else "auto",
    )
