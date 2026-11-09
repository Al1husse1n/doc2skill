import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncSingleFlight:
    """Coalesces concurrent in-flight requests for the same key into a single execution."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._calls: dict[str, asyncio.Future[Any]] = {}

    async def do(self, key: str, factory: Callable[[], Awaitable[T]]) -> T:
        """Execute factory for key, or wait for an in-flight execution of the same key."""
        async with self._lock:
            if key in self._calls:
                future: asyncio.Future[T] = self._calls[key]
                is_leader = False
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._calls[key] = future
                is_leader = True

        if not is_leader:
            return await asyncio.shield(future)

        try:
            result = await factory()
            if not future.done():
                future.set_result(result)
            return result
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._calls.pop(key, None)
