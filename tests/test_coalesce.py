import asyncio
import pytest

from doc2skill.ingest.coalesce import AsyncSingleFlight


@pytest.mark.asyncio
async def test_coalesce_concurrent_same_key():
    """Verify 10 concurrent calls with the same key execute the factory exactly once."""
    sf = AsyncSingleFlight()
    counter = 0

    async def factory() -> str:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0.1)
        return f"result-{counter}"

    tasks = [asyncio.create_task(sf.do("same-key", factory)) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # Factory executed only once
    assert counter == 1
    # All 10 callers received identical result
    assert len(results) == 10
    assert all(r == "result-1" for r in results)


@pytest.mark.asyncio
async def test_coalesce_distinct_keys():
    """Verify 10 concurrent calls with 10 distinct keys execute 10 times."""
    sf = AsyncSingleFlight()
    counter = 0

    async def factory() -> int:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0.05)
        return counter

    tasks = [asyncio.create_task(sf.do(f"key-{i}", factory)) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # 10 distinct calls made
    assert counter == 10
    assert len(results) == 10


@pytest.mark.asyncio
async def test_coalesce_exception_propagation():
    """Verify exceptions are propagated to all waiting concurrent callers."""
    sf = AsyncSingleFlight()
    call_count = 0

    async def failing_factory() -> str:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)
        raise ValueError("Failed to fetch")

    tasks = [
        asyncio.create_task(sf.do("fail-key", failing_factory)) for _ in range(5)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert call_count == 1
    assert len(results) == 5
    for res in results:
        assert isinstance(res, ValueError)
        assert str(res) == "Failed to fetch"


@pytest.mark.asyncio
async def test_coalesce_subsequent_call_runs_after_completion():
    """Verify subsequent call runs again after previous call has completed."""
    sf = AsyncSingleFlight()
    counter = 0

    async def factory() -> int:
        nonlocal counter
        counter += 1
        return counter

    res1 = await sf.do("key", factory)
    res2 = await sf.do("key", factory)

    assert res1 == 1
    assert res2 == 2
    assert counter == 2
