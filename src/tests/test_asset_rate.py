"""Requests to the 3D asset server: shared client-side rate limit and no MCP event stream."""
import asyncio
from types import SimpleNamespace

import httpx
import pytest

import asset_rate
from asset_rate import RateLimiter
from research.mcp_client import EventStreamDisabled, _no_event_stream
from url_safety import public_http_client


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(round(seconds, 3))
        self.now += seconds


def test_requests_within_the_limit_do_not_wait():
    clock = FakeClock()
    limiter = RateLimiter(3, 60, clock=clock, sleep=clock.sleep)
    waits = [asyncio.run(limiter.acquire()) for _ in range(3)]
    assert waits == [0, 0, 0] and clock.sleeps == []


def test_request_over_the_limit_waits_until_the_oldest_leaves_the_window():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock=clock, sleep=clock.sleep)
    asyncio.run(limiter.acquire())
    clock.now = 10.0
    asyncio.run(limiter.acquire())
    clock.now = 20.0
    assert asyncio.run(limiter.acquire()) == 40.0  # the first request (t=0) leaves the window at t=60
    assert limiter.stats == {"requests": 3, "waits": 1, "waitedS": 40.0}


def test_parallel_requests_share_the_limit():
    clock = FakeClock()
    limiter = RateLimiter(2, 60, clock=clock, sleep=clock.sleep)

    granted = []

    async def request():
        await limiter.acquire()
        granted.append(clock.now)

    async def burst():
        await asyncio.gather(*(request() for _ in range(5)))
    asyncio.run(burst())
    granted.sort()
    assert granted[:2] == [0.0, 0.0] and all(later - earlier >= 60 for earlier, later in zip(granted, granted[2:]))


@pytest.mark.parametrize("url, throttled", [("https://3d.shep.bot/v1/search", True),
                                            ("https://cdn.polyhaven.com/thumb.png", False)])
def test_only_the_asset_server_is_throttled(monkeypatch, url, throttled):
    calls = []

    async def acquire():
        calls.append(url)
        return 0.0
    monkeypatch.setattr(asset_rate.limiter, "acquire", acquire)

    def handler(request):
        return httpx.Response(200, json={})

    async def fetch():
        async with public_http_client(transport=httpx.MockTransport(handler)) as client:
            await client.get(url)
    asyncio.run(fetch())
    assert calls == ([url] if throttled else [])


def test_mcp_event_stream_is_refused_before_sending():
    with pytest.raises(EventStreamDisabled):
        asyncio.run(_no_event_stream(SimpleNamespace(method="GET")))
    for method in ("POST", "DELETE"):  # tool calls, session start and end still go through
        asyncio.run(_no_event_stream(SimpleNamespace(method=method)))
