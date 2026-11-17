"""Client-side throttle for the 3D asset server (3d.shep.bot).

The server allows 120 requests per 60 s per client. One furniture list easily needs more: every
item's online run opens an MCP session (several requests) and calls search/details tools, and every
download fetches details, the file selection, the model, each texture or buffer and a thumbnail.
All HTTP clients that can reach the server (MCP session, asset API, downloads, previews) send their
requests through `throttle_request`, which waits for a free slot instead of running into the limit.
Requests to other hosts are not affected. Process-wide, so parallel items and runs share the budget.
"""
import asyncio
import os
import time
from collections import deque
from urllib.parse import urlparse

from observability import progress

ASSET_HOSTS = {"3d.shep.bot", urlparse(os.environ.get("ASSET_MCP_URL", "https://3d.shep.bot/mcp")).hostname}
# Below the server's 120: the user's own asset searches in the UI and clock differences need room.
REQUESTS_PER_MINUTE = int(os.environ.get("ASSET_REQUESTS_PER_MINUTE", "100"))
WINDOW_S = 60.0


class RateLimiter:
    """Sliding window: at most `limit` requests in any `window` seconds. Single event loop, no lock
    needed: the check and the reservation happen without an await in between."""

    def __init__(self, limit: int, window: float, clock=time.monotonic, sleep=asyncio.sleep):
        self.limit, self.window, self.clock, self.sleep = limit, window, clock, sleep
        self.sent: deque[float] = deque()
        self.stats = {"requests": 0, "waits": 0, "waitedS": 0.0}

    def _expire(self, now: float) -> None:
        while self.sent and self.sent[0] <= now - self.window:
            self.sent.popleft()

    async def acquire(self) -> float:
        """Reserves a slot; returns the seconds waited for it."""
        waited = 0.0
        while True:
            now = self.clock()
            self._expire(now)
            if len(self.sent) < self.limit:
                self.sent.append(now)
                self.stats["requests"] += 1
                if waited:
                    self.stats["waits"] += 1
                    self.stats["waitedS"] = round(self.stats["waitedS"] + waited, 1)
                return waited
            delay = max(self.sent[0] + self.window - now, 0.05)
            if not waited:  # reported once per wait; outside a run progress has no writer and ignores it
                progress.step("asset_rate", f"Asset server: pausing {delay:.0f} s to stay below "
                                            f"{self.limit} requests per minute")
            await self.sleep(delay)
            waited += delay


limiter = RateLimiter(REQUESTS_PER_MINUTE, WINDOW_S)


async def throttle_request(request) -> None:
    """Request hook for httpx / httpx2 clients."""
    if request.url.host in ASSET_HOSTS:
        await limiter.acquire()
