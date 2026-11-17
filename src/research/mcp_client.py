"""Client for the 3D Asset Server's MCP endpoint (tools: search_assets, get_asset, list_providers).

One MCP session per research run:

    async with AssetMCP() as mcp:
        results = await mcp.search("wooden cabinet")
        details = await mcp.get_asset(results["results"][0]["id"])

The MCP tools only describe assets; downloading happens later in the backend (inventory),
and only after the user approved it.
"""
import asyncio
import json
import os
from contextlib import AsyncExitStack

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client

from asset_rate import throttle_request

MCP_URL = os.environ.get("ASSET_MCP_URL", "https://3d.shep.bot/mcp")
CALL_TIMEOUT_S = 30.0


class AssetMCPError(Exception):
    pass


class EventStreamDisabled(Exception):
    pass


async def _no_event_stream(request) -> None:
    """Refuses the optional GET stream for server-initiated messages before it is sent. The asset
    server closes that stream at once and the MCP client reconnects every second while the session
    is open (about 60 requests per minute per session, measured), which alone exhausted the server's
    rate limit. Tool results arrive in the responses to their POST requests, so nothing is lost; the
    client stops trying after two refused attempts."""
    if request.method == "GET":
        raise EventStreamDisabled("The GET event stream is not used.")


def parse_tool_result(result) -> dict:
    """The server answers with one JSON text block; errors come back as is_error results."""
    text = "\n".join(getattr(c, "text", "") for c in result.content or [])
    if result.is_error:
        raise AssetMCPError(text.strip() or "The asset server reported an error.")
    if getattr(result, "structured_content", None):
        return result.structured_content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise AssetMCPError("The asset server returned an unreadable answer.")


class AssetMCP:
    def __init__(self, url: str = MCP_URL, timeout_s: float = CALL_TIMEOUT_S):
        self.url = url
        self.timeout_s = timeout_s
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "AssetMCP":
        self._stack = AsyncExitStack()
        try:
            # MCP's own client (its timeouts), without the GET event stream and with the shared rate limit
            # of the asset server on every request.
            http = await self._stack.enter_async_context(create_mcp_http_client())
            http.event_hooks = {**http.event_hooks,
                                "request": [*http.event_hooks.get("request", []), _no_event_stream, throttle_request]}
            streams = await self._stack.enter_async_context(streamable_http_client(self.url, http_client=http))
            self._session = await self._stack.enter_async_context(ClientSession(streams[0], streams[1]))
            with anyio.fail_after(self.timeout_s):
                await self._session.initialize()
        except BaseException as exc:  # anyio reports connection problems as exception groups
            try:
                await self._stack.aclose()
            except BaseException:
                pass
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            task = asyncio.current_task()
            if task is not None and task.cancelling():  # cancellation wrapped in an exception group
                raise asyncio.CancelledError() from exc
            raise AssetMCPError(f"MCP server {self.url} is not reachable ({_first_error(exc)}).") from None
        return self

    async def __aexit__(self, *exc_info):
        try:
            await self._stack.aclose()
        except BaseException:
            pass  # closing a broken session must not hide the original result

    async def call(self, tool: str, arguments: dict) -> dict:
        try:
            result = await self._session.call_tool(tool, arguments, read_timeout_seconds=self.timeout_s)
        except AssetMCPError:
            raise
        except Exception as exc:
            raise AssetMCPError(f"MCP call {tool} failed ({_first_error(exc)}).") from None
        return parse_tool_result(result)

    async def search(self, query: str, limit: int = 12) -> dict:
        # Free, directly downloadable 3D models only: everything else could not be used in the room anyway.
        return await self.call("search_assets", {"query": query, "types": ["model"], "free_only": True,
                                                 "downloadable_only": True, "limit": limit})

    async def get_asset(self, asset_id: str) -> dict:
        return await self.call("get_asset", {"id": asset_id, "format": "glb"})

    async def list_providers(self) -> dict:
        return await self.call("list_providers", {})


def _first_error(exc: BaseException) -> str:
    """The most telling error inside (nested) exception groups, skipping cancellations."""
    def leaves(e):
        if isinstance(e, BaseExceptionGroup):
            for inner in e.exceptions:
                yield from leaves(inner)
        else:
            yield e
    errors = list(leaves(exc))
    chosen = next((e for e in errors if not isinstance(e, (anyio.get_cancelled_exc_class(), TimeoutError))),
                  errors[0] if errors else exc)
    if isinstance(chosen, TimeoutError) or isinstance(exc, TimeoutError):
        return "timeout"
    return f"{type(chosen).__name__}: {chosen}"[:200]
