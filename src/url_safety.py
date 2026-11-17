"""Guards for URLs the backend fetches on behalf of external data (asset files, preview images):
only public http(s) hosts, also after redirects."""
import ipaddress
from urllib.parse import urlparse

import httpx

from asset_rate import throttle_request


class UnsafeURLError(ValueError):
    pass


def check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURLError(f"Refusing to fetch non-http(s) URL: {url[:120]}")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise UnsafeURLError(f"Refusing to fetch local address: {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # a host name; resolving it here would not stop DNS rebinding anyway
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        raise UnsafeURLError(f"Refusing to fetch private address: {host}")


async def _check_request(request: httpx.Request) -> None:
    check_public_url(str(request.url))


def public_http_client(**kwargs) -> httpx.AsyncClient:
    """httpx client that checks every request, including each redirect hop. Requests to the 3D asset
    server also wait for the shared rate limit (asset_rate)."""
    return httpx.AsyncClient(follow_redirects=True, event_hooks={"request": [_check_request, throttle_request]},
                             **kwargs)
