"""Allowlisting HTTPS egress proxy for the codex sandbox user (codex cells only; started by run.sh as root).

The pod's iptables rules (lockdown.sh) let the sandbox uid open TCP connections to exactly two loopback
ports: the MCP server and this proxy. Everything codex (and any shell command it runs) sends off the pod
therefore has to go through here, and here only `CONNECT <allowed host>:443` is tunneled: the model
provider. Everything else (S3 / STS with a stolen token, GitHub / Hugging Face copies of the benchmark,
the instance metadata endpoint, the chroma server) is refused and logged, so an attempt is visible in
the runner's log even though it failed.

    python3 egress_proxy.py --port 3128 --allow openrouter.ai
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys

MAX_HEADER_BYTES = 16384


def _log(msg: str) -> None:
    print(f"[egress {datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}] {msg}", file=sys.stderr, flush=True)


def allowed(host: str, port: int, allow: set[str]) -> bool:
    host = host.lower().rstrip(".")
    return port == 443 and any(host == a or host.endswith("." + a) for a in allow)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, allow: set[str]) -> None:
    peer = writer.get_extra_info("peername")
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
        writer.close()
        return
    request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
    parts = request_line.split()
    if len(parts) != 3 or parts[0].upper() != "CONNECT":
        # plain-HTTP proxying is never needed (the provider is HTTPS-only), so it is refused outright
        _log(f"DENY {peer} {request_line[:200]!r} (only CONNECT is proxied)")
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    host, _, port_s = parts[1].rpartition(":")
    try:
        port = int(port_s)
    except ValueError:
        host, port = parts[1], -1
    host = host.strip("[]")
    if not allowed(host, port, allow):
        _log(f"DENY {peer} CONNECT {host}:{port}")
        writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    try:
        up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=30)
    except (OSError, asyncio.TimeoutError) as e:
        _log(f"FAIL {peer} CONNECT {host}:{port}: {type(e).__name__}: {e}")
        writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        return
    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await writer.drain()
    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer))


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=3128)
    p.add_argument("--allow", nargs="+", required=True, help="hostnames (and their subdomains) that may be tunneled on :443")
    args = p.parse_args()
    allow = {a.lower().rstrip(".") for a in args.allow}
    server = await asyncio.start_server(lambda r, w: handle(r, w, allow), args.host, args.port, limit=MAX_HEADER_BYTES)
    _log(f"listening on {args.host}:{args.port}; allow {sorted(allow)} on :443")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
