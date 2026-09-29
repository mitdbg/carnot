"""Fail-closed self-check of the codex sandbox, run by run.sh as root before a codex cell starts.

Forks a child that drops to the sandbox uid (exactly as the runner launches codex: no supplementary groups)
and verifies from its side that it can read none of the runner's data and reach nothing but the egress proxy,
and that the proxy tunnels to the model provider but refuses anything else. Exit 0 only if every check holds.

    python3 sandbox_check.py --uid 10001 --proxy-port 3128 --mcp-port 8765
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

# paths the sandbox uid must not be able to read (listing a dir or opening a file)
FORBIDDEN_PATHS = ["/data", "/data/benchmarks", "/data/chromadb", "/results", "/proc/1/environ"]
# (host, port) the sandbox uid must not be able to connect to directly
FORBIDDEN_CONNECTS = [
    ("169.254.169.254", 80),   # instance metadata (node role credentials)
    ("1.1.1.1", 443),          # the internet
    ("127.0.0.1", 8001),       # the chroma server (the raw corpus, every collection)
    ("::1", 3128),             # IPv6 loopback (fenced entirely)
]


def _connect(host: str, port: int, timeout: float = 5.0) -> str | None:
    """None if the connection opened, else the error."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return None
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    finally:
        s.close()


def _proxy_connect(proxy_port: int, target: str) -> str:
    """The proxy's status line for CONNECT `target`."""
    s = socket.create_connection(("127.0.0.1", proxy_port), timeout=30)
    try:
        s.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        return s.recv(4096).split(b"\r\n", 1)[0].decode("latin-1")
    finally:
        s.close()


def checks(proxy_port: int, allow_target: str) -> list[str]:
    failures: list[str] = []
    for path in FORBIDDEN_PATHS:
        try:
            if os.path.isdir(path):
                os.listdir(path)
            else:
                open(path, "rb").close()
            failures.append(f"sandbox uid can read {path}")
        except PermissionError:
            pass
        except FileNotFoundError:
            # a path that does not exist is not readable either (e.g. /data before a store pull)
            pass
    for host, port in FORBIDDEN_CONNECTS:
        if _connect(host, port) is None:
            failures.append(f"sandbox uid can connect to {host}:{port}")
    try:
        denied = _proxy_connect(proxy_port, "example.com:443")
        if " 403 " not in f"{denied} ":
            failures.append(f"proxy did not refuse example.com:443 ({denied!r})")
        ok = _proxy_connect(proxy_port, allow_target)
        if " 200 " not in f"{ok} ":
            failures.append(f"proxy did not tunnel to {allow_target} ({ok!r})")
    except OSError as e:
        failures.append(f"sandbox uid cannot reach the egress proxy on 127.0.0.1:{proxy_port}: {e}")
    return failures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--proxy-port", type=int, required=True)
    p.add_argument("--mcp-port", type=int, required=True)  # informational: nothing listens there until the runner starts
    p.add_argument("--allow-target", default="openrouter.ai:443")
    args = p.parse_args()
    if os.geteuid() != 0:
        print("[sandbox-check] must run as root (it drops to the sandbox uid itself)", file=sys.stderr)
        return 2

    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        os.setgroups([])
        os.setgid(args.uid)
        os.setuid(args.uid)
        failures = checks(args.proxy_port, args.allow_target)
        os.write(w, "\n".join(failures).encode())
        os._exit(0)
    os.close(w)
    out = b""
    while chunk := os.read(r, 65536):
        out += chunk
    os.waitpid(pid, 0)
    failures = [f for f in out.decode().splitlines() if f]
    for f in failures:
        print(f"[sandbox-check] FAIL: {f}", file=sys.stderr)
    if not failures:
        print(f"[sandbox-check] ok: uid {args.uid} reads none of {FORBIDDEN_PATHS}, reaches none of "
              f"{[f'{h}:{p}' for h, p in FORBIDDEN_CONNECTS]}, and the proxy tunnels only to {args.allow_target}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
