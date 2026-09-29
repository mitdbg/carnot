#!/usr/bin/env bash
# init container (codex cells only; needs CAP_NET_ADMIN): restrict the codex sandbox uid's network access.
# The rules live in the pod's network namespace, which every container shares, so they bind the runner's codex
# subprocess (and any shell it spawns) after this container exits. Packets from every other uid are untouched.
#
# The sandbox uid may only open TCP connections to 127.0.0.1 on the MCP port (the corpus tools) and the egress
# proxy port (files/egress_proxy.py, which tunnels only to the model provider). No DNS, no chroma, no S3/STS,
# no instance metadata, no Kubernetes API, no internet. Fails closed: the pod never starts a codex cell with
# the rules missing.
set -euo pipefail

UID_="${CODEX_SANDBOX_UID:?}"
MCP_PORT="${CODEX_MCP_PORT:?}"
PROXY_PORT="${CODEX_PROXY_PORT:?}"

pick() {  # the first iptables backend that works in this kernel (nft on AL2023; legacy as a fallback)
    local v4 v6
    for v4 in iptables iptables-legacy; do
        v6="${v4/iptables/ip6tables}"
        if command -v "$v4" >/dev/null && "$v4" -w -L OUTPUT -n >/dev/null 2>&1; then
            IPT="$v4"; IP6T="$v6"; return 0
        fi
    done
    echo "[lockdown] ERROR: no working iptables backend" >&2
    return 1
}
pick
echo "[lockdown] backend $IPT; uid $UID_ may reach only 127.0.0.1:$MCP_PORT (mcp) and 127.0.0.1:$PROXY_PORT (egress proxy)"

$IPT -w -N CODEX_SANDBOX 2>/dev/null || $IPT -w -F CODEX_SANDBOX
$IPT -w -A CODEX_SANDBOX -o lo -d 127.0.0.1 -p tcp --dport "$MCP_PORT" -j ACCEPT
$IPT -w -A CODEX_SANDBOX -o lo -d 127.0.0.1 -p tcp --dport "$PROXY_PORT" -j ACCEPT
# replies on connections the sandbox did not open itself never originate from it, so nothing else is needed
$IPT -w -A CODEX_SANDBOX -j REJECT
$IPT -w -C OUTPUT -m owner --uid-owner "$UID_" -j CODEX_SANDBOX 2>/dev/null \
    || $IPT -w -I OUTPUT 1 -m owner --uid-owner "$UID_" -j CODEX_SANDBOX

# IPv6: nothing at all for the sandbox uid (the MCP server and the proxy listen on 127.0.0.1). Fail closed
# without ip6tables: telling "no usable IPv6 route" apart from the loopback's unreachable default is not worth it
"$IP6T" -w -L OUTPUT -n >/dev/null 2>&1 || { echo "[lockdown] ERROR: $IP6T unusable; cannot fence IPv6" >&2; exit 1; }
$IP6T -w -C OUTPUT -m owner --uid-owner "$UID_" -j REJECT 2>/dev/null \
    || $IP6T -w -I OUTPUT 1 -m owner --uid-owner "$UID_" -j REJECT

# verify: the rule must be in place (a silent no-op would leave codex unfenced)
$IPT -w -S OUTPUT | grep -q -- "--uid-owner $UID_" || { echo "[lockdown] ERROR: OUTPUT rule missing" >&2; exit 1; }
$IPT -w -S CODEX_SANDBOX
echo "[lockdown] done"
