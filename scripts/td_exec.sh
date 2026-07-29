#!/usr/bin/env bash
# Run a TouchDesigner build script via touchmcp (http://127.0.0.1:8080 by default).
set -euo pipefail

SCRIPT="${1:?usage: td_exec.sh /path/to/build_script.py}"
PORT="${TD_MCP_PORT:-9981}"
URL="http://127.0.0.1:${PORT}/api/td/server/exec"

if ! curl -sf -m 2 "${URL%/exec}/td" >/dev/null; then
  echo "TouchDesigner MCP not reachable at ${URL} (start TD + mcp_webserver_base)" >&2
  exit 2
fi

payload=$(python3 -c 'import json,sys; print(json.dumps({"script": open(sys.argv[1], encoding="utf-8").read()}))' "$SCRIPT")
resp=$(curl -sf -m 120 -X POST "$URL" -H 'Content-Type: application/json' -d "$payload")
echo "$resp" | python3 -c '
import json, sys
r = json.load(sys.stdin)
if not r.get("success"):
    print(r.get("error") or r, file=sys.stderr)
    sys.exit(1)
data = r.get("data") or {}
stdout = data.get("stdout") or ""
stderr = data.get("stderr") or ""
if stdout:
    print(stdout, end="" if stdout.endswith("\n") else "\n")
if stderr:
    print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
'
