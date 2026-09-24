#!/usr/bin/env bash
# LEGACY fallback — not needed when the mlflow image is built with APPUSER_UID equal
# to the host uid (default 1000; see Dockerfile.mlflow). The ACL is dropped whenever
# Claude rewrites .credentials.json, so it is not a durable fix.
# Grant the MLflow container appuser (uid 999) read access to host Claude auth.
# Only relevant for an image built with the old uid 999.
set -euo pipefail

CLAUDE_HOME="${CLAUDE_HOME:-${HOME}/.claude}"
APPUSER_UID="${APPUSER_UID:-999}"

if [[ ! -d "${CLAUDE_HOME}" ]]; then
  echo "FAIL: missing ${CLAUDE_HOME} — run Claude Code login on the host first." >&2
  exit 1
fi

if [[ ! -f "${CLAUDE_HOME}/.credentials.json" ]]; then
  echo "FAIL: missing ${CLAUDE_HOME}/.credentials.json — authenticate Claude on the host first." >&2
  exit 1
fi

if ! command -v setfacl >/dev/null 2>&1; then
  echo "FAIL: setfacl not found. Install acl (e.g. sudo apt install acl)." >&2
  exit 1
fi

# Only auth paths the host owns. Do NOT recurse into cache/projects: the
# container (uid 999) writes session files there on the bind mount, and
# setfacl by uid 1000 on those files fails with "Operation not permitted".
setfacl -m "u:${APPUSER_UID}:rx" "${CLAUDE_HOME}"
setfacl -m "u:${APPUSER_UID}:r" "${CLAUDE_HOME}/.credentials.json"

echo "OK: ACL granted for uid ${APPUSER_UID} on ${CLAUDE_HOME}/.credentials.json"
getfacl -p "${CLAUDE_HOME}/.credentials.json" | sed -n '1,20p'
echo "Note: ignore older recursive ACL noise on cache/projects — not required for login."
