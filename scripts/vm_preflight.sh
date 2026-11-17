#!/usr/bin/env bash
# vm_preflight.sh — Fix cpdctl config and verify CPD auth on the VM before any test runs.
# Supports s390x only.
# Usage: sshpass -p'PASS' ssh root@HOST 'bash -s' < vm_preflight.sh

set -e

CPD_URL="${CPD_URL:-}"
CPD_USERNAME="${CPD_USERNAME:-cpadmin}"
CPD_PASSWORD="${CPD_PASSWORD:-}"

echo "[preflight] === Step 1: Verify architecture ==="
ARCH=$(uname -m)
if [ "$ARCH" != "s390x" ]; then
  echo "[preflight] ERROR: This script is for s390x only. Detected: $ARCH"
  exit 1
fi
echo "[preflight] Architecture: s390x OK"

echo "[preflight] === Step 2: Check / fix cpdctl config ==="
if [ -f /root/.cpdctl/config ]; then
  if cpdctl config profile list &>/dev/null; then
    echo "[preflight] Existing config is valid — keeping it"
  else
    echo "[preflight] Existing config is corrupted — removing"
    rm -f /root/.cpdctl/config
  fi
else
  echo "[preflight] No config file found — will create"
fi

echo "[preflight] === Step 3: Verify cpdctl binary ==="
if ! command -v cpdctl &>/dev/null; then
  echo "[preflight] cpdctl not found — installing s390x build..."
  cd /tmp && curl -sL https://github.com/IBM/cpdctl/releases/latest/download/cpdctl_linux_s390x.tar.gz -o cpdctl.tar.gz \
    && tar -zxf cpdctl.tar.gz && mv cpdctl /usr/local/bin/ && rm cpdctl.tar.gz
fi
cpdctl version
echo "[preflight] cpdctl OK"

if [ -z "$CPD_URL" ] || [ -z "$CPD_PASSWORD" ]; then
  echo "[preflight] WARN: CPD_URL or CPD_PASSWORD not set — skipping auth setup"
  exit 0
fi

echo "[preflight] === Step 4: Configure cpdctl profile ==="
if cpdctl config profile current &>/dev/null; then
  echo "[preflight] Profile ds-profile already active — skipping configure"
else
  cpdctl config profile set ds-profile \
    --url "$CPD_URL" \
    --username "$CPD_USERNAME" \
    --password "$CPD_PASSWORD"
  cpdctl config profile use ds-profile
  echo "[preflight] Profile set to ds-profile"
fi

echo "[preflight] === Step 5: Test authentication ==="
if cpdctl project list --output json &>/dev/null; then
  echo "[preflight] Auth OK — CPD is reachable"
else
  echo "[preflight] WARN: cpdctl project list failed — check credentials or CPD availability"
  exit 1
fi

echo "[preflight] === DONE — s390x VM is ready for testing ==="
