#!/usr/bin/env bash
set -euo pipefail

# Copy TLS material to repo + hal (paths match config/web.toml).
#
#   ./scripts/sync_web_certs.sh
#
# Env:
#   SDTD_CERT_SRC=/Users/samy/Code/cert/wildcard
#   SDTD_CERT_FULLCHAIN=fullchain1.pem
#   SDTD_CERT_PRIVKEY=privkey1.pem
#   SDTD_HAL_HOST=samy@hal
#   SDTD_HAL_ROOT_REMOTE=/home/samy/c/samysd

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CERT_SRC="${SDTD_CERT_SRC:-/Users/samy/Code/cert/wildcard}"
CERT_FULLCHAIN="${SDTD_CERT_FULLCHAIN:-fullchain1.pem}"
CERT_PRIVKEY="${SDTD_CERT_PRIVKEY:-privkey1.pem}"
HAL_HOST="${SDTD_HAL_HOST:-samy@hal}"
HAL_ROOT="${SDTD_HAL_ROOT_REMOTE:-/home/samy/c/samysd}"
CERT_NAME="$(basename "${CERT_SRC}")"
LOCAL_DEST="${LOCAL_ROOT}/certs/${CERT_NAME}"
REMOTE_DEST="${HAL_HOST}:${HAL_ROOT}/certs/${CERT_NAME}/"

for file in "${CERT_FULLCHAIN}" "${CERT_PRIVKEY}"; do
  if [[ ! -f "${CERT_SRC}/${file}" ]]; then
    echo "Missing ${CERT_SRC}/${file}" >&2
    exit 1
  fi
done

mkdir -p "${LOCAL_DEST}"
chmod 700 "${LOCAL_ROOT}/certs" "${LOCAL_DEST}" 2>/dev/null || true
cp -f "${CERT_SRC}/${CERT_FULLCHAIN}" "${LOCAL_DEST}/fullchain.pem"
cp -f "${CERT_SRC}/${CERT_PRIVKEY}" "${LOCAL_DEST}/privkey.pem"
chmod 644 "${LOCAL_DEST}/fullchain.pem"
chmod 600 "${LOCAL_DEST}/privkey.pem"

ssh "${HAL_HOST}" "mkdir -p '${HAL_ROOT}/certs/${CERT_NAME}' && chmod 700 '${HAL_ROOT}/certs' '${HAL_ROOT}/certs/${CERT_NAME}'"
rsync -avz \
  "${LOCAL_DEST}/fullchain.pem" \
  "${LOCAL_DEST}/privkey.pem" \
  "${REMOTE_DEST}"
ssh "${HAL_HOST}" "chmod 644 '${HAL_ROOT}/certs/${CERT_NAME}/fullchain.pem' && chmod 600 '${HAL_ROOT}/certs/${CERT_NAME}/privkey.pem'"

echo "Synced certs to ${LOCAL_DEST} and ${HAL_HOST}:${HAL_ROOT}/certs/${CERT_NAME}/"
