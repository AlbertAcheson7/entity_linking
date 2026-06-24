#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?archive path required}"
ARCHIVE_DIR="$(cd "$(dirname "$ARCHIVE")" && pwd)"
ARCHIVE_NAME="$(basename "$ARCHIVE")"
(cd "$ARCHIVE_DIR" && sha256sum -c "$ARCHIVE_NAME.sha256")
echo "Archive checksum verified."
echo "After extraction run: (cd entity_linking && sha256sum -c SHA256SUMS)"
