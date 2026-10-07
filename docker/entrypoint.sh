#!/bin/sh
# Seed the knowledge base with sample notes, then hand off to the main command.
# Ingestion is incremental, so restarts only re-embed notes that changed.
set -e

if [ -n "$SEED_DIR" ] && [ -d "$SEED_DIR" ]; then
    echo "Seeding knowledge base from $SEED_DIR..."
    gnosis add "$SEED_DIR" || echo "Warning: seeding failed; continuing without sample notes." >&2
fi

exec "$@"
