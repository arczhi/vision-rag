#!/usr/bin/env sh
set -eu

set -- /data/*.db /data/*.sqlite /data/*.sqlite3
dbs=""
for f do
  [ -e "$f" ] || continue
  dbs="$dbs${dbs:+ }$f"
done

if [ -z "$dbs" ]; then
  echo "No SQLite database files found in /data (*.db, *.sqlite, *.sqlite3)."
  exec sleep infinity
fi

echo "Starting Datasette for: $dbs"
# Paths in this project do not contain spaces; keep shell expansion so every DB is passed separately.
exec datasette serve $dbs \
  --host 0.0.0.0 \
  --port "${SQLITE_UI_INTERNAL_PORT:-8001}" \
  --setting default_page_size "${DATASETTE_DEFAULT_PAGE_SIZE:-100}" \
  --setting sql_time_limit_ms "${DATASETTE_SQL_TIME_LIMIT_MS:-5000}"
