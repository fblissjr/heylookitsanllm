#!/usr/bin/env python3
"""Copy a conversation store into a fresh one on the CURRENT schema.

    uv run python scripts/migrate_conversations.py --from data/conversations.duckdb

WHY THIS EXISTS. `db.Store` recreates on version mismatch rather than
migrating -- a deliberate policy for a fresh-start store (CLAUDE.md: "NEVER
write migration code"). The consequence is that opening an older store with
newer code DROPS `conversations`, `messages`, `media_blobs` and `notebooks`
with no prompt and no backup, on the next server start. That is fine for a
store you do not mind losing and a disaster for one you do. This is the escape
hatch: run it BEFORE starting a server on newer code, and keep the result.

It is not migration code in the app. It is a one-off, run by hand, that the
app knows nothing about -- which is the distinction the policy is drawing.

WHAT IT WILL NOT DO:

  * It never writes to the source. Opened read-only, always.
  * It never overwrites the destination. Refuses if the path exists.
  * It never prints stored content -- titles, prompts, message text, blobs.
    Counts and column names only. Your conversations are not this script's
    business and should not end up in a terminal scrollback or a log.

HOW IT MAPS. Columns are matched BY NAME against the current schema, which is
imported from `heylook_llm.db._SCHEMA_SQL` rather than restated here -- so this
script cannot drift from the app's idea of the schema. The result contains the
current schema and nothing else: a column the old store had and the new one
does not is NOT carried over.

RENAMES ARE THE DANGEROUS CASE, and the reason this script fails closed. A
renamed column looks exactly like a dropped one plus a defaulted new one, and
nothing can tell the two apart from the schemas alone -- so a name-matching
copy would silently throw the data away. When the diff has that shape (old
columns with no home AND new columns being defaulted) this refuses to write
until you say which it is:

    --rename messages.old_name=new_name     # carry the data across
    --accept-drops                          # no, they really are gone

ALTER TABLE INSTEAD? For a pure rename, `ALTER TABLE ... RENAME COLUMN`
in place is simpler and lossless, and needs no copy -- just remember to bump
the `schema_meta` version row afterwards or the app will drop the tables on
open anyway. The tradeoff is that it mutates the original. This script copies
because that leaves you something to go back to.

THE SOURCE MUST NOT BE IN USE. DuckDB takes a lock per file; a running
heylookllm holds it. Stop the server first. This script will say so rather
than fail cryptically -- and will not try to work around it, because a torn
read of a live database is exactly the failure it exists to prevent.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from heylook_llm.db import _SCHEMA_SQL, _SCHEMA_VERSION  # noqa: E402

# Every table the current schema defines. Order matters for nothing here (no
# FKs are declared), but conversations-before-messages reads better in output.
TABLES = ["conversations", "messages", "media_blobs", "notebooks",
          "presets", "settings"]


def columns_of(con, table: str) -> list[str]:
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = ? ORDER BY ordinal_position", (table,)
    ).fetchall()
    return [r[0] for r in rows]


def open_source(path: Path):
    try:
        return duckdb.connect(str(path), read_only=True)
    except duckdb.IOException as e:
        if "lock" in str(e).lower():
            sys.exit(
                f"\n{path} is in use -- a running heylookllm holds DuckDB's file lock.\n"
                "Stop the server and run this again. This script will not copy a\n"
                "database that is being written to; a torn snapshot is worse than\n"
                "no snapshot.\n"
            )
        raise


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True, type=Path,
                    help="existing store (opened READ-ONLY, never modified)")
    ap.add_argument("--to", dest="dst", type=Path,
                    help="destination (default: <source>.migrated.duckdb)")
    ap.add_argument("--rename", action="append", default=[], metavar="[TABLE.]OLD=NEW",
                    help="carry a renamed column across; repeatable")
    ap.add_argument("--accept-drops", action="store_true",
                    help="confirm that unmatched old columns really are gone")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the column diff and row counts, write nothing")
    args = ap.parse_args()

    # {(table|None, old): new}
    renames: dict[tuple[str | None, str], str] = {}
    for spec in args.rename:
        if "=" not in spec:
            sys.exit(f"--rename needs OLD=NEW, got: {spec}")
        lhs, new = spec.split("=", 1)
        table, _, old = lhs.rpartition(".")
        renames[(table or None, old)] = new

    src: Path = args.src
    if not src.exists():
        sys.exit(f"no such file: {src}")
    dst: Path = args.dst or src.with_suffix(".migrated.duckdb")
    if dst.exists() and not args.dry_run:
        sys.exit(f"destination already exists, refusing to overwrite: {dst}")

    scon = open_source(src)
    src_version = "unknown"
    try:
        row = scon.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
        if row:
            src_version = row[0]
    except duckdb.Error:
        pass
    src_tables = {r[0] for r in scon.execute("SHOW TABLES").fetchall()}

    print(f"source      {src}  (schema v{src_version})")
    print(f"target      schema v{_SCHEMA_VERSION}")
    if src_version == str(_SCHEMA_VERSION):
        print("\nThe source is ALREADY on the current schema. Nothing would be "
              "dropped by\nopening it with this code; a migration is not needed.")

    # Build the destination from the app's own schema, then diff against it.
    scratch = Path(str(dst) + ".partial")
    if scratch.exists():
        scratch.unlink()
    dcon = duckdb.connect(":memory:" if args.dry_run else str(scratch))
    for stmt in _SCHEMA_SQL.split(";\n\n"):
        if stmt.strip():
            dcon.execute(stmt)

    # Pass 1: diff every table before writing anything, so an ambiguity in the
    # LAST table still stops the FIRST from being written.
    plans = []
    ambiguous = []
    for table in TABLES:
        new_cols = columns_of(dcon, table)
        if table not in src_tables:
            plans.append((table, None, [], [], []))
            continue
        old_cols = columns_of(scon, table)

        # old column -> destination column. A declared rename redirects it;
        # otherwise it maps to itself if the new schema still has that name.
        mapping: dict[str, str] = {}
        for c in old_cols:
            target = renames.get((table, c)) or renames.get((None, c))
            if target:
                if target not in new_cols:
                    sys.exit(f"--rename {table}.{c}={target}: "
                             f"'{target}' is not a column of the new {table}")
                mapping[c] = target
            elif c in new_cols:
                mapping[c] = c

        carried = list(mapping.values())
        gained = [c for c in new_cols if c not in carried]
        dropped = [c for c in old_cols if c not in mapping]
        plans.append((table, mapping, gained, dropped, old_cols))
        # The rename shape: something lost AND something defaulted.
        if dropped and gained:
            ambiguous.append((table, dropped, gained))

    if ambiguous and not (args.accept_drops or args.dry_run):
        print("\nREFUSING TO WRITE -- this diff has the shape of a rename.\n")
        for table, dropped, gained in ambiguous:
            print(f"  {table}: losing {dropped}, defaulting {gained}")
        print("\nNothing distinguishes a renamed column from a dropped one plus a new\n"
              "one, so writing now could throw data away silently. Say which it is:\n"
              "  --rename TABLE.OLD=NEW   carry it across\n"
              "  --accept-drops           they really are gone\n"
              "  --dry-run                just show me the diff\n")
        dcon.close()
        scon.close()
        # Leave nothing behind: a half-built .partial from a refused run is a
        # file the next person has to reason about.
        if scratch.exists():
            scratch.unlink()
        return 2

    # Pass 2: copy.
    print()
    total = 0
    for table, mapping, gained, dropped, _old in plans:
        if mapping is None:
            print(f"  {table:<14} absent in source -- created empty")
            continue
        n = scon.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if n and not args.dry_run and mapping:
            src_cols = list(mapping.keys())
            dst_cols = [mapping[c] for c in src_cols]
            select = ", ".join(f'"{c}"' for c in src_cols)
            insert = ", ".join(f'"{c}"' for c in dst_cols)
            rows = scon.execute(f"SELECT {select} FROM {table}").fetchall()
            dcon.executemany(
                f"INSERT INTO {table} ({insert}) VALUES "
                f"({', '.join('?' for _ in dst_cols)})", rows)
        total += n

        note = []
        moved = [f"{o}->{n2}" for o, n2 in (mapping or {}).items() if o != n2]
        if moved:
            note.append(f"renamed {','.join(moved)}")
        if gained:
            note.append(f"+{','.join(gained)} (defaulted)")
        if dropped:
            note.append(f"-{','.join(dropped)} (dropped)")
        print(f"  {table:<14} {n:>7} rows   {'  '.join(note)}")

    if not args.dry_run:
        dcon.execute("DELETE FROM schema_meta WHERE key = 'version'")
        dcon.execute("INSERT INTO schema_meta (key, value) VALUES ('version', ?)",
                     (str(_SCHEMA_VERSION),))
    dcon.close()
    scon.close()

    print(f"\n  {total} rows total")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    shutil.move(str(scratch), str(dst))
    print(f"\nwrote  {dst}")
    print(f"The source is untouched. To adopt the new store, point\n"
          f"HEYLOOK_DB_PATH at it, or move it into place yourself -- this script\n"
          f"does not move or delete anything of yours.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
