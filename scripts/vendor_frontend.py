#!/usr/bin/env python3
"""Vendored frontend dependency manager: install, verify, and report staleness.

The v3 frontend has no build step, so its two external libraries are committed
files rather than a lockfile entry -- which means nothing could tell you they
had drifted. Answering "are we current?" used to require reading a version
banner out of a minified file and hand-checking npm. This script is that
answer, and the manifest beside the files is the pinning record.

THREE MODES, and the split is deliberate:

  --verify   OFFLINE. Do the committed files match the manifest? Two file
             reads and a regex, no network, so it can run in a pre-commit
             hook without breaking a commit made on a plane. A mismatch is a
             HARD failure: the files and the record must never disagree.
  --check    --verify plus "what is the latest published version?". Needs the
             network, so staleness is REPORTED, never fatal -- an npm outage
             must not read as a broken repo. This is the release-time mode.
  --update   Download, verify the banner, install, rewrite the manifest.

Stdlib only, on purpose: a pre-commit hook must not depend on the project
venv being synced, and this has to keep working while dependencies are exactly
the thing in flux.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parent.parent
# One constant, so the step-4 move of the frontend tree is a one-line edit
# here rather than a hunt. Kept in sync with api.py's own frontend path.
VENDOR_DIR = REPO_ROOT / "apps" / "heylook-frontend-v3" / "js" / "vendor"
MANIFEST = VENDOR_DIR / "vendor.json"
REGISTRY = "https://registry.npmjs.org"
# Enough of the file to hold the licence banner without reading a 133KB module.
BANNER_BYTES = 8192


def load_manifest() -> dict:
    if not MANIFEST.exists():
        die(f"no manifest at {rel(MANIFEST)} -- run with --update to create it")
    return json.loads(MANIFEST.read_text())


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def die(msg: str) -> NoReturn:
    print(f"vendor_frontend: {msg}", file=sys.stderr)
    sys.exit(1)


def banner_version(head: str, pattern: str) -> str | None:
    """The version the FILE claims, read from its licence banner.

    The banner is the only self-description a minified bundle carries, which
    makes it the only thing that can contradict the manifest.
    """
    match = re.search(pattern, head)
    return match.group(1) if match else None


def read_head(pkg: str, entry: dict, staged: bool) -> str:
    """File head from the working tree, or from the STAGED blob.

    Staged is what the hook wants: git's own guards here judge what is being
    committed, not what happens to be lying in the working tree, so a dirty
    checkout cannot block an unrelated commit.
    """
    path = VENDOR_DIR / entry["dest"]
    if not staged:
        if not path.exists():
            die(f"{pkg}: {rel(path)} is missing")
        return path.read_text(errors="replace")[:BANNER_BYTES]
    proc = subprocess.run(
        ["git", "show", f":{rel(path)}"],
        capture_output=True, cwd=REPO_ROOT,
    )
    if proc.returncode != 0:
        # Not staged and not in the index -- nothing being committed to check.
        return ""
    return proc.stdout.decode(errors="replace")[:BANNER_BYTES]


def verify(manifest: dict, staged: bool = False) -> bool:
    ok = True
    for pkg, entry in manifest["packages"].items():
        head = read_head(pkg, entry, staged)
        if not head:
            continue
        want = entry["version"]
        got = banner_version(head, entry["banner_re"])
        if got is None:
            print(f"  {pkg}: FAIL -- no version banner in {entry['dest']}")
            ok = False
        elif got != want:
            print(f"  {pkg}: FAIL -- file says {got}, manifest says {want}")
            ok = False
        else:
            print(f"  {pkg}: {got} (matches manifest)")
    return ok


def latest(pkg: str) -> str | None:
    try:
        with urllib.request.urlopen(f"{REGISTRY}/{pkg}/latest", timeout=15) as resp:
            return json.load(resp)["version"]
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        # Staleness is INFORMATION. A registry that will not answer is not a
        # broken repo, so this degrades to a note rather than a failure.
        print(f"  {pkg}: could not reach the registry ({exc.__class__.__name__})")
        return None


def fetch_file(pkg: str, version: str, member: str) -> bytes:
    url = f"{REGISTRY}/{pkg}/-/{pkg}-{version}.tgz"
    with urllib.request.urlopen(url, timeout=60) as resp:
        blob = resp.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        # npm tarballs put everything under "package/".
        extracted = tar.extractfile(f"package/{member}")
        if extracted is None:
            die(f"{pkg} {version}: no {member} inside the tarball")
        return extracted.read()


def update(manifest: dict, only: str | None) -> None:
    for pkg, entry in manifest["packages"].items():
        if only and pkg != only:
            continue
        newest = latest(pkg)
        if newest is None:
            continue
        if newest == entry["version"]:
            print(f"  {pkg}: already at {newest}")
            continue
        print(f"  {pkg}: {entry['version']} -> {newest}")
        content = fetch_file(pkg, newest, entry["source"])
        head = content[:BANNER_BYTES].decode(errors="replace")
        got = banner_version(head, entry["banner_re"])
        if got != newest:
            # The registry said one thing and the artifact says another. Do
            # not install a file that cannot identify itself.
            die(f"{pkg}: downloaded {newest} but its banner says {got!r}")
        (VENDOR_DIR / entry["dest"]).write_bytes(content)
        entry["version"] = newest
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def check(manifest: dict) -> None:
    print("Vendored frontend libraries:")
    integrity = verify(manifest)
    print("\nLatest published:")
    behind = []
    for pkg, entry in manifest["packages"].items():
        newest = latest(pkg)
        if newest is None:
            continue
        if newest == entry["version"]:
            print(f"  {pkg}: {newest} (current)")
        else:
            print(f"  {pkg}: {newest} available, vendored {entry['version']}")
            behind.append(pkg)
    if behind:
        print(f"\nBehind: {', '.join(behind)}. Update with: {rel(Path(__file__))} --update")
    if not integrity:
        # Only the offline half can fail the command.
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install, verify and report on the vendored frontend libraries.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true",
                      help="offline: do the files match the manifest? (hook mode)")
    mode.add_argument("--check", action="store_true",
                      help="verify, then report what npm has published")
    mode.add_argument("--update", action="store_true",
                      help="download and install the latest, rewrite the manifest")
    parser.add_argument("--staged", action="store_true",
                        help="with --verify, read staged blobs instead of the working tree")
    parser.add_argument("--package", help="limit --update to one package")
    args = parser.parse_args()

    manifest = load_manifest()
    if args.update:
        update(manifest, args.package)
    elif args.check:
        check(manifest)
    else:
        if not verify(manifest, staged=args.staged):
            print("\nA vendored file disagrees with vendor.json. Either the file was "
                  "hand-edited or the manifest was not updated; re-run with --update.",
                  file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
