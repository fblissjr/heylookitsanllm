#!/usr/bin/env bash
#
# Pre-commit guard: the COMMITTED pyproject.toml and uv.lock must always point
# at published releases. A git pin (a [tool.uv.sources] entry with a 40-hex
# `rev`, a git-sourced entry in uv.lock) is a personal "run this commit today"
# experiment -- committing one strands every cloner's `uv sync` on it.
#
# uv leaves no gitignored home for this state (override-dependencies in uv.toml
# is silently ignored -- verified on uv 0.11.32 -- and source pins always
# propagate into uv.lock), so experiments necessarily dirty the tracked files
# and this guard is what keeps them out of history.
#
# Checks STAGED blobs (git show :<file>), not the working tree, so a clean
# commit made while the worktree runs a pinned experiment passes.
#
# Failure posture is CLOSED: `set -e` plus captured `git show` output means a
# broken read blocks the commit instead of silently passing. Matching is
# bash-native on purpose -- hooks inherit the user PATH, and a grep shim there
# has silently returned zero matches before.
#
# To commit anyway (a deliberate, reviewed git dependency):
#   HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ...
#
# The normal fix: delete the [tool.uv.sources] entry (or the whole table),
# run `uv lock`, commit, then re-add your pin.

set -euo pipefail

if [ "${HEYLOOK_ALLOW_CHANNEL_COMMIT:-}" = "1" ]; then
    exit 0
fi

errors=0
staged=$(git diff --cached --name-only --diff-filter=ACMR)

fail() {
    echo "ERROR: $1"
    errors=1
}

staged_has() {
    local f
    while IFS= read -r f; do
        [ "$f" = "$1" ] && return 0
    done <<< "$staged"
    return 1
}

if staged_has "pyproject.toml"; then
    content=$(git show :pyproject.toml)  # set -e: a failed read blocks the commit
    sha_re='rev[[:space:]]*=[[:space:]]*"[0-9a-f]{40}"'
    latest_re='^[[:space:]]*overrides[[:space:]]*=.*latest'
    comment_re='^[[:space:]]*#'
    while IFS= read -r line; do
        [[ $line =~ $comment_re ]] && continue
        [[ $line =~ $sha_re ]] && fail "staged pyproject.toml pins a git commit (rev = \"<sha>\") -- pins are local-only"
        [[ $line =~ $latest_re ]] && fail "staged pyproject.toml has a 'latest' channel override -- pins are local-only"
    done <<< "$content"
fi

if staged_has "uv.lock"; then
    lock=$(git show :uv.lock)  # set -e: a failed read blocks the commit
    if [[ $lock == *'source = { git = '* ]]; then
        fail "staged uv.lock resolves a dependency from git -- relock without the pin before committing"
    fi
fi

if [ "$errors" -ne 0 ]; then
    echo ""
    echo "Committed pyproject.toml/uv.lock must stay on published releases."
    echo "Remove the [tool.uv.sources] entry, run 'uv lock', commit, then re-add the pin."
    echo "Deliberate exception: HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ..."
    exit 1
fi

exit 0
