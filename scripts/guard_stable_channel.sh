#!/usr/bin/env bash
#
# Pre-commit guard: a git pin (a [tool.uv.sources] entry with a 40-hex `rev`,
# a git-sourced entry in uv.lock) never lands in a commit by ACCIDENT.
#
# Since 2026-09-05 (v1.79.69) mlx-lm and mlx-vlm ARE committed git pins --
# exact revs, because mlx-lm is release-starved. This guard did not go away
# with the releases-only rule: it is what makes moving a pin a named act
# (override below, SHA in CHANGELOG) and what keeps a `branch = "main"`
# experiment in the working tree from riding along with an unrelated commit.
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
    echo "A git pin does not land by accident. If this commit MOVES a committed pin"
    echo "(mlx-lm / mlx-vlm: exact rev, new SHA named in CHANGELOG):"
    echo "    HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ..."
    echo "If the pin is a working-tree experiment, unstage pyproject.toml/uv.lock"
    echo "(or remove the entry and 'uv lock') and commit the rest."
    exit 1
fi

exit 0
