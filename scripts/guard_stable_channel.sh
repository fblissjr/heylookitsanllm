#!/usr/bin/env bash
#
# Pre-commit guard: blocks accidental commits of floating git refs (branch/tag/
# unpinned git), non-40-hex revs, local path sources, or 'latest' channel
# overrides in pyproject.toml / uv.lock. Exact 40-hex commit pins (`rev =
# "<40-hex>"` and `#<40-hex>` lock entries) plus `source = { editable = "." }`
# commit cleanly without an override.
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
# To commit an intentional override (floating ref, path source, or channel override):
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
    section_re='^[[:space:]]*\[([^]]+)\]'
    uv_sources_re='^tool\.uv\.sources(\.|$)'
    sha_rev_re='(^|[[:space:],{])rev[[:space:]]*=[[:space:]]*"[0-9a-f]{40}"([[:space:],}]|$)'
    any_rev_re='(^|[[:space:],{])rev[[:space:]]*='
    git_re='(^|[[:space:],{])git[[:space:]]*='
    branch_re='(^|[[:space:],{])branch[[:space:]]*='
    tag_re='(^|[[:space:],{])tag[[:space:]]*='
    path_re='(^|[[:space:],{])path[[:space:]]*='
    latest_re='^[[:space:]]*overrides[[:space:]]*=.*latest'
    comment_re='^[[:space:]]*#'
    in_uv_sources=0
    pending_table_git=0
    while IFS= read -r line; do
        [[ $line =~ $comment_re ]] && continue
        if [[ $line =~ $section_re ]]; then
            if (( pending_table_git )); then
                fail "staged pyproject.toml uses an unpinned git source -- pin rev = \"<40-hex>\""
            fi
            section="${BASH_REMATCH[1]}"
            if [[ $section =~ $uv_sources_re ]]; then
                in_uv_sources=1
            else
                in_uv_sources=0
            fi
            continue
        fi
        [[ $line =~ $latest_re ]] && fail "staged pyproject.toml has a 'latest' channel override -- pins are local-only"
        if (( in_uv_sources )); then
            [[ $line =~ $branch_re ]] && fail "staged pyproject.toml pins a floating git branch -- pin rev = \"<40-hex>\""
            [[ $line =~ $tag_re ]] && fail "staged pyproject.toml pins a git tag -- pin rev = \"<40-hex>\""
            [[ $line =~ $path_re ]] && fail "staged pyproject.toml uses a local path source -- pins are local-only"
            if [[ $line =~ $any_rev_re ]]; then
                if ! [[ $line =~ $sha_rev_re ]]; then
                    fail "staged pyproject.toml uses a non-40-hex rev -- pin a full 40-char git SHA"
                fi
                pending_table_git=0
            fi
            if [[ $line =~ $git_re ]]; then
                if [[ $line == *"{"* ]]; then
                    if ! [[ $line =~ $sha_rev_re ]]; then
                        fail "staged pyproject.toml uses an unpinned git source -- pin rev = \"<40-hex>\""
                    fi
                elif ! [[ $line =~ $sha_rev_re ]]; then
                    pending_table_git=1
                fi
            fi
        fi
    done <<< "$content"
    if (( pending_table_git )); then
        fail "staged pyproject.toml uses an unpinned git source -- pin rev = \"<40-hex>\""
    fi
fi

if staged_has "uv.lock"; then
    lock=$(git show :uv.lock)  # set -e: a failed read blocks the commit
    pinned_git_re='^[[:space:]]*source[[:space:]]*=[[:space:]]*\{[[:space:]]*git[[:space:]]*=[[:space:]]*"[^"]*#[0-9a-f]{40}"[[:space:]]*\}[[:space:]]*$'
    self_editable_re='^[[:space:]]*source[[:space:]]*=[[:space:]]*\{[[:space:]]*editable[[:space:]]*=[[:space:]]*"\."[[:space:]]*\}[[:space:]]*$'
    while IFS= read -r line; do
        if [[ $line == *'source = { git = '* ]] && ! [[ $line =~ $pinned_git_re ]]; then
            fail "staged uv.lock has an unpinned git dependency (expected #<40-hex>)"
        fi
        if [[ $line == *'source = { editable = '* ]] && ! [[ $line =~ $self_editable_re ]]; then
            fail "staged uv.lock has a non-self editable source -- relock without local path pins"
        fi
        if [[ $line == *'source = { path = '* || $line == *'source = { directory = '* ]]; then
            fail "staged uv.lock has a local path dependency -- relock without local path pins"
        fi
    done <<< "$lock"
fi

if [ "$errors" -ne 0 ]; then
    echo ""
    echo "Pin an exact 40-hex SHA (`rev = \"<40-hex>\"`) and relock (`uv lock`),"
    echo "or override intentionally for a local/floating experiment:"
    echo "    HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ..."
    echo "If the pin is a working-tree experiment, unstage pyproject.toml/uv.lock"
    echo "(or remove the entry and 'uv lock') and commit the rest."
    exit 1
fi

exit 0
