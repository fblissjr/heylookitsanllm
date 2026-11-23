#!/usr/bin/env bash
#
# Pre-commit guard: blocks accidental commits of external git sources (SHAs,
# branches, tags), local path sources, or 'latest' channel overrides in
# pyproject.toml / uv.lock. Public history stays on clean PyPI releases (plus
# `source = { editable = "." }` for the package itself); local bleeding-edge
# upstream commits live outside the repo under ~/.heylook/ via
# scripts/build_llama.py and scripts/build_mlx.py.
#
# Checks STAGED blobs (git show :<file>), not the working tree, so a clean
# commit made while the worktree runs a local experiment passes.
#
# Failure posture is CLOSED: `set -e` plus captured `git show` output means a
# broken read blocks the commit instead of silently passing. Matching is
# bash-native on purpose -- hooks inherit the user PATH, and a grep shim there
# has silently returned zero matches before.
#
# To commit an intentional override:
#   HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ...

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
    git_re='(^|[[:space:],{])git[[:space:]]*='
    branch_re='(^|[[:space:],{])branch[[:space:]]*='
    tag_re='(^|[[:space:],{])tag[[:space:]]*='
    path_re='(^|[[:space:],{])path[[:space:]]*='
    latest_re='^[[:space:]]*overrides[[:space:]]*=.*latest'
    comment_re='^[[:space:]]*#'
    in_uv_sources=0
    while IFS= read -r line; do
        [[ $line =~ $comment_re ]] && continue
        if [[ $line =~ $section_re ]]; then
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
            [[ $line =~ $git_re ]] && fail "staged pyproject.toml uses a git source -- use ~/.heylook/overlays via scripts/build_mlx.py"
            [[ $line =~ $branch_re ]] && fail "staged pyproject.toml pins a floating git branch -- use local overlays"
            [[ $line =~ $tag_re ]] && fail "staged pyproject.toml pins a git tag -- use local overlays"
            [[ $line =~ $path_re ]] && fail "staged pyproject.toml uses a local path source -- pins are local-only"
        fi
    done <<< "$content"
fi

if staged_has "uv.lock"; then
    lock=$(git show :uv.lock)  # set -e: a failed read blocks the commit
    self_editable_re='^[[:space:]]*source[[:space:]]*=[[:space:]]*\{[[:space:]]*editable[[:space:]]*=[[:space:]]*"\."[[:space:]]*\}[[:space:]]*$'
    while IFS= read -r line; do
        if [[ $line == *'source = { git = '* ]]; then
            fail "staged uv.lock has an external git dependency -- relock on PyPI releases and use local overlays"
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
    echo "Public pyproject.toml / uv.lock stay on PyPI releases; use"
    echo "scripts/build_mlx.py for local git tracking."
    echo "To override intentionally:"
    echo "    HEYLOOK_ALLOW_CHANNEL_COMMIT=1 git commit ..."
    exit 1
fi

exit 0
