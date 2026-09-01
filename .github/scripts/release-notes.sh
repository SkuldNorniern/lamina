#!/usr/bin/env bash
# Group the commits since the previous tag into release notes by conventional
# commit type. Writes markdown to stdout.
#
# The format is strict: `type: subject` or `type(scope): subject`, with the
# optional `!` breaking marker. Anything else lands under "Other" so it is
# visible rather than quietly dropped — commit-lint.yml keeps new commits from
# landing there.
#
# Needs full history: check out with fetch-depth: 0.
set -euo pipefail

TAG="${GITHUB_REF_NAME:-}"
PREV_TAG=$(git tag -l 'v*' --sort=-v:refname | grep -v "^${TAG}$" | head -1 || true)
if [ -n "$PREV_TAG" ]; then
  RANGE="${PREV_TAG}..HEAD"
else
  RANGE="HEAD"
fi

KNOWN='feat|fix|perf|refactor|docs|test|style|ci|chore|build|revert'

subjects() {
  git log "$RANGE" --no-merges --pretty=format:'%s'
}

# Scopes are kept and rendered in bold, so `fix(parser): x` reads
# `- **parser:** x`.
section() {
  subjects \
    | grep -E "^${1}(\([^)]*\))?: " \
    | sed -E "s/^${1}\(([^)]*)\): /- **\1:** /; s/^${1}: /- /" \
    || true
}

# `type!: subject` and `type(scope)!: subject`. The `!` sits before the colon,
# so these never match the plain sections above and cannot be listed twice.
BREAKING=$(subjects \
  | grep -E "^(${KNOWN})(\([^)]*\))?!: " \
  | sed -E "s/^(${KNOWN})\(([^)]*)\)!: /- **\2:** /; s/^(${KNOWN})!: /- /" \
  || true)

FEATS=$(section feat)
FIXES=$(section fix)
PERFS=$(section perf)
REFACTORS=$(section refactor)
DOCS=$(section docs)
TESTS=$(section test)
# Grouped at the end rather than discarded, so the counts add up.
INTERNAL=$(printf '%s\n' "$(section chore)" "$(section style)" "$(section ci)" \
  "$(section build)" "$(section revert)" | grep -v '^$' || true)
# Whatever is left is malformed: wrong type, or a stray space before the colon.
OTHERS=$(subjects | grep -Ev "^(${KNOWN})(\([^)]*\))?!?: " | sed 's/^/- /' || true)

emit() {
  [ -n "$2" ] || return 0
  echo "### $1"
  echo "$2"
  echo ""
}

echo "## What's Changed"
echo ""
emit "Breaking Changes" "$BREAKING"
emit "Features" "$FEATS"
emit "Bug Fixes" "$FIXES"
emit "Performance" "$PERFS"
emit "Refactor" "$REFACTORS"
emit "Documentation" "$DOCS"
emit "Tests" "$TESTS"
emit "Internal" "$INTERNAL"
emit "Other" "$OTHERS"

if [ -n "$PREV_TAG" ] && [ -n "$TAG" ]; then
  echo "**Full Changelog**: https://github.com/${GITHUB_REPOSITORY}/compare/${PREV_TAG}...${TAG}"
fi
