#!/usr/bin/env bash
# Reject commit subjects that are not conventional commits.
#
# Checks BASE_SHA..HEAD when BASE_SHA is set and reachable, otherwise just the
# tip commit. Merge commits are skipped.
set -uo pipefail

PATTERN='^(feat|fix|perf|refactor|docs|test|style|ci|chore|build|revert)(\([a-z0-9._/-]+\))?!?: .+'

BASE="${BASE_SHA:-}"
if [ -n "$BASE" ] && git cat-file -e "${BASE}^{commit}" 2>/dev/null; then
  SUBJECTS=$(git log "${BASE}..HEAD" --no-merges --pretty=format:'%s')
else
  SUBJECTS=$(git log -1 --no-merges --pretty=format:'%s' || true)
fi

bad=0
while IFS= read -r subject; do
  [ -n "$subject" ] || continue
  if ! printf '%s' "$subject" | grep -qE "$PATTERN"; then
    echo "::error::not a conventional commit: ${subject}"
    case "$subject" in
      *" : "*) echo "  hint: no space before the colon — write 'chore: ...'" ;;
      *) echo "  hint: use 'type: subject' or 'type(scope): subject'" ;;
    esac
    bad=1
  fi
done <<EOF
$SUBJECTS
EOF

if [ "$bad" -ne 0 ]; then
  echo "types: feat fix perf refactor docs test style ci chore build revert"
  exit 1
fi
echo "all commit subjects are conventional"
