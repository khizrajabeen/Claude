#!/usr/bin/env bash
# One cycle: trade the day, write the dashboard's data, push it.
#
# Run this from a timer rather than looping here. A crashed loop looks
# identical to a quiet market; a timer that fails to fire is visible in
# the journal.
set -euo pipefail

cd "$(dirname "$0")/.."

# Credentials live here and nowhere else. .env is gitignored.
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

MODE="${MERIDIAN_MODE:-day}"
PUBLISH_BRANCH="${MERIDIAN_PUBLISH_BRANCH:-}"

python3 main.py "$MODE" "$@"

# Publish whatever the journal now says, including a fresh market screen.
python3 main.py publish --screen

# Pushing is opt-in: a machine without write access should still trade.
if [[ -n "$PUBLISH_BRANCH" ]] && ! git diff --quiet -- web/data; then
  git add web/data
  git -c user.name="meridian-bot" -c user.email="bot@localhost" \
      commit -m "Publish dashboard data $(date -u +%Y-%m-%dT%H:%MZ)" --quiet
  git push origin "HEAD:${PUBLISH_BRANCH}" --quiet
  echo "Published to ${PUBLISH_BRANCH}."
else
  echo "Nothing to publish, or MERIDIAN_PUBLISH_BRANCH is unset."
fi
