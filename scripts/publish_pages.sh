#!/bin/sh
# Copy public/ onto origin/gh-pages. Canonical tree is public/ on main.
set -e
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone -q --branch gh-pages --single-branch https://github.com/martialsystems/fly_pong.git "$TMP"
rsync -a --delete --exclude .git "$ROOT/public/" "$TMP/"
cd "$TMP"
git add -A
if git diff --cached --quiet; then
  echo "gh-pages already matches public/"
  exit 0
fi
git -c user.email="martialsystems@users.noreply.github.com" -c user.name="Martial Systems" \
  commit -m "PPO court page from public/"
git push origin gh-pages
echo "published https://martialsystems.github.io/fly_pong/"
