#!/usr/bin/env bash
# Runs ON the vast.ai instance. Idempotent - safe to re-run.
#
#   provision.sh <repo_url> <branch> <dataset> [--skip-smoke]
#
# Leaves a ready-to-train checkout at /workspace/autoresearch.
set -euo pipefail

REPO_URL="${1:?repo url required}"
BRANCH="${2:?branch required}"
DATASET="${3:-tinystories}"
SKIP_SMOKE="${4:-}"

WORK=/workspace
DIR="$WORK/autoresearch"

echo "=== provision: repo=$REPO_URL branch=$BRANCH dataset=$DATASET ==="
mkdir -p "$WORK"

echo "--- host ---"
nvidia-smi || { echo "FATAL: no nvidia-smi, this box has no usable GPU"; exit 1; }
echo "cpu cores: $(nproc)"
df -h "$WORK" | tail -1

echo "--- packages ---"
export DEBIAN_FRONTEND=noninteractive
if ! command -v git >/dev/null || ! command -v curl >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq git curl ca-certificates
fi

echo "--- uv ---"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "--- checkout ---"
if [ -d "$DIR/.git" ]; then
  echo "existing checkout, fetching $BRANCH"
  git -C "$DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$DIR"
fi
cd "$DIR"

# the sweep runner makes keep/revert commits, so git needs an identity
git config --global --add safe.directory "$DIR" || true
git config user.email "sweep@scas.local"
git config user.name "scas-sweep"
# results.tsv and per-run logs stay untracked (program.md rule)
grep -qxF 'results.tsv' .git/info/exclude 2>/dev/null || {
  printf 'results.tsv\nruns/\nsweep_state.json\n' >> .git/info/exclude
}

echo "--- deps (uv sync) ---"
uv sync

echo "--- data prep (dataset=$DATASET) ---"
export AUTORESEARCH_DATASET="$DATASET"
uv run prepare.py --dataset "$DATASET"

if [ "$SKIP_SMOKE" = "--skip-smoke" ]; then
  echo "--- smoke test skipped ---"
else
  echo "--- smoke test ---"
  uv run train.py --smoke-test 2>&1 | tail -25
fi

echo "--- baseline git state ---"
git rev-parse --short HEAD

cat > "$DIR/config.json" <<JSON
{
  "repo": "$REPO_URL",
  "branch": "$BRANCH",
  "dataset": "$DATASET",
  "base_commit": "$(git rev-parse HEAD)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)",
  "cpu_cores": $(nproc),
  "provisioned_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "=== provision complete ==="
