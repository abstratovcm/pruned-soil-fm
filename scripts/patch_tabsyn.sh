#!/usr/bin/env bash
# Compatibility patch for the vendored TabSyn baseline.
#
# TabSyn passes `verbose=True` to a PyTorch scheduler signature that no longer
# accepts it, so its VAE training raises TypeError unpatched. Run once after
# cloning TabSyn, before eval/run_tabsyn_baseline.py. Idempotent.
#
# Usage:  bash scripts/patch_tabsyn.sh [path/to/tabsyn]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABSYN_DIR="${1:-$REPO_ROOT/tabsyn}"

if [ ! -d "$TABSYN_DIR" ]; then
    echo "error: TabSyn not found at $TABSYN_DIR" >&2
    echo "       git clone https://github.com/amazon-science/tabsyn.git" >&2
    exit 1
fi

patched=0
for target in "$TABSYN_DIR/tabsyn/vae/main.py" "$TABSYN_DIR/tabsyn/main.py"; do
    if [ ! -f "$target" ]; then
        echo "error: expected file missing: $target" >&2
        exit 1
    fi
    if grep -q ', verbose=True' "$target"; then
        sed -i 's/, verbose=True//g' "$target"
        echo "patched  $target"
        patched=$((patched + 1))
    else
        echo "already  $target"
    fi
done

echo "Done ($patched file(s) modified)."
