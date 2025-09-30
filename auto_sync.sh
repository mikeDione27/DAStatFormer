#!/usr/bin/env bash
set -euo pipefail

# === Réglages ===
REPO_DIR="/d/Michel/DAStatFormer"   # <- adapte si besoin
BRANCH="main"
LOGFILE="$REPO_DIR/auto_sync.log"

cd "$REPO_DIR"

{
    echo "============================"
    echo "Auto-sync run at: $(date -Iseconds)"
    echo "Working directory: $REPO_DIR"
    echo "Branch: $BRANCH"
    echo

    # Empêcher chevauchements (run concurrent)
    LOCKFILE=".git/.auto_sync.lock"
    if [ -f "$LOCKFILE" ]; then
      echo "Lock present, exiting."
      exit 0
    fi
    trap 'rm -f "$LOCKFILE"' EXIT
    touch "$LOCKFILE"

    # Identité Git (au cas où elle n'est pas configurée globalement)
    git config user.name  "Auto Sync Bot"
    git config user.email "auto-sync@example.com"
    git config --global safe.directory "$REPO_DIR"

    echo ">>> Pulling latest changes..."
    git fetch origin
    git checkout "$BRANCH"
    git pull --rebase origin "$BRANCH"

    echo ">>> Adding files..."
    git add -A

    if ! git diff --cached --quiet; then
        echo ">>> Changes detected, committing..."
        git commit -m "Auto-sync: $(date -Iseconds)"
        echo ">>> Pushing to remote..."
        git push origin "$BRANCH"
        echo "Commit pushed successfully."
    else
        echo "No changes to commit."
    fi

    echo
    echo ">>> Current status:"
    git status -s

    echo ">>> Last commit:"
    git log -1 --oneline --decorate
    echo "============================"
    echo
} >> "$LOGFILE" 2>&1
