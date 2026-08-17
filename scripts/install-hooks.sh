#!/bin/bash
# Script to install local git hooks for AGENTS.md compliance checking

HOOK_DIR="$(git rev-parse --git-dir)/hooks"
PRE_PUSH_TARGET="$HOOK_DIR/pre-push"
SCRIPT_SOURCE=".githooks/pre-push"

echo "Configuring git hooks directory..."
git config core.hooksPath .githooks

if [ -f "$PRE_PUSH_TARGET" ]; then
    chmod +x "$PRE_PUSH_TARGET"
fi

if [ -f "$SCRIPT_SOURCE" ]; then
    chmod +x "$SCRIPT_SOURCE"
fi

echo "✅ Git hooks configured successfully! Pre-push audit will now run on every 'git push'."
