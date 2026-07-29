#!/usr/bin/env bash
#
# install_claude_code_wsl.sh
# Install Claude Code inside a WSL (Ubuntu/Debian) shell.
#
# IMPORTANT: run this from your *WSL* terminal, not Windows PowerShell/CMD:
#     bash install_claude_code_wsl.sh
#
set -euo pipefail

echo "==> Claude Code installer for WSL"

# 1. Sanity check: are we actually inside WSL/Linux?
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  echo "    Detected WSL."
else
  echo "    Warning: this doesn't look like WSL. Continuing anyway (plain Linux is fine)."
fi

# 2. Prerequisites (curl + git). Skips gracefully if not apt-based.
if command -v apt-get >/dev/null 2>&1; then
  echo "==> Installing prerequisites (curl, git, ca-certificates)"
  sudo apt-get update -y
  sudo apt-get install -y curl git ca-certificates
else
  echo "    Non-apt system; make sure 'curl' and 'git' are installed."
fi

# 3. Install Claude Code (official native installer — no Node.js required).
echo "==> Installing Claude Code"
curl -fsSL https://claude.ai/install.sh | bash

# 4. Ensure ~/.local/bin is on PATH now and in future shells.
CLAUDE_BIN="$HOME/.local/bin"
if [ -d "$CLAUDE_BIN" ]; then
  case ":${PATH}:" in
    *":${CLAUDE_BIN}:"*) ;;                       # already on PATH
    *)
      export PATH="${CLAUDE_BIN}:${PATH}"
      if ! grep -qs '.local/bin' "${HOME}/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "${HOME}/.bashrc"
        echo "    Added ~/.local/bin to PATH in ~/.bashrc"
      fi
      ;;
  esac
fi

# 5. Verify.
echo "==> Verifying install"
if command -v claude >/dev/null 2>&1; then
  claude --version || true
  echo
  echo "Done. Open a NEW terminal (or run 'source ~/.bashrc'), then start it with:"
  echo "    claude"
  echo
  echo "On first launch it'll walk you through signing in to your Anthropic/Claude account."
else
  echo "Install finished but 'claude' isn't on PATH in this shell yet."
  echo "Run:  source ~/.bashrc   (or open a new terminal), then:  claude"
fi




# bash /mnt/c/Users/<you>/Dropbox/install_claude_code_wsl.sh