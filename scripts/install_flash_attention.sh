#!/usr/bin/env bash
# Deprecated name: use install_post_conda_pip.sh (installs torch-scatter + flash-attn).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_post_conda_pip.sh" "$@"
