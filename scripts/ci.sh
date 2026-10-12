#!/usr/bin/env bash
set -euo pipefail

build_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${build_dir}"
}
trap cleanup EXIT

uv sync --all-packages --all-groups --locked
scripts/check.sh
git diff --check
uv build --all-packages --out-dir "${build_dir}"
