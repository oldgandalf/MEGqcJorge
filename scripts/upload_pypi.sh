#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
rm -rf dist
python -m build
twine upload dist/*
