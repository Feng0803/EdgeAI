#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${PART2_PYTHON:-}" ]]; then
    part2_python="$PART2_PYTHON"
else
    if [[ ! -x .venv/bin/python ]]; then
        python3.11 -m venv .venv >&2
    fi
    part2_python=.venv/bin/python
fi
"$part2_python" -c 'import sys; assert sys.version_info[:2] == (3,11), "Python 3.11 required"'
"$part2_python" -m pip install -r requirements.txt >&2
exec "$part2_python" -u run_all.py "$@"
