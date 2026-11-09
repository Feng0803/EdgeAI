#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -n "${PART3_PYTHON:-}" ]]; then
    part3_python="$PART3_PYTHON"
elif [[ -x .venv/bin/python ]]; then
    part3_python=.venv/bin/python
else
    python3.11 -m venv .venv >&2
    part3_python=.venv/bin/python
fi

"$part3_python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
"$part3_python" -m pip install -r requirements.txt >&2
"$part3_python" test_adaptquant.py >&2

# Default grading path: load the frozen reference-model policy and apply it
# directly.  No threshold derivation or target-model search is performed.
"$part3_python" -u run_adaptquant.py "$@"
