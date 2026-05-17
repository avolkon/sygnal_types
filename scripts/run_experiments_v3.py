"""Generate submission3a/b/c and submission3 (copy of 3b)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.pipeline_v3 import run_all_v3  # noqa: E402


def main() -> None:
    report = run_all_v3()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
