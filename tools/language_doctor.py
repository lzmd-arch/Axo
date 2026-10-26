"""从源码检出运行语言体检。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axo.agent.language_doctor import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
