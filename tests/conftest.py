"""测试公共配置。

只做一件事：把 ``src/`` 挂进 ``sys.path``。这样在没装包的情况下也能直接跑
``pytest``——开发时改了代码立刻能验，不必先 ``pip install -e .`` 一遍。

显式插到列表最前面（而不是 ``append``）：万一系统里装过同名的旧包，我们要测的
是眼前这份源码，不是 site-packages 里那份。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
