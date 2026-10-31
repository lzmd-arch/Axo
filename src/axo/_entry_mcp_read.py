"""冻结入口：``axo-mcp``（只读档位）。

PyInstaller 把入口当 ``__main__`` 跑，``__package__`` 是空的，所以这里只能是**绝对
import**——同 ``axo/__main__.py`` 的理由。源码检出时没人 import 这个文件，它只服务打包。
"""

from axo.mcp_server import main

raise SystemExit(main())
