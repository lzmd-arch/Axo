"""Agent 层：工具、循环、闸门、语域路由。

**为什么单独一个包，而不是塞进** :mod:`axo.session` **或** :mod:`memcore`：

- 不进 :mod:`memcore`：那是纯 numpy 的记忆核心，``tests/test_layering.py`` 用 ast
  强制它只许 import numpy / biosnn-bus / 标准库。工具要碰记忆库、文件系统、时钟，
  它住不进去。
- 不进 ``session/``：装配器的职责是"把上下文凑出来"，它有一份很硬的性质——**不碰
  Qt、不碰网络**，所以能逐条单测。工具循环要调后端、要等用户点确认，是另一种东西。

所以这一层进 ``_PURE_LAYERS``（**纯层：不碰 Qt、不碰网络**），但比 ``session/``
宽松一点：它要用 :mod:`datetime`、要读文件。**边界是"不许 import PySide6 / httpx"**
——Ollama 后端靠**注入**（循环收一个 :class:`~axo.llm.base.ChatBackend`），
所以这一层在核心腿里能用假后端逐事件断言，一行网络都不用碰。
"""
