"""axo —— Axo（千岁 / 阿螈）。

一句话：**把长期记忆与性格放在终端侧，而不是放在模型里。**

两个它要解决的问题是同一个机制的两种说法：

* *上下文长度有限* → 不靠塞满窗口。分层装配 + 硬 token 预算 + 超预算按层反向
  驱逐 + 滚动摘要，每轮产出一份可审计的"注入了什么"账本。
* *记忆跟着模型走* → 记忆库是终端侧的本地文件（sqlite + npz + 纯文本 yaml），
  **模型只通过检索注入读它，永远不拥有它**。换模型 = 换一个读者，库、人格锚、
  偏好权重一字节不变。这就是"每个模型第一次接入就拥有完全一致的记忆与性格"的
  工程口径。

包结构::

    memcore            ★ 记忆核心：纯 numpy，不 import Qt、不 import HTTP
    axo
        llm            模型后端：Ollama（本机）+ OpenAI 兼容端点（云端，只做对话）
        session        会话逐字稿、token 预算、上下文装配
        agent          工具、循环、闸门、路由
        memory         记忆的写与看：策略、写入、回填、两条人格锚
        bench          量具层：读语料、跑被测对象、算子项
        ui             PySide6 壳 + 嵌在里面的网页对话区
        paths          数据目录解析
        secrets        密钥加解密（Windows DPAPI / macOS Keychain，**平台相关的东西集中在这一处**）

依赖方向单向：:mod:`axo` 依赖 :mod:`memcore`，反过来不成立。
"""

from __future__ import annotations

__version__ = "0.12.0"

__all__ = ["__version__"]
