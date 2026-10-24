"""换引擎的产品级约定：**换的是读者，不是库。**

真实的两引擎对照要两把 key（DeepSeek / Kimi），那是量具不是 CI——这里钉的是能
确定的三件事：

1. 两个具名预设的地址都过得了 ``endpoint_problem``（https）；
2. 记忆草稿里**没有任何引擎身份字段**——写进去的内容不随引擎变；
3. 巩固路径的签名里没有 backend/model——它根本看不见引擎。
"""

from __future__ import annotations

import dataclasses
import inspect

from axo.llm.openai_compat import endpoint_problem
from axo.memory.consolidation import run_consolidation
from axo.memory.policy import MemoryDraft
from axo.ui.settings_store import CLOUD_PRESETS


def test_every_named_preset_is_a_secure_endpoint() -> None:
    for preset in CLOUD_PRESETS:
        assert endpoint_problem(preset.base_url) == "", preset.key


def test_a_memory_draft_carries_no_engine_identity() -> None:
    names = {field.name for field in dataclasses.fields(MemoryDraft)}
    assert not any("model" in name or "engine" in name for name in names)


def test_consolidation_never_sees_a_backend() -> None:
    parameters = inspect.signature(run_consolidation).parameters
    assert not any("backend" in name or "model" in name for name in parameters)
