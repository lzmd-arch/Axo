"""混合渲染：模板先、模型后。

Phase 2 的入口。它**不改变** :class:`~axo.expression.layer.ExpressionLayer` 的失败
语义：模板命中但事实核对不过、或问句被写成陈述句，就当作没命中，继续走模型；模型
失败，再退回草稿。三层（模板 → 模型 → 草稿）任何一层都不抛异常。
"""

from __future__ import annotations

from collections.abc import Callable

from .layer import ExpressionLayer, ExpressionOutcome
from .sir import SIR
from .templates import TemplateBank

__all__ = ["MixedRenderer"]


class MixedRenderer:
    """模板库 + Phase 1 渲染器。

    Args:
        bank: 模板库（见 :class:`~axo.expression.templates.TemplateBank`）。
        layer: 模型渲染器（Phase 1）。
    """

    def __init__(self, bank: TemplateBank, layer: ExpressionLayer) -> None:
        self._bank = bank
        self._layer = layer

    def render(
        self,
        sir: SIR,
        *,
        keep_going: Callable[[], bool] | None = None,
    ) -> ExpressionOutcome:
        hit = self._bank.render(sir)
        if hit is not None:
            return ExpressionOutcome(
                hit.text,
                "styled",
                "",
                route="template",
                template_id=hit.template_id,
            )
        return self._layer.render(sir, keep_going=keep_going)
