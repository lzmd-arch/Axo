"""本地把 CI 那六条腿、外加一条 CI 上没有的 gui 腿，一起跑一遍。

**为什么命令写在这里而不是 ``ci.bat`` 里。** CI 的判据是
``.github/workflows/ci.yml``，本地入口是 ``ci.bat``。两边各抄一份命令的话，某个
job 在 CI 里加了参数（比如 ``--ignore-glob`` 换了写法）而本地没跟上，症状是"本地
全绿、推上去红"——而那正是本地入口存在的理由。所以命令只有一份，写在这个 ``.py``
里，``ci.bat`` 只是 ``uv run python tools/ci_local.py`` 的一层壳。

**六条腿与 CI 一一对应**（job 名见 ci.yml）：``lint`` / ``test`` / ``goldset`` /
``router`` / ``substrate`` / ``package``。本地**不跑 package**——它要构造 wheel 并在
一个干净 venv 里装一遍，那是"发布物对不对"的问题，跟日常改代码无关，而且它是唯一
会碰网络的（``biosnn-bus`` 从 PyPI 装）。CI 上它照跑。

**外加一条 CI 上没有的 gui 腿。** CI 的 ``test`` job 用
``--ignore-glob='tests/test_ui_*'`` 把 11 个 gui 文件排除在外（为了在无头 Linux 上
不拖进整套 Qt 系统库），于是那两百多条——**包括路由的接线守卫，和"收尾期间不崩
``0xC0000409``"那条守卫**——在推上去之前没有任何东西会跑它们。

在这条腿加进来之前，跑与不跑**完全取决于有没有人记得手敲一行** ``pytest -m gui``。
一句"记得跑"不是一条守卫，而这两组守卫恰好守的都是本仓踩过的坑。所以它排在这里，
与其余几条共用同一个解释器、同一个工作目录，一次跑完。

它**进不了** ``--fast``：``--fast`` 的用途正是"只改了界面"，而那正是这条腿要看的东西。

覆盖边界要说清楚：这条腿证明的是"**本机**上 gui 测试全过"，证明不了"CI 机器上会过"
——后者要 gui 腿自己进 ``ci.yml``，是另一件事（需要重新评估在无头 Linux 上装 Qt 的
代价）。所以汇总里它标着「本地独有」，而不是假装它是 CI 的一部分。

**三条门槛腿判的是不相干的东西**（召回、路由、基质），但共用一个退出码约定：1 = 跌破
容差、2 = 用法或环境不对、3 = 结构不变量被违反（**那是 bug**）。所以 1 从来不是
"程序出错"，而"本地红、CI 绿"这种话在门槛腿上是没有意义的——两边跑的是同一份
基线文件。

**都跑完再报结论**，不是第一条红就停：CI 上它们是几个独立 job，同时红两条是常态，
而本地 fail-fast 只会让你跑三次才发现是三件事。退出码取**第一条**非零的那个，
其余的在汇总里列出来。

用法::

    python tools/ci_local.py           # 七条腿
    python tools/ci_local.py --fast    # 跳过三条门槛腿（改文档或只改界面时用）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

_ROOT = Path(__file__).resolve().parents[1]


class _Leg(NamedTuple):
    """一条腿：跑什么、算 CI 上的哪个 job、要不要额外的环境变量。

    用具名元组而不是裸元组，是因为加了 ``env`` 之后四个位置挤在一行里已经读不出
    谁是谁了——而 ``job``（给人看的"这对应 CI 的哪个 job"）与 ``env``（给机器的）
    挨着放，正是最容易传错位的一对。
    """

    name: str
    command: list[str]
    job: str
    env: dict[str, str] | None = None


def _legs(fast: bool) -> list[_Leg]:
    """要跑的腿，按"先快后慢"排。

    命令一律走 ``sys.executable -m ...`` 而不是裸 ``ruff``/``pytest``：这个脚本
    本来就是 ``uv run`` 起来的，用同一个解释器能保证两边跑的是同一套依赖——而
    "本地那份 ruff 是别的环境里的"正是会让本地与 CI 给出不同答案的那类偏差。
    """
    legs: list[_Leg] = [
        _Leg("ruff check", [sys.executable, "-m", "ruff", "check", "."], "lint"),
        _Leg(
            "ruff format --check", [sys.executable, "-m", "ruff", "format", "--check", "."], "lint"
        ),
        _Leg(
            "核心腿 pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                # **`-p no:pytest-qt` 不能省**：排除 test_ui_* 只是不选那些文件，
                # 而 pytest-qt 的 `pytest_configure` 在收集之前就把 QtGui 导入了
                # ——它在环境里（dev 组带 PySide6），这条腿就一个测试也跑不到。
                # 无头 Linux 上表现为 `ImportError: libEGL.so.1`，退出码 3，报错里
                # 一个字都没提 Qt 测试，很难往回找。本机（有显示器的 Windows）即使
                # 忘了加也看不出来，所以这条只在 CI 上暴露——CI 第一次跑就是死在这。
                "-p",
                "no:pytest-qt",
                "--ignore-glob=tests/test_ui_*",
            ],
            "test",
        ),
        _Leg(
            "gui 腿 pytest（离屏）",
            [sys.executable, "-m", "pytest", "-m", "gui"],
            "本地独有",
            # 离屏：这些测试里有的会 `show()` 真控件，在有显示器的机器上会让窗口
            # 一闪而过；而它们本来就不看画面——断言全在布局宽度与调色板上。
            # 用 env 合并而不是 `os.environ` 就地改：那会漏给后面几条腿，而门槛腿
            # 的行为不该因为"前面跑过 gui"而变。
            {"QT_QPA_PLATFORM": "offscreen"},
        ),
    ]
    if not fast:
        legs.append(
            _Leg(
                "金标准门槛（纯词法）",
                [
                    sys.executable,
                    str(_ROOT / "tools" / "bench_recall.py"),
                    "--no-semantic",
                    "--check",
                    "--quiet",
                ],
                "goldset",
            )
        )
        legs.append(
            _Leg(
                "路由门槛（规则层）",
                [
                    sys.executable,
                    str(_ROOT / "tools" / "bench_router.py"),
                    "--no-model",
                    "--check",
                    "--quiet",
                ],
                "router",
            )
        )
        legs.append(
            _Leg(
                "基质门槛（图 A 与 W）",
                [
                    sys.executable,
                    str(_ROOT / "tools" / "bench_substrate.py"),
                    "--check",
                    "--quiet",
                ],
                "substrate",
            )
        )
    return legs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fast",
        action="store_true",
        help="跳过三条门槛腿（只改了文档或界面时用；召回/路由/基质相关的改动别用）",
    )
    args = parser.parse_args(argv)

    results: list[tuple[str, str, int]] = []
    for leg in _legs(args.fast):
        print(f"\n{'═' * 72}\n▸ {leg.name}   （CI job: {leg.job}）\n{'═' * 72}", flush=True)
        env = None if leg.env is None else {**os.environ, **leg.env}
        code = subprocess.run(leg.command, cwd=_ROOT, check=False, env=env).returncode
        results.append((leg.name, leg.job, code))
        print(f"── {leg.name}: {'通过' if code == 0 else f'失败（退出码 {code}）'}", flush=True)

    failed = [(name, code) for name, _job, code in results if code != 0]
    print(f"\n{'═' * 72}\n汇总：{len(results) - len(failed)}/{len(results)} 条腿通过\n{'═' * 72}")
    for name, job, code in results:
        print(
            f"  {'✅' if code == 0 else '❌'} {name:<24} job={job}"
            + ("" if code == 0 else f"  退出码 {code}")
        )
    if failed:
        print(
            "\n三条门槛腿的退出码有含义：1 = 跌破容差（行为确实变了，不是程序出错）、"
            "2 = 用法或环境不对、3 = 结构不变量被违反（**那是 bug**）。"
            "\n分流办法见 README 的排障表。"
        )
        # 退出码取**第一条**非零的：它的含义是具体的（谁先坏的），而"非零的条数"
        # 没有含义，也没有任何工具会去读。
        return failed[0][1]
    print("\n与 CI 一致（标了「本地独有」的那条除外——CI 上没有它，见模块文档）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
