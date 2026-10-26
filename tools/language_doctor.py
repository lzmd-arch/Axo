"""语言支持体检：符号索引、LSP 映射、服务器是否真的装在本机。

用法：
    python tools/language_doctor.py
    python tools/language_doctor.py --json

它**不安装任何东西**，只回答“这个语言的哪一层已经就绪、缺什么”。LSP 服务器可由
`AXO_LSP_<语言>` 指定，或放进 `AXO_LSP_BIN_DIRS`（分号/冒号分隔）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from axo.agent import lsp, repo_map  # noqa: E402

COMMON = (
    (".py", "python", "python"),
    (".js", "javascript", "javascript"),
    (".ts", "typescript", "typescript"),
    (".tsx", "typescript", "typescript"),
    (".go", "go", "go"),
    (".rs", "rust", "rust"),
    (".java", "java", "java"),
    (".kt", "kotlin", "kotlin"),
    (".cs", "csharp", "csharp"),
    (".c", "c", "c"),
    (".cpp", "cpp", "cpp"),
    (".rb", "ruby", "ruby"),
    (".php", "php", "php"),
    (".swift", "swift", "swift"),
    (".scala", "scala", "scala"),
    (".lua", "lua", "lua"),
    (".r", "r", "r"),
    (".sh", "shell", "shell"),
    (".ps1", "shell", "powershell"),
    (".pl", "perl", "perl"),
    (".dart", "dart", "dart"),
    (".ex", "elixir", "elixir"),
    (".hs", "haskell", "haskell"),
    (".clj", "clojure", "clojure"),
    (".vue", "javascript", "vue"),
    (".svelte", "javascript", "svelte"),
)


def _short_server(server: list[str]) -> str:
    if os.name == "nt" and len(server) >= 3 and server[1].lower() == "/c":
        return Path(server[2]).name
    return Path(server[0]).name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()
    manager = lsp.LspManager()
    rows = []
    for suffix, repo_language, lsp_language in COMMON:
        server = manager.server_for(lsp_language)
        hint = ""
        if server is None:
            candidates = lsp._DEFAULT_SERVERS.get(lsp_language, ())
            hint = candidates[0][0] if candidates else f"AXO_LSP_{lsp_language.upper()}"
            if lsp_language == "java":
                hint += "（或设 AXO_JDTLS_HOME）"
        rows.append(
            {
                "suffix": suffix,
                "symbol_index": suffix in repo_map._SUFFIX_LANGUAGE,
                "repo_language": repo_language,
                "lsp_language": lsp_language,
                "server": server,
                "installed": server is not None,
                "install_hint": hint,
            }
        )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    print(f"{'后缀':<7} {'符号索引':<8} {'LSP 语言':<12} {'服务器'}")
    for row in rows:
        status = _short_server(row["server"]) if row["server"] else f"缺：{row['install_hint']}"
        print(
            f"{row['suffix']:<7} {'yes' if row['symbol_index'] else 'NO':<8} "
            f"{row['lsp_language']:<12} {status}"
        )
    installed = sum(1 for row in rows if row["installed"])
    print(f"\n{installed}/{len(rows)} 个常见语言后缀找到 LSP 服务器。")
    print("提示：用 AXO_LSP_<语言> 指定命令，或把便携工具链 bin 目录放进 AXO_LSP_BIN_DIRS。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
