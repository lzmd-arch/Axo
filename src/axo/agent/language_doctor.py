"""语言支持体检：符号索引、LSP 映射、服务器是否真的装在本机。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import lsp, repo_map

__all__ = ["COMMON", "build_rows", "format_rows", "main"]

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


def build_rows() -> list[dict[str, object]]:
    manager = lsp.LspManager()
    rows: list[dict[str, object]] = []
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
    return rows


def format_rows(rows: list[dict[str, object]]) -> str:
    lines = [f"{'后缀':<7} {'符号索引':<8} {'LSP 语言':<12} {'服务器'}"]
    for row in rows:
        server = row["server"]
        status = _short_server(server) if isinstance(server, list) else f"缺：{row['install_hint']}"
        lines.append(
            f"{row['suffix']:<7} {'yes' if row['symbol_index'] else 'NO':<8} "
            f"{row['lsp_language']:<12} {status}"
        )
    installed = sum(1 for row in rows if row["installed"])
    lines.append(f"\n{installed}/{len(rows)} 个常见语言后缀找到 LSP 服务器。")
    lines.append("提示：用 AXO_LSP_<语言> 指定命令，或把便携工具链 bin 目录放进 AXO_LSP_BIN_DIRS。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)
    rows = build_rows()
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
    else:
        print(format_rows(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
