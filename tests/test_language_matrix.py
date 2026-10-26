"""常见语言的工具面一致性：符号索引、LSP 映射与服务器候选都不能漏。"""

from __future__ import annotations

import pytest

from axo.agent import lsp, repo_map

#: 后缀 -> 期望的语言 id。**只列“我们声明支持”的常见语言**；不是所有语言。
_COMMON = {
    ".py": ("python", "python"),
    ".js": ("javascript", "javascript"),
    ".ts": ("typescript", "typescript"),
    ".tsx": ("typescript", "typescript"),
    ".go": ("go", "go"),
    ".rs": ("rust", "rust"),
    ".java": ("java", "java"),
    ".kt": ("kotlin", "kotlin"),
    ".cs": ("csharp", "csharp"),
    ".c": ("c", "c"),
    ".cpp": ("cpp", "cpp"),
    ".rb": ("ruby", "ruby"),
    ".php": ("php", "php"),
    ".swift": ("swift", "swift"),
    ".scala": ("scala", "scala"),
    ".lua": ("lua", "lua"),
    ".r": ("r", "r"),
    ".sh": ("shell", "shell"),
    ".ps1": ("shell", "powershell"),
    ".pl": ("perl", "perl"),
    ".dart": ("dart", "dart"),
    ".ex": ("elixir", "elixir"),
    ".hs": ("haskell", "haskell"),
    ".clj": ("clojure", "clojure"),
    ".vue": ("javascript", "vue"),
    ".svelte": ("javascript", "svelte"),
}


@pytest.mark.parametrize("suffix", sorted(_COMMON))
def test_common_language_has_symbol_index(suffix: str) -> None:
    repo_language, _lsp_language = _COMMON[suffix]
    assert repo_map._SUFFIX_LANGUAGE.get(suffix) == repo_language
    if repo_language != "python":
        assert repo_language in repo_map._PATTERNS, f"{repo_language} 没有声明行模式"


@pytest.mark.parametrize("suffix", sorted(_COMMON))
def test_common_language_has_lsp_mapping_and_server(suffix: str) -> None:
    _repo_language, lsp_language = _COMMON[suffix]
    expected = _COMMON[suffix][1]
    mapped = lsp._LANGUAGE_BY_SUFFIX.get(suffix)
    assert mapped is not None, f"{suffix} 没有 LSP 语言映射"
    assert mapped[0] == expected, f"{suffix} 映射到了 {mapped[0]}，期望 {expected}"
    assert lsp._DEFAULT_SERVERS.get(lsp_language), f"{lsp_language} 没有默认 LSP 服务器候选"


def test_suffix_tables_are_lowercase() -> None:
    assert all(key == key.lower() for key in repo_map._SUFFIX_LANGUAGE)
    assert all(key == key.lower() for key in lsp._LANGUAGE_BY_SUFFIX)
