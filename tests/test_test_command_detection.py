"""测试命令探测：常见生态要认出各自的入口，认不出就返回空串。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from axo.agent.workspace import build_workspace_context, detect_test_command


def _touch(root: Path, name: str, text: str = "") -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (
            lambda root: (_touch(root, "pyproject.toml"), (root / "tests").mkdir()),
            "python -m pytest -q",
        ),
        (lambda root: _touch(root, "pytest.ini"), "python -m pytest -q"),
        (lambda root: _touch(root, "tox.ini"), "python -m tox"),
        (lambda root: _touch(root, "noxfile.py"), "python -m nox"),
        (
            lambda root: _touch(root, "package.json", '{"scripts": {"test": "jest"}}'),
            "npm test",
        ),
        (
            lambda root: (
                _touch(root, "package.json", '{"scripts": {"test": "vitest"}}'),
                _touch(root, "pnpm-lock.yaml"),
            ),
            "pnpm test",
        ),
        (lambda root: _touch(root, "Cargo.toml", "[package]\nname='x'\n"), "cargo test"),
        (lambda root: _touch(root, "go.mod", "module example.com/x\n"), "go test ./..."),
        (lambda root: _touch(root, "pom.xml"), "mvn -q test"),
        (lambda root: _touch(root, "build.gradle"), "gradle test"),
        (lambda root: _touch(root, "build.gradle.kts"), "gradle test"),
        (lambda root: _touch(root, "App.sln"), "dotnet test"),
        (lambda root: _touch(root, "App.csproj"), "dotnet test"),
        (
            lambda root: (_touch(root, "Gemfile"), (root / "spec").mkdir()),
            "bundle exec rspec",
        ),
        (lambda root: _touch(root, "Gemfile"), "bundle exec rake test"),
        (lambda root: _touch(root, "Rakefile"), "rake test"),
        (lambda root: _touch(root, "phpunit.xml"), "phpunit"),
        (lambda root: _touch(root, "Package.swift"), "swift test"),
        (lambda root: _touch(root, "mix.exs"), "mix test"),
        (lambda root: _touch(root, "pubspec.yaml", "dependencies:\n  flutter:\n"), "flutter test"),
        (lambda root: _touch(root, "pubspec.yaml", "name: app\n"), "dart test"),
        (lambda root: _touch(root, "CTestTestfile.cmake"), "ctest --output-on-failure"),
        (lambda root: _touch(root, "build.sbt"), "sbt test"),
        (lambda root: _touch(root, "stack.yaml"), "stack test"),
        (lambda root: _touch(root, "app.cabal"), "cabal test"),
        (lambda root: _touch(root, "deps.edn"), "clojure -X:test"),
        (lambda root: _touch(root, "project.clj"), "lein test"),
        (lambda root: _touch(root, "build.zig"), "zig build test"),
        (lambda root: _touch(root, "MODULE.bazel"), "bazel test //..."),
        (lambda root: _touch(root, "WORKSPACE"), "bazel test //..."),
        (lambda root: _touch(root, "Makefile.PL"), "prove -lr t"),
        (lambda root: _touch(root, "Thing.Tests.ps1"), "pwsh -NoProfile -Command Invoke-Pester"),
        (
            lambda root: (
                _touch(root, "tests/testthat/test-a.R"),
                _touch(root, "DESCRIPTION"),
            ),
            "Rscript -e \"testthat::test_dir('tests/testthat')\"",
        ),
        (lambda root: _touch(root, "Makefile"), "make test（如果 Makefile 有 test 目标）"),
    ],
)
def test_detects_common_project_test_commands(tmp_path: Path, setup, expected: str) -> None:
    setup(tmp_path)
    assert detect_test_command(tmp_path) == expected


def test_gradle_wrapper_is_preferred_when_present(tmp_path: Path) -> None:
    _touch(tmp_path, "build.gradle")
    wrapper = tmp_path / ("gradlew.bat" if os.name == "nt" else "gradlew")
    wrapper.write_text("", encoding="utf-8")
    assert detect_test_command(tmp_path) == f'"{wrapper}" test'


def test_cmake_with_build_directory_is_detected(tmp_path: Path) -> None:
    _touch(tmp_path, "CMakeLists.txt", "project(x)\n")
    (tmp_path / "build").mkdir()
    assert detect_test_command(tmp_path) == (
        "cmake --build build && ctest --test-dir build --output-on-failure"
    )


def test_meson_uses_existing_build_directory(tmp_path: Path) -> None:
    _touch(tmp_path, "meson.build", "project('x')\n")
    assert detect_test_command(tmp_path) == "meson setup build && meson test -C build"
    (tmp_path / "build").mkdir()
    assert detect_test_command(tmp_path) == "meson test -C build"


def test_unknown_project_returns_empty(tmp_path: Path) -> None:
    assert detect_test_command(tmp_path) == ""


def test_multilingual_test_files_are_listed(tmp_path: Path) -> None:
    _touch(tmp_path, "web/app.test.ts", "")
    _touch(tmp_path, "tests/test_app.py", "")
    _touch(tmp_path, "spec/thing_spec.rb", "")
    _touch(tmp_path, "src/test/java/AppTest.java", "")
    _touch(tmp_path, "go/thing_test.go", "")

    text = build_workspace_context((tmp_path,))
    assert "测试文件：" in text
    for name in (
        "web/app.test.ts",
        "tests/test_app.py",
        "spec/thing_spec.rb",
        "src/test/java/AppTest.java",
    ):
        assert name in text
