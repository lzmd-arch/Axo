"""多语言仓库地图：常见语言的声明要能摘出来，控制语句不能被当成方法。"""

from __future__ import annotations

from pathlib import Path

import pytest

from axo.agent.repo_map import build_repo_map

_CASES = {
    "app.js": (
        """export class Greeter {
  greet(name) { return name; }
}
export function main() {}
const helper = (x) => x;
if (ready) { main(); }
""",
        ("class Greeter", "method greet", "function main", "function helper"),
    ),
    "types.ts": (
        """export interface User { id: number }
export type ID = string;
export enum Color { Red }
export class Store<T> {
  get(key: string): T { return null; }
}
export function load(id: ID): User { return null; }
""",
        ("type User", "type ID", "type Color", "class Store", "method get", "function load"),
    ),
    "main.go": (
        """package main
type Server struct {}
func (s *Server) Start() {}
func main() {}
""",
        ("type Server", "function Start", "function main"),
    ),
    "lib.rs": (
        """pub struct Config {}
pub enum Mode { A }
pub trait Run { fn run(&self); }
impl Config { pub fn new() -> Self { Config {} } }
pub fn helper() {}
mod tests;
""",
        ("type Config", "type Mode", "type Run", "impl Config", "function helper", "mod tests"),
    ),
    "Main.java": (
        """public class Main {
    private int value;
    public Main() { this.value = 0; }
    public static void main(String[] args) { }
    private int add(int a, int b) { return a + b; }
}
interface Thing { void run(); }
""",
        ("type Main", "constructor Main", "method main", "method add", "type Thing"),
    ),
    "App.kt": (
        """data class User(val id: Int)
class Greeter {
    fun greet(name: String): String { return name }
}
fun helper() {}
""",
        ("type User", "type Greeter", "function greet", "function helper"),
    ),
    "Program.cs": (
        """public class Program {
    public Program() { }
    public static void Main(string[] args) { }
    private int Add(int a, int b) { return a + b; }
}
""",
        ("type Program", "constructor Program", "method Main", "method Add"),
    ),
    "main.c": (
        """#include <stdio.h>
struct Point { int x; };
static int add(int a, int b) { return a + b; }
int main(void) { return 0; }
if (ready) { add(1, 2); }
""",
        ("type Point", "function add", "function main"),
    ),
    "main.cpp": (
        """namespace app {
class Widget { public: void draw(); };
void Widget::draw() { }
int helper(int x) { return x; }
}
""",
        ("namespace app", "type Widget", "function Widget::draw", "function helper"),
    ),
    "app.rb": (
        """class Greeter
  def greet(name)
    name
  end
end
def helper; end
""",
        ("type Greeter", "method greet", "method helper"),
    ),
    "app.php": (
        """<?php
class Greeter {
    public function greet($name) { return $name; }
}
function helper() {}
""",
        ("type Greeter", "function greet", "function helper"),
    ),
    "App.swift": (
        """struct User { let id: Int }
class Greeter {
    func greet(name: String) -> String { return name; }
}
func helper() {}
""",
        ("type User", "type Greeter", "function greet", "function helper"),
    ),
    "App.scala": (
        """case class User(id: Int)
object Greeter { def greet(name: String) = name }
class Main { def run(): Unit = () }
""",
        ("declaration User", "declaration Greeter", "declaration Main"),
    ),
    "app.lua": (
        """local function helper() end
function Greeter:greet(name) return name end
""",
        ("function helper", "function Greeter:greet"),
    ),
    "app.r": ("helper <- function(x) x\n", ("function helper",)),
    "app.sh": (
        """#!/bin/bash
helper() { echo hi; }
function other { echo hi; }
""",
        ("function helper", "function other"),
    ),
    "app.dart": (
        """class Greeter { void greet(String name) {} }
Future<void> main() async {}
""",
        ("type Greeter", "function main"),
    ),
    "app.ex": (
        """defmodule Greeter do
  def greet(name), do: name
  defp helper(x), do: x
end
""",
        ("module Greeter", "function greet", "function helper"),
    ),
    "App.hs": (
        """data User = User Int
greet :: String -> String
greet name = name
""",
        ("type User", "function greet"),
    ),
    "app.clj": (
        """(defn greet [name] name)
(defprotocol Thing (run [this]))
""",
        ("function greet", "declaration Thing"),
    ),
    "app.pl": (
        """package Greeter;
sub greet { return $_[0]; }
""",
        ("package Greeter", "function greet"),
    ),
    "app.ps1": (
        """function Get-Thing { return 1 }
function helper() { }
""",
        ("function Get-Thing", "function helper"),
    ),
    "view.vue": (
        """<template><div /></template>
<script>
export function setup() { return null; }
</script>
""",
        ("function setup",),
    ),
}


@pytest.mark.parametrize("filename", sorted(_CASES))
def test_common_language_symbols(tmp_path: Path, filename: str) -> None:
    source, expected = _CASES[filename]
    (tmp_path / filename).write_text(source, encoding="utf-8")
    text = build_repo_map(tmp_path, max_files=20, max_symbols=100)
    for needle in expected:
        assert needle in text, f"{filename} 缺了 {needle!r}：\n{text}"
    assert "method if " not in text
    assert "method for " not in text


def test_unknown_extension_is_listed_without_symbols(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("def not_a_symbol():\n", encoding="utf-8")
    text = build_repo_map(tmp_path)
    assert "notes.txt" in text
    assert "not_a_symbol" not in text


def test_tsx_uses_typescript_patterns(tmp_path: Path) -> None:
    (tmp_path / "view.tsx").write_text(
        "export function View() { return null; }\nexport const useThing = () => null;\n",
        encoding="utf-8",
    )
    text = build_repo_map(tmp_path)
    assert "function View" in text
    assert "function useThing" in text
