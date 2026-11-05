def test_bundled_llama_model_uses_the_packaged_gguf(monkeypatch, tmp_path):
    """发布包内置模型：目录里有 llama-server.exe + 约定文件名才算数。"""
    from axo.paths import LLAMA_BUNDLED_MODEL_NAME, bundled_llama_model

    bundle = tmp_path / "llama"
    bundle.mkdir()
    (bundle / "llama-server.exe").write_bytes(b"stub")
    (bundle / LLAMA_BUNDLED_MODEL_NAME).write_bytes(b"gguf")
    monkeypatch.setenv("AXO_LLAMA_DIR", str(bundle))

    assert bundled_llama_model() == bundle / LLAMA_BUNDLED_MODEL_NAME

    (bundle / LLAMA_BUNDLED_MODEL_NAME).unlink()
    assert bundled_llama_model() is None
