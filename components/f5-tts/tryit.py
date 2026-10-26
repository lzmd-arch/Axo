"""对着组件跑一次合成：量延迟、存 wav。组件本体不依赖它。"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--voice", default="")
    parser.add_argument("--out", default="试听.wav")
    parser.add_argument("--port", type=int, default=8800)
    args = parser.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}/v1/audio/speech"
    body = json.dumps(
        {"input": args.text, "voice": args.voice, "response_format": "wav"},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    print(f"→ {args.text[:40]}…  音色={args.voice or '(默认)'}")
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        print("组件回了错误：", exc.read().decode("utf-8", "replace")[:300])
        return 1
    except urllib.error.URLError as exc:
        print(f"组件连不上：{exc.reason}")
        return 1
    elapsed = time.monotonic() - started
    Path(args.out).write_bytes(data)
    print(f"← {len(data) / 1024:.0f} KB wav，{elapsed:.1f} 秒 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
