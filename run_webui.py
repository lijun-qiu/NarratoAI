#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Launch NarratoAI WebUI; supports multiple instances via --instance."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_BASE_PORT = 8501


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动 NarratoAI WebUI（支持多开）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run_webui.py              # 单实例，端口 8501\n"
            "  python run_webui.py -i 1         # 实例 1，端口 8501\n"
            "  python run_webui.py -i 2         # 实例 2，端口 8502\n"
            "  python run_webui.py -i dev -p 8600  # 自定义 ID 与端口\n"
        ),
    )
    parser.add_argument(
        "-i",
        "--instance",
        metavar="ID",
        help="实例编号或名称（数字 ID 自动映射端口：8501、8502…）",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="WebUI 端口（默认：单实例 8501；数字实例为 8500+ID）",
    )
    parser.add_argument(
        "streamlit_args",
        nargs="*",
        help="传递给 streamlit 的额外参数",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    env = os.environ.copy()
    if args.instance:
        env["NARRATO_INSTANCE_ID"] = str(args.instance)

    if args.port:
        env["NARRATO_PORT"] = str(args.port)
    elif args.instance and str(args.instance).isdigit() and "NARRATO_PORT" not in env:
        env["NARRATO_PORT"] = str(DEFAULT_BASE_PORT + int(args.instance) - 1)

    port = int(env.get("NARRATO_PORT", str(DEFAULT_BASE_PORT)))

    project_root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "webui.py",
        f"--server.port={port}",
        "--server.maxUploadSize=2048",
        *args.streamlit_args,
    ]

    if args.instance:
        print(f"[NarratoAI] 实例 {args.instance} → http://localhost:{port}")
    else:
        print(f"[NarratoAI] http://localhost:{port}")

    return subprocess.call(cmd, env=env, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())
