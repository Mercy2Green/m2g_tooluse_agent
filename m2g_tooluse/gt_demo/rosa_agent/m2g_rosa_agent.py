from __future__ import annotations

"""ROSA agent for controlling gt_demo through ROS2 services.

Run in the ``rosa`` conda environment after sourcing ROS2 Humble.
"""

import argparse
import os
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rosa import ROSA

from m2g_tooluse.gt_demo.rosa_agent.prompts import make_gt_demo_prompts
from m2g_tooluse.gt_demo.rosa_agent.tools import (
    create_gt_demo_rosa_tools,
    get_gt_demo_status,
    list_gt_demo_services,
    run_full_gt_demo,
)


DEFAULT_TEST_PROMPTS = [
    "列出当前 ROS2 nodes 和 services。",
    "检查 gt_demo 的 service 是否存在。",
    "启动 gt_demo。",
    "让 Go2 移动到物体旁边。",
    "用 Piper 抓取物体。",
    "完整运行 gt_demo。",
    "用训练好的 RL locomotion policy 让 Go2 向前移动。",
    "运行一次 locomotion policy test。",
    "查询 locomotion test 后的状态。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROSA agent for M2G gt_demo.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--prompt", default=None, help="Run one prompt and exit.")
    parser.add_argument("--self-test-tools", action="store_true", help="List gt_demo services without invoking an LLM.")
    parser.add_argument(
        "--self-test-run-full-demo",
        action="store_true",
        help="Run gt_demo service tools end-to-end without creating an LLM.",
    )
    parser.add_argument("--run-scripted-prompts", action="store_true", help="Run the documented natural-language smoke prompts.")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def create_agent(model: str, verbose: bool = False, base_url: str | None = None) -> ROSA:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY or OPENAI_ADMIN_KEY is required for ROSA natural-language mode. "
            "Use --self-test-tools or --self-test-run-full-demo to verify ROS2 tools without an LLM."
        )
    llm_kwargs = {"model": model, "temperature": 0.0, "api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)
    return ROSA(
        ros_version=2,
        llm=llm,
        tools=create_gt_demo_rosa_tools(),
        prompts=make_gt_demo_prompts(),
        verbose=verbose,
        streaming=False,
        max_iterations=20,
    )


def run_self_test_full_demo() -> None:
    print("=== list gt_demo services ===")
    print(list_gt_demo_services())
    print("=== run full gt_demo ===")
    print(run_full_gt_demo())
    print("=== wait for IsaacLab controller ===")
    time.sleep(5.0)
    print("=== status ===")
    print(get_gt_demo_status())


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.self_test_tools:
        print(list_gt_demo_services())
        return
    if args.self_test_run_full_demo:
        run_self_test_full_demo()
        return

    agent = create_agent(args.model, verbose=args.verbose, base_url=args.base_url)
    if args.prompt:
        print(agent.invoke(args.prompt))
        return
    if args.run_scripted_prompts:
        for prompt in DEFAULT_TEST_PROMPTS:
            print(f"\nUSER> {prompt}")
            print(agent.invoke(prompt))
        return

    print("M2G gt_demo ROSA agent ready. Type Ctrl-D or Ctrl-C to exit.")
    while True:
        try:
            query = input("ROSA> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        print(agent.invoke(query))


if __name__ == "__main__":
    main()
