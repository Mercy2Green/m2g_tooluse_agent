from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .search_and_pick import GtDemoSearchAndPick


def _result_text(result: Any) -> str:
    return f"success={result.success}; message={result.message}; data={result.data}"


def create_gt_demo_callables(env: Any, *, enable_demo_attach_fallback: bool = True) -> dict[str, Callable[..., str]]:
    demo = GtDemoSearchAndPick(env, enable_demo_attach_fallback=enable_demo_attach_fallback)

    def find_object_tool(name: str = "object") -> str:
        return _result_text(demo.find_object(name))

    def navigate_to_object_tool(name: str = "object") -> str:
        return _result_text(demo.navigate_to_object(name))

    def pick_object_tool(name: str = "object") -> str:
        return _result_text(demo.pick_object(name))

    def search_and_pick_tool(name: str = "object") -> str:
        return _result_text(demo.search_and_pick(name))

    return {
        "find_object_tool": find_object_tool,
        "navigate_to_object_tool": navigate_to_object_tool,
        "pick_object_tool": pick_object_tool,
        "search_and_pick_tool": search_and_pick_tool,
    }


def create_gt_demo_tools(env: Any, *, enable_demo_attach_fallback: bool = True) -> Any:
    """Create ROSA/LangChain-compatible tools when available.

    This function does not start Isaac Sim and does not hold a global env. If
    LangChain is unavailable, it returns plain Python callables in a dict.
    """

    callables = create_gt_demo_callables(env, enable_demo_attach_fallback=enable_demo_attach_fallback)
    try:
        from langchain_core.tools import StructuredTool
    except Exception:
        return callables

    return [
        StructuredTool.from_function(
            func=callables["find_object_tool"],
            name="find_object_gt",
            description="Find the target object using Isaac ground-truth pose.",
        ),
        StructuredTool.from_function(
            func=callables["navigate_to_object_tool"],
            name="navigate_to_grasp_standoff_gt",
            description="Move the Go2 base kinematically to a grasp standoff near the object.",
        ),
        StructuredTool.from_function(
            func=callables["pick_object_tool"],
            name="pick_object_gt",
            description="Plan and execute the gt_demo Piper pick sequence.",
        ),
        StructuredTool.from_function(
            func=callables["search_and_pick_tool"],
            name="search_and_pick",
            description="Run find, navigate, refine, fixed grasp planning, scripted pick, and verification.",
        ),
    ]
