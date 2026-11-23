import json
import re
import time
from pathlib import Path
from typing import Protocol

from langchain_core.messages import AIMessage

from .models import Demo, Finish, ModelAction, Read, Search


ARGUMENTS = {"search": Search, "read": Read, "finish": Finish}
TOOL_SCHEMAS = [
    {"name": name, "description": description, "parameters": ARGUMENTS[name].model_json_schema()}
    for name, description in (
        ("search", "Search the assigned paper. Search hits identify passages; use read to see the original text."),
        ("read", "Read the original text of up to three passage IDs returned by search."),
        ("finish", "Answer the question and list the passage IDs you actually used. Explicitly qualify unsupported parts."),
    )
]


def model_tokens(messages: list[dict]) -> int:
    """本地统一估算，避免把文件交接中的估算值写成服务端用量。"""
    return len(re.findall(r"\w+|[^\w\s]", json.dumps(messages, ensure_ascii=False)))


class ModelClient(Protocol):
    name: str

    def choose(self, messages: list[dict], step: int) -> ModelAction: ...


def as_ai_message(action: ModelAction, step: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[{
        "name": action.name, "args": action.arguments, "id": f"tool-{step:03d}", "type": "tool_call",
    }])


class MockClient:
    name = "mock"

    def __init__(self, demo: Demo):
        self.demo = demo

    def choose(self, messages: list[dict], step: int) -> ModelAction:
        state = json.loads(messages[-1]["content"])
        observed = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
        attempted = {pid for item in observed if item["kind"] == "read" for pid in item["paragraph_ids"]}
        readable = set(state["readable_source_ids"])
        matches = {item["paragraph_id"] for obs in observed if obs["kind"] == "search" for item in obs["hits"]}
        d = self.demo
        if d.result_paragraph_id not in attempted:
            if d.result_paragraph_id in matches:
                settings = [pid for pid in (d.result_paragraph_id, "s11p0", "s11p3") if pid in matches]
                return ModelAction(name="read", arguments={"paragraph_ids": settings},
                                   source_ids=settings, action_reason="阅读搜索命中的结果与试验条件。")
            return ModelAction(name="search", arguments={"query": d.result_query}, action_reason="找数量结果。")
        if d.ablation_paragraph_id not in attempted:
            if d.ablation_paragraph_id in matches:
                ids = [pid for pid in (d.ablation_paragraph_id, "s11p4") if pid in matches]
                return ModelAction(name="read", arguments={"paragraph_ids": ids}, source_ids=ids,
                                   action_reason="读取逐项消融结果与对照设置以核验归因。")
            return ModelAction(name="search", arguments={"query": d.ablation_query},
                               source_ids=sorted(readable), action_reason="当前结果无法回答单个结构变更的归因。")
        result = d.result_paragraph_id in readable
        attribution = d.ablation_paragraph_id in readable
        answer = d.result_answer if result else "The reported performance difference could not be verified."
        answer += " " + (d.attribution_answer if attribution else "The available evidence does not establish the mutation attribution.")
        return ModelAction(name="finish", arguments={"answer": answer, "evidence_ids": sorted(readable)},
                           source_ids=sorted(readable), action_reason="根据可见的结果和消融证据作答。")


class ExchangeClient:
    """由指定的 Codex 子 Agent 响应本地结构化的模型请求。"""

    name = "gpt-6.2-sol"

    def __init__(self, directory: Path, timeout: float = 240):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout

    def choose(self, messages: list[dict], step: int) -> ModelAction:
        request = self.directory / f"request-{step:03d}.json"
        response = self.directory / f"response-{step:03d}.json"
        temporary = request.with_suffix(".tmp")
        temporary.write_text(json.dumps({"step": step, "messages": messages, "tools": TOOL_SCHEMAS},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(request)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if response.is_file():
                return ModelAction.model_validate_json(response.read_text(encoding="utf-8"))
            time.sleep(0.25)
        raise TimeoutError(f"没有收到步骤 {step} 的子 Agent 响应")
