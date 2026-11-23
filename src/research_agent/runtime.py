from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
import networkx as nx
from pydantic import ValidationError

from .models import Finish, Fixture, Read, RunConfig, Search
from .policy import ModelClient, as_ai_message, model_tokens
from .store import KnowledgeStore
from .working import compare_graphs, other_outcome_changes, projection, state_changes, update_online_state


SYSTEM = (
    "You are a single research agent reading one assigned scientific paper. "
    "Choose exactly one search, read or finish action per request. Search may return passage IDs, "
    "proposition excerpts, and source quotes; read returns the full original paragraphs. "
    "Cite only passages successfully read. Read can fetch up to three IDs at once. "
    "Use targeted queries based on the question and inspect the original evidence for each claim. "
    "Use alternative evidence where available. If a remaining requirement is unsupported, finish with the supported "
    "parts and identify the gap. On your first action, state the requirements implied by the user's question. "
    "You may assess claims only with verbatim quotes from passages already read. "
    "Return the requested structured action and a short evidence-grounded reason."
)


class InformationExposure(TypedDict):
    subject_type: Literal["PARAGRAPH_TEXT", "PROPOSITION"]
    subject_id: str
    paragraph_id: str
    step: int
    tool_call_id: str
    observation_kind: Literal["search", "read"]


class RunState(TypedDict, total=False):
    run_id: str
    branch: str
    parent_run_id: str | None
    step: int
    messages: list[dict]
    events: list[dict]
    observations: list[dict]
    graph: dict
    action: dict
    pending: dict
    submission: dict | None
    termination_reason: str | None
    estimated_tokens: int
    information_exposures: list[InformationExposure]
    requirements: list[dict] | None
    claim_assessments: list[dict]


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: object) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def _event(state: RunState, event_type: str, summary: str, **details) -> dict:
    return {"step": state["step"], "event_type": event_type, "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(), **details}


class ResearchRuntime:
    def __init__(self, fixture: Fixture, store: KnowledgeStore, snapshot: str, config: RunConfig,
                 clients: dict[str, ModelClient]):
        self.fixture, self.store, self.snapshot, self.config = fixture, store, snapshot, config
        self.clients = clients
        self.propositions = store.get_propositions(snapshot)
        self.assessments = [item.model_dump(mode="json") for item in store.get_assessments(snapshot)]
        self.reviewed_propositions_by_source: dict[str, set[str]] = {}
        for assessment in self.assessments:
            for pid in assessment["source_ids"]:
                self.reviewed_propositions_by_source.setdefault(pid, set()).add(assessment["claim_id"])
        self.dependency_graph = nx.DiGraph()
        for rel in store.get_relations(snapshot):
            if rel.kind in {"REQUIRES", "DERIVED_FROM"}:
                self.dependency_graph.add_edge(rel.source_id, rel.target_id, kind=rel.kind)
            elif rel.kind == "SUPPORTS":
                self.dependency_graph.add_edge(rel.target_id, rel.source_id, kind=rel.kind)
        self.tools = {
            "search": StructuredTool.from_function(self.search, args_schema=Search, description="Search one paper."),
            "read": StructuredTool.from_function(self.read, args_schema=Read, description="Read exact paper passages."),
            "finish": StructuredTool.from_function(self.finish, args_schema=Finish, description="Submit the cited answer."),
        }

    def search(self, query: str) -> dict:
        mode = self.config.retrieval
        seeds = self.store.search(self.snapshot, query, mode="paragraph" if mode == "paragraph" else "proposition", limit=14)
        hits = [self._hit(item["id"], item["paragraph_id"], item["score"]) for item in seeds]
        if mode == "dependency":
            seen = {item["id"] for item in hits}
            for seed in seeds:
                if seed["id"] not in self.dependency_graph:
                    continue
                for neighbor in sorted(self.dependency_graph.successors(seed["id"])):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    prop = self.store.get_proposition(self.snapshot, neighbor)
                    hits.append(self._hit(neighbor, prop.paragraph_id, None, expanded_from=seed["id"],
                                          relation=self.dependency_graph[seed["id"]][neighbor]["kind"]))
        return {"kind": "search", "query": query, "hits": hits}

    def _hit(self, id: str, paragraph_id: str, score: float | None, **extra) -> dict:
        item = self.store.get_paragraph(self.snapshot, paragraph_id)
        hit = {"id": id, "paragraph_id": paragraph_id, "section": item.section, "score": score, **extra}
        if self.config.retrieval != "paragraph":
            prop = self.store.get_proposition(self.snapshot, id)
            hit["statement"] = prop.statement
            hit["source"] = prop.anchor(item).model_dump(mode="json")
        return hit

    def read(self, paragraph_ids: list[str]) -> dict:
        paragraphs = []
        for pid in dict.fromkeys(paragraph_ids):
            item = self.store.get_paragraph(self.snapshot, pid)
            props = []
            for prop in self.propositions:
                reviewed_here = prop.id in self.reviewed_propositions_by_source.get(pid, set())
                if (prop.paragraph_id == pid or reviewed_here) and prop.quote in item.text:
                    source = prop.model_copy(update={"paragraph_id": pid}).anchor(item)
                    props.append({**prop.model_dump(mode="json"), "source": source.model_dump(mode="json")})
            paragraphs.append({**item.model_dump(mode="json"), "propositions": props})
        return {"kind": "read", "paragraph_ids": paragraph_ids, "paragraphs": paragraphs, "unavailable": []}

    @staticmethod
    def finish(answer: str, evidence_ids: list[str]) -> dict:
        return {"kind": "finish", "answer": answer, "evidence_ids": list(dict.fromkeys(evidence_ids))}

    @staticmethod
    def _read_ids(state: RunState) -> set[str]:
        return {paragraph["id"] for obs in state["observations"] if obs["kind"] == "read" for paragraph in obs["paragraphs"]}

    @staticmethod
    def _seen_ids(state: RunState) -> set[str]:
        return {item["paragraph_id"] for obs in state["observations"] if obs["kind"] == "search" for item in obs["hits"]}

    def _initial(self) -> RunState:
        requirements = None if getattr(self.clients["original"], "uses_mock_fixture", False) else []
        graph = projection(self.fixture, [], self.assessments, requirements=requirements, claim_assessments=[])
        state: RunState = {
            "run_id": uuid4().hex, "branch": "original", "parent_run_id": None, "step": 0,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": self.fixture.task.question}],
            "events": [], "observations": [], "graph": graph, "termination_reason": None,
            "estimated_tokens": 0, "submission": None, "information_exposures": [],
            "requirements": requirements, "claim_assessments": [],
        }
        state["events"] = [_event(state, "run_started", "载入固定的单论文知识快照和问题。",
                                  snapshot_id=self.snapshot, model=self.clients["original"].name, graph=graph,
                                  config=self.config.model_dump())]
        return state

    def select_action(self, state: RunState) -> dict:
        if state["step"] >= self.config.max_tool_calls:
            return self._stop(state, "tool_budget_exhausted", "用完工具调用预算。")
        step = state["step"] + 1
        run = {**state, "step": step}
        context = {"remaining_tool_calls": self.config.max_tool_calls - state["step"],
                   "readable_source_ids": sorted(self._read_ids(state))}
        if state.get("requirements") is not None:
            context["task_requirements"] = state["requirements"]
        if self.config.frontier:
            context["requirements"] = [node for node in state["graph"]["nodes"] if node["kind"] == "REQUIREMENT"]
            context["unresolved_frontier"] = state["graph"]["frontier"]
        request = [*state["messages"], {"role": "user", "content": encode(context)}]
        if len(encode(request)) > self.config.max_context_chars:
            return self._stop(state, "context_budget_exhausted", "达到上下文字符预算。")
        events = [*state["events"], _event(run, "model_request", "将当前可见证据和运行状态提供给 Agent。",
                                           messages=request, remaining=context["remaining_tool_calls"])]
        try:
            command = self.clients[state["branch"]].choose(request, step)
        except (TimeoutError, ValidationError, OSError) as exc:
            return {**self._stop({**run, "events": events}, "model_error", str(exc)), "step": step}
        semantic_update = {}
        if state.get("requirements") is not None:
            try:
                requirements, claim_assessments = update_online_state(
                    state["requirements"], state["claim_assessments"], command, state["observations"])
            except ValueError as exc:
                rejected = [*events, _event(run, "model_rejected", "模型动作中的要求或核验不合法。",
                                            action=command.model_dump(mode="json"), reason=str(exc))]
                return {**self._stop({**run, "events": rejected}, "model_error", str(exc)), "step": step}
            updated_graph = projection(self.fixture, state["observations"], self.assessments,
                                       requirements=requirements, claim_assessments=claim_assessments)
            semantic_update = {"requirements": requirements, "claim_assessments": claim_assessments,
                               "graph": updated_graph}
            changes = state_changes(state["graph"], updated_graph)
            events.append(_event(run, "state_update", "Agent 更新问题要求与语义核验。", phase="assessment",
                                 changes=changes, graph=updated_graph, frontier=updated_graph["frontier"]))
        response = as_ai_message(command, step)
        call = response.tool_calls[0]
        wire = {"role": "assistant", "content": response.content, "tool_calls": [
            {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": encode(call["args"])}}
        ]}
        events.append(_event(run, "model_action", f"Agent 选择 {call['name']}。", action=command.model_dump(mode="json"),
                             tool_call_id=call["id"], trigger={"task_id": self.fixture.task.id,
                             "frontier": context.get("unresolved_frontier", []), "source_ids": command.source_ids}))
        return {"step": step, "action": {**command.model_dump(mode="json"), "tool_call_id": call["id"]},
                "messages": [*state["messages"], wire], "events": events,
                "estimated_tokens": state["estimated_tokens"] + model_tokens(request) + model_tokens([wire]),
                **semantic_update}

    @staticmethod
    def _stop(state: RunState, reason: str, message: str) -> dict:
        return {"termination_reason": reason, "events": [*state["events"], _event(state, "stop", message, reason=reason)]}

    def stage_observation(self, state: RunState) -> dict:
        action = state["action"]
        name, args = action["name"], action["arguments"]
        try:
            if not set(action["source_ids"]).issubset(self._seen_ids(state) | self._read_ids(state)):
                raise ValueError("动作理由只能引用本分支已看到的段落 ID。")
            if name == "read" and not set(Read.model_validate(args).paragraph_ids).issubset(self._seen_ids(state)):
                raise ValueError("先 search，再读取已返回的段落 ID。")
            if name == "finish" and not set(Finish.model_validate(args).evidence_ids).issubset(self._read_ids(state)):
                raise ValueError("最终引用必须指向本分支实际读到的原文。")
            observation = self.tools[name].invoke(args)
        except (ValidationError, ValueError, KeyError) as exc:
            observation = {"kind": "tool_error", "message": str(exc)}
        events = [*state["events"], _event(state, "observation_staged", "工具调用完成，结果等待交付。",
                                          tool_name=name, tool_call_id=action["tool_call_id"])]
        return {"pending": observation, "events": events}

    def _observation_exposures(self, observation: dict, state: RunState) -> list[InformationExposure]:
        """只登记进入模型消息的原文或命题；段落检索返回的 ID 不算正文曝光。"""
        seen = {(item["subject_type"], item["subject_id"])
                for item in state.get("information_exposures", [])}
        rows: list[InformationExposure] = []

        def add(subject_type: Literal["PARAGRAPH_TEXT", "PROPOSITION"], subject_id: str, paragraph_id: str) -> None:
            key = (subject_type, subject_id)
            if key in seen:
                return
            seen.add(key)
            rows.append({"subject_type": subject_type, "subject_id": subject_id,
                         "paragraph_id": paragraph_id, "step": state["step"],
                         "tool_call_id": state["action"]["tool_call_id"],
                         "observation_kind": observation["kind"]})

        if observation["kind"] == "search":
            for hit in observation["hits"]:
                if hit.get("statement") or hit.get("source", {}).get("quote"):
                    add("PROPOSITION", hit["id"], hit["paragraph_id"])
                    visible = f"{hit.get('statement', '')} {hit.get('source', {}).get('quote', '')}"
                    for prop in self.propositions:
                        if prop.quote in visible:
                            add("PROPOSITION", prop.id, hit["paragraph_id"])
        elif observation["kind"] == "read":
            for paragraph in observation["paragraphs"]:
                add("PARAGRAPH_TEXT", paragraph["id"], paragraph["id"])
                for prop in paragraph["propositions"]:
                    add("PROPOSITION", prop["id"], paragraph["id"])
                for prop in self.propositions:
                    if prop.quote in paragraph["text"]:
                        add("PROPOSITION", prop.id, paragraph["id"])
        return rows

    def apply_observation(self, state: RunState) -> dict:
        raw = state["pending"]
        action = state["action"]
        message = ToolMessage(content=encode(raw), tool_call_id=action["tool_call_id"])
        messages = [*state["messages"], {"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id}]
        observations = [*state["observations"], raw]
        submission = raw if raw["kind"] == "finish" else None
        exposures = self._observation_exposures(raw, state)
        graph = projection(self.fixture, observations, self.assessments, submission,
                           requirements=state.get("requirements"),
                           claim_assessments=state.get("claim_assessments", []))
        changes = state_changes(state["graph"], graph)
        events = [*state["events"], _event(state, "tool_observation", "工具观测进入本分支消息。",
                                          tool_call_id=action["tool_call_id"], observation=raw),
                  *([_event(state, "information_exposure", "内容首次进入本分支 Agent 可见上下文。",
                            exposures=exposures)] if exposures else []),
                  _event(state, "state_update", f"重新核对依赖，{len(changes)} 项状态或依据变化。",
                         phase="observation", changes=changes, graph=graph, frontier=graph["frontier"])]
        result = {"messages": messages, "observations": observations, "graph": graph, "events": events,
                  "information_exposures": [*state.get("information_exposures", []), *exposures]}
        if submission is not None:
            unresolved = [node for node in graph["nodes"]
                          if (node["kind"] == "REQUIREMENT" and node["status"] != "SUPPORTED")
                          or (node["kind"] == "EVIDENCE_COVERAGE" and node["status"] != "COVERED")]
            reason = "finished_with_gaps" if unresolved else "finished"
            result.update(self._stop({**state, "events": events}, reason, "Agent 提交答案及原文引用。"), submission=submission)
        return result

    def compile(self, saver: SqliteSaver):
        builder = StateGraph(RunState)
        builder.add_node("select_action", self.select_action)
        builder.add_node("stage_observation", self.stage_observation)
        builder.add_node("apply_observation", self.apply_observation)
        builder.add_edge(START, "select_action")
        builder.add_conditional_edges("select_action", lambda state: "stop" if state["termination_reason"] else "tool",
                                      {"stop": END, "tool": "stage_observation"})
        builder.add_edge("stage_observation", "apply_observation")
        builder.add_conditional_edges("apply_observation", lambda state: "stop" if state["termination_reason"] else "next",
                                      {"stop": END, "next": "select_action"})
        return builder.compile(checkpointer=saver)

    @staticmethod
    def _run_report(state: RunState, checkpoint_id: str | None = None) -> dict:
        submission = state["submission"] or {}
        branch = state["branch"]
        return {"run_id": state["run_id"], "branch": branch, "parent_run_id": state["parent_run_id"],
                "label": {"original": "原始轨迹", "control": "对照重放", "intervention": "屏蔽观测"}[branch],
                "checkpoint_id": checkpoint_id, "termination_reason": state["termination_reason"],
                "answer": submission.get("answer"), "citations": submission.get("evidence_ids", []),
                "answer_correctness": "NOT_EVALUATED", "events": state["events"], "graph": state["graph"],
                "information_exposures": state.get("information_exposures", []),
                "impacts": [], "preserved": [],
                "usage": {"tool_steps": state["step"], "estimated_tokens": state["estimated_tokens"],
                          "token_accounting": "local transcript estimate"}}

    def run_experiment(self, path: Path, save_run, *, intervention_target: str | None = None) -> dict:
        with SqliteSaver.from_conn_string(str(path)) as saver:
            graph = self.compile(saver)
            initial = self._initial()
            config = {"configurable": {"thread_id": initial["run_id"]}, "recursion_limit": 80}
            original = graph.invoke(initial, config)
            runs = [self._run_report(original)]
            save_run(runs[0])
            metadata = {"paragraph_id": intervention_target, "status": "INTERVENTION_NOT_APPLICABLE",
                        "reason": "target_not_requested" if intervention_target is None else "target_not_exposed"}

            def result() -> dict:
                return {"paper": self.fixture.paper.model_dump(exclude={"paragraphs", "abstract"}),
                        "task": self.fixture.task.model_dump(), "snapshot_id": self.snapshot,
                        "intervention": metadata, "runs": runs}

            if intervention_target is None:
                return result()
            target = intervention_target
            target_props = {prop.id for prop in self.propositions if prop.paragraph_id == target}

            def is_target(row: InformationExposure) -> bool:
                if row["subject_type"] == "PARAGRAPH_TEXT":
                    return row["subject_id"] == target
                return row["subject_id"] in target_props

            parent = None
            masked_observation = None
            for item in reversed(list(graph.get_state_history(config))):
                if item.next != ("apply_observation",):
                    continue
                new = self._observation_exposures(item.values["pending"], item.values)
                relevant = [row for row in new if is_target(row)]
                if not relevant:
                    continue
                metadata["first_exposure"] = relevant
                if any(row["paragraph_id"] != target for row in relevant):
                    metadata["reason"] = "target_content_already_exposed_via_alternative_source"
                    return result()
                pending = item.values["pending"]
                if pending["kind"] == "read":
                    masked_observation = {**pending, "paragraphs": [p for p in pending["paragraphs"] if p["id"] != target]}
                else:
                    masked_observation = {**pending, "hits": [h for h in pending["hits"] if h["paragraph_id"] != target]}
                still_exposed = self._observation_exposures(masked_observation, item.values)
                if any(is_target(row) for row in still_exposed):
                    metadata["reason"] = "target_content_in_sibling_observation"
                    return result()
                masked_observation["unavailable"] = [*pending.get("unavailable", []), target]
                if item.values["step"] >= self.config.max_tool_calls:
                    metadata["reason"] = "no_followup_model_turn"
                    return result()
                parent = item
                break
            if parent is None or masked_observation is None:
                return result()
            checkpoint_id = parent.config["configurable"]["checkpoint_id"]
            prefix_fields = sorted(set(parent.values) - {"branch", "run_id", "parent_run_id", "pending"})
            prefix_sha = fingerprint({name: parent.values[name] for name in prefix_fields})
            original_obs = parent.values["pending"]
            branches = {}
            prefix_equal = True
            for branch in ("control", "intervention"):
                pending = deepcopy(original_obs if branch == "control" else masked_observation)
                fork = graph.update_state(parent.config, {"branch": branch, "run_id": uuid4().hex,
                                          "parent_run_id": original["run_id"], "pending": pending},
                                          as_node="stage_observation")
                before = graph.get_state(fork)
                same = (before.next == parent.next and
                        fingerprint({name: before.values[name] for name in prefix_fields}) == prefix_sha)
                prefix_equal = prefix_equal and same
                if not same:
                    raise RuntimeError("重放前缀和原检查点不一致。")
                outcome = graph.invoke(None, {**fork, "recursion_limit": 80})
                report = self._run_report(outcome, checkpoint_id)
                runs.append(report)
                branches[branch] = report
                if branch == "control":
                    save_run(report)
            control, intervention = branches["control"], branches["intervention"]
            impacts, preserved = compare_graphs(control["graph"], intervention["graph"], target)
            intervention["impacts"], intervention["preserved"] = impacts, preserved
            additional = other_outcome_changes(control["graph"], intervention["graph"], impacts)
            intervention["additional_state_changes"] = additional
            step = parent.values["step"]
            branch_step = {}
            for branch in (control, intervention):
                branch_step[branch["branch"]] = next(event["graph"] for event in branch["events"]
                    if event["step"] == step and event["event_type"] == "state_update"
                    and event.get("phase") == "observation")
            immediate, _ = compare_graphs(branch_step["control"], branch_step["intervention"], target)
            metadata.update({"status": "INTERVENTION_APPLIED", "reason": None,
                             "checkpoint_id": checkpoint_id,
                             "tool_call_id": parent.values["action"]["tool_call_id"],
                             "prefix_equal": prefix_equal, "prefix_sha256": prefix_sha, "step": step,
                             "immediate_impacts": immediate,
                             "mode": "mask_one_source_in_first_exposing_tool_observation"})
            intervention["events"] = [*intervention["events"], {
                "step": intervention["usage"]["tool_steps"], "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "intervention_analysis", "summary": "追查被屏蔽证据影响的上层结论。",
                "checkpoint_id": checkpoint_id, "impacts": impacts, "preserved": preserved,
                "additional_state_changes": additional,
            }]
            save_run(intervention)
            return result()
