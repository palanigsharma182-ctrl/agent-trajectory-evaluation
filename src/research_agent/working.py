from collections.abc import Iterable

import networkx as nx

from .models import Fixture, ModelAction


def update_online_state(requirements: list[dict], claim_assessments: list[dict], command: ModelAction,
                        observations: list[dict]) -> tuple[list[dict], list[dict]]:
    """接受模型从用户问题分解的要求，以及能在已经读到的原文中核对的判断。"""
    known = {item["id"]: item for item in requirements}
    for item in command.requirements:
        requirement = item.model_dump(mode="json")
        if item.id in known and known[item.id] != requirement:
            raise ValueError(f"要求 {item.id} 的定义已经固定")
        known[item.id] = requirement
    if not known:
        raise ValueError("真实运行的首次动作必须先从用户问题分解要求")

    texts = {item["id"]: item["text"] for obs in observations if obs["kind"] == "read" for item in obs["paragraphs"]}
    updated = {item["requirement_id"]: item for item in claim_assessments}
    seen: set[str] = set()
    for item in command.claim_assessments:
        id = item.requirement_id
        if id not in known or id in seen:
            raise ValueError(f"核验必须指向一个已知的要求，且每一步只更新它一次：{id}")
        seen.add(id)
        if item.status != "UNKNOWN" and not item.evidence:
            raise ValueError(f"核验 {id} 必须提供已读原文")
        if any(evidence.quote not in texts.get(evidence.paragraph_id, "") for evidence in item.evidence):
            raise ValueError(f"核验 {id} 的引文必须逐字来自本分支已读的段落")
        updated[id] = item.model_dump(mode="json")
    return list(known.values()), list(updated.values())


def _active_checks(assessments: list[dict]) -> list[dict]:
    newest: dict[tuple[str, str, str], int] = {}
    for item in assessments:
        key = (item["claim_id"], item["assessor"], item["scope"])
        newest[key] = max(newest.get(key, 0), item["revision"])
    return [item for item in assessments
            if item["revision"] == newest[(item["claim_id"], item["assessor"], item["scope"])]]


def _support_status(statuses: set[str]) -> str:
    if "CONFLICTED" in statuses or ("REFUTED" in statuses and statuses & {"SUPPORTED", "PARTIALLY_SUPPORTED"}):
        return "CONFLICTED"
    return next((status for status in ("REFUTED", "SUPPORTED", "PARTIALLY_SUPPORTED") if status in statuses), "UNKNOWN")


def projection(fixture: Fixture, observations: list[dict], assessments: list[dict],
               submission: dict | None = None, *, requirements: list[dict] | None = None,
               claim_assessments: list[dict] | None = None) -> dict:
    """每个工作图都只使用本分支已接收的原文与核验记录。"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    available: dict[str, dict[str, dict]] = {}
    retrieved: dict[str, dict[str, dict]] = {}
    read_ids: set[str] = set()

    def node(id: str, label: str, kind: str, status: str, sources: Iterable[str] = (), reason: str = "") -> None:
        nodes[id] = {"id": id, "label": label, "kind": kind, "status": status,
                     "source_ids": sorted(set(sources)), "reason": reason}

    def link(source: str, target: str, kind: str) -> None:
        edges.append({"source": source, "target": target, "kind": kind})

    for obs in observations:
        for pid in obs.get("unavailable", []):
            if f"source:{pid}" not in nodes:
                node(f"source:{pid}", pid, "EVIDENCE", "UNAVAILABLE", [pid], "本次工具观测未提供原文。")
        if obs["kind"] == "search":
            for hit in obs["hits"]:
                if not hit.get("statement") or not hit.get("source", {}).get("quote"):
                    continue
                pid = hit["paragraph_id"]
                if pid not in read_ids:
                    node(f"source:{pid}", f"{pid} · {hit['section']}", "EVIDENCE", "EXCERPT", [pid])
                retrieved.setdefault(hit["id"], {})[pid] = hit
            continue
        if obs["kind"] != "read":
            continue
        for item in obs["paragraphs"]:
            pid = item["id"]
            read_ids.add(pid)
            node(f"source:{pid}", f"{pid} · {item['section']}", "EVIDENCE", "READ", [pid])
            for prop in item["propositions"]:
                available.setdefault(prop["id"], {})[pid] = prop

    active_checks = _active_checks(assessments)
    prop_states: dict[str, str] = {}
    for id in dict.fromkeys([*retrieved, *available]):
        anchors = available.get(id, {})
        visible = {**retrieved.get(id, {}), **anchors}
        checks = [item for item in active_checks if item["claim_id"] == id and set(item["source_ids"]) & anchors.keys()]
        prop_states[id] = _support_status({item["status"] for item in checks})
        status = prop_states[id] if anchors else "RETRIEVED"
        prop_id = f"prop:{id}"
        node(prop_id, next(iter(visible.values()))["statement"], "PROPOSITION", status, visible)
        for pid in visible:
            link(f"source:{pid}", prop_id, "GROUNDS" if pid in anchors else "EXPOSES")
        for item in checks:
            check_id = f"assessment:{item['id']}"
            seen_anchors = set(item["source_ids"]) & anchors.keys()
            node(check_id, f"离线核验 · {id}", "OFFLINE_ASSESSMENT", item["status"], seen_anchors, item["reason"])
            for pid in seen_anchors:
                link(f"source:{pid}", check_id, "CHECKED_SOURCE")
            link(check_id, prop_id, "ASSESSES")
    frontier = []
    cited = set(submission["evidence_ids"]) if submission else set()

    def add_requirement(key: str, label: str, claim_text: str | None, status: str, coverage_status: str,
                        sources: set[str], reason: str, inputs: list[tuple[str, str]], judgment: dict | None = None) -> None:
        assessment, claim, requirement, answer, coverage = (f"{kind}:{key}" for kind in
                                                           ("check", "claim", "req", "answer", "coverage"))
        node(coverage, f"证据覆盖 · {label}", "EVIDENCE_COVERAGE", coverage_status, sources,
             "覆盖反映已获得的相关原文，语义判断单独记录。")
        node(requirement, label, "REQUIREMENT", status, sources, reason)
        node(answer, f"最终答案 · {label}", "ANSWER", "NOT_EVALUATED" if submission else "PENDING",
             sources & cited, "最终答案正确性由独立 Evaluation 判断。")
        link(coverage, requirement, "EVIDENCE_AVAILABILITY")
        if claim_text is not None:
            node(assessment, f"支持判断 · {claim_text}", "RUN_ASSESSMENT", status, sources, reason)
            if judgment is not None:
                nodes[assessment]["judgment"] = judgment
            node(claim, claim_text, "ANSWER_CLAIM", status, sources, reason)
            link(assessment, claim, "ASSESSES")
            link(claim, requirement, "ADDRESSES")
        for id, relation in inputs:
            link(id, coverage, "COVERAGE_SOURCE")
            if claim_text is not None:
                link(id, assessment, relation)
        link(requirement, answer, "ANSWER_BASIS")
        if status != "SUPPORTED":
            frontier.append({"requirement_id": requirement, "question": label, "gap": reason})

    if requirements is None:
        demo = fixture.demo
        if demo is None:
            raise ValueError("Mock 需要显式的演示配置")
        for key, label, required, gap in (
            ("result", fixture.task.result_requirement, demo.result_proposition_ids, "查清两项度量、比较对象和实验范围。"),
            ("attribution", fixture.task.attribution_requirement, demo.ablation_proposition_ids,
             "查找逐项消融的设置与结果，并区分单项归因和联合效应。"),
        ):
            present = [pid for pid in required if pid in available]
            states = [prop_states.get(pid, "UNKNOWN") for pid in required]
            if all(state == "SUPPORTED" for state in states):
                status = "SUPPORTED"
            elif "CONFLICTED" in states:
                status = "CONFLICTED"
            elif "REFUTED" in states:
                status = "REFUTED"
            else:
                status = "UNKNOWN"
            coverage_status = "COVERED" if len(present) == len(required) else ("PARTIALLY_COVERED" if present else "NOT_COVERED")
            sources = {pid for prop in present for pid in available[prop]}
            reason = "Mock 指定的全部前提均有有效支持。" if status == "SUPPORTED" else gap
            claim_text = demo.result_answer if key == "result" else demo.attribution_answer
            add_requirement(key, label, claim_text, status, coverage_status, sources, reason,
                            [(f"prop:{pid}", "ALL") for pid in present])
    else:
        current = {item["requirement_id"]: item for item in claim_assessments or []}
        for item in requirements:
            judgment = current.get(item["id"])
            sources = {evidence["paragraph_id"] for evidence in judgment["evidence"]} if judgment else set()
            if judgment and not sources.issubset(read_ids):
                judgment, sources = None, set()
            add_requirement(item["id"], item["question"], judgment["claim"] if judgment else None,
                            judgment["status"] if judgment else "UNKNOWN", "COVERED" if sources else "NOT_COVERED",
                            sources, judgment["reason"] if judgment else "等待与此要求相关的已读证据及语义核验。",
                            [(f"source:{pid}", "CHECKED_SOURCE") for pid in sorted(sources)], judgment)
    return {"nodes": list(nodes.values()), "edges": edges, "frontier": frontier}


def state_changes(before: dict, after: dict) -> list[dict]:
    old = {node["id"]: node for node in before.get("nodes", [])}
    return [{"id": node["id"], "before": old.get(node["id"], {}).get("status", "NOT_OBSERVED"),
             "after": node["status"], "source_ids": node["source_ids"]}
            for node in after["nodes"] if node != old.get(node["id"])]


def compare_graphs(control: dict, intervention: dict, paragraph_id: str) -> tuple[list[dict], list[dict]]:
    """沿证据到上层结论遍历两分支使用的全部依赖，包含未改变的替代路径。"""
    graph = nx.DiGraph()
    for view in (control, intervention):
        graph.add_nodes_from(node["id"] for node in view["nodes"])
        graph.add_edges_from((edge["source"], edge["target"]) for edge in view["edges"])
    root = f"source:{paragraph_id}"
    reachable = nx.descendants(graph, root) | {root} if root in graph else set()
    left = {node["id"]: node for node in control["nodes"]}
    right = {node["id"]: node for node in intervention["nodes"]}
    impacts, preserved = [], []
    for id in sorted(left.keys() | right.keys()):
        node = right.get(id) or left[id]
        before = left.get(id, {}).get("status", "NOT_OBSERVED")
        after = right.get(id, {}).get("status", "NOT_OBSERVED")
        row = {"id": id, "label": node["label"], "kind": node["kind"], "before": before, "after": after,
               "reason": ("源段落未在该分支暴露。" if after == "NOT_OBSERVED" else node["reason"]),
               "source_ids": node["source_ids"]}
        if id in reachable:
            impacts.append(row)
        elif node["kind"] in {"PROPOSITION", "ANSWER_CLAIM", "REQUIREMENT", "EVIDENCE_COVERAGE"} and before == after:
            preserved.append(row)
    return impacts, preserved


def other_outcome_changes(control: dict, intervention: dict, direct_impacts: list[dict]) -> list[dict]:
    """报告两条轨迹实际出现、但无法沿工作图追到被屏蔽来源的状态变化。"""
    reachable = {item["id"] for item in direct_impacts}
    left = {node["id"]: node for node in control["nodes"]}
    rows = []
    for node in intervention["nodes"]:
        id = node["id"]
        if id in reachable or id not in left or node["kind"] not in {"ANSWER_CLAIM", "REQUIREMENT", "EVIDENCE_COVERAGE"}:
            continue
        baseline = left[id]
        if baseline["status"] == node["status"]:
            continue
        missing = sorted(set(baseline["source_ids"]) - set(node["source_ids"]))
        reason = node["reason"]
        if missing:
            reason += " 对照中参与该判断、在本分支支持链中缺少的段落：" + "、".join(missing) + "。"
        rows.append({"id": id, "label": node["label"], "kind": node["kind"], "before": baseline["status"],
                     "after": node["status"], "reason": reason, "source_ids": missing})
    return rows
