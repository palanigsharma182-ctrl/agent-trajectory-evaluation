from collections.abc import Iterable

import networkx as nx

from .models import DependencyGroup, Fixture


def projection(fixture: Fixture, observations: list[dict], assessments: list[dict],
               submission: dict | None = None) -> dict:
    """每个工作图都只使用本分支已接收的原文与核验记录。"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    available: dict[str, dict[str, dict]] = {}
    read_ids: set[str] = set()

    def node(id: str, label: str, kind: str, status: str, sources: Iterable[str] = (), reason: str = "") -> None:
        nodes[id] = {"id": id, "label": label, "kind": kind, "status": status,
                     "source_ids": sorted(set(sources)), "reason": reason}

    def link(source: str, target: str, kind: str) -> None:
        edges.append({"source": source, "target": target, "kind": kind})

    for obs in observations:
        if obs["kind"] != "read":
            continue
        for pid in obs.get("unavailable", []):
            if pid not in read_ids:
                node(f"source:{pid}", pid, "EVIDENCE", "UNAVAILABLE", [pid], "本次工具观测未提供原文。")
        for item in obs["paragraphs"]:
            pid = item["id"]
            read_ids.add(pid)
            node(f"source:{pid}", f"{pid} · {item['section']}", "EVIDENCE", "READ", [pid])
            for prop in item["propositions"]:
                available.setdefault(prop["id"], {})[pid] = prop

    grounded: dict[str, set[str]] = {}
    for id, anchors in available.items():
        checks = [item for item in assessments if item["claim_id"] == id and set(item["source_ids"]) & anchors.keys()]
        supported_anchors = {pid for item in checks if item["status"] == "SUPPORTED"
                             for pid in item["source_ids"] if pid in anchors}
        status = "SUPPORTED" if supported_anchors else "UNKNOWN"
        prop_id = f"prop:{id}"
        node(prop_id, next(iter(anchors.values()))["statement"], "PROPOSITION", status, anchors)
        for pid in anchors:
            link(f"source:{pid}", prop_id, "GROUNDS")
        for item in checks:
            check_id = f"assessment:{item['id']}"
            seen_anchors = set(item["source_ids"]) & anchors.keys()
            node(check_id, f"离线核验 · {id}", "OFFLINE_ASSESSMENT", item["status"], seen_anchors, item["reason"])
            for pid in seen_anchors:
                link(f"source:{pid}", check_id, "CHECKED_SOURCE")
            link(check_id, prop_id, "ASSESSES")
        if supported_anchors:
            grounded[id] = supported_anchors

    task, demo = fixture.task, fixture.demo
    # 模型能看到任务要求和已核验证据。此处仅使用 fixture 规则判定演示的依赖覆盖。
    requirements = (
        ("result", task.result_requirement, demo.result_proposition_ids,
         "查清两项度量、比较对象和实验范围。"),
        ("attribution", task.attribution_requirement, demo.ablation_proposition_ids,
         "查找逐项消融的设置与结果，并区分单项归因和联合效应。"),
    )
    frontier = []
    for key, label, required, gap in requirements:
        group = DependencyGroup(required_ids=required)
        present = [pid for pid in group.required_ids if pid in grounded]
        supported = len(present) == len(group.required_ids)
        status = "SUPPORTED" if supported else "UNKNOWN"
        sources = sorted({source for pid in present for source in grounded[pid]})
        reason = "已读原文及对应离线核验覆盖该要求的全部前提。" if supported else gap
        assessment, claim, requirement, answer = (f"{kind}:{key}" for kind in ("check", "claim", "req", "answer"))
        node(assessment, f"证据覆盖 · {label}", "RUN_ASSESSMENT", status, sources, reason)
        node(claim, label, "ANSWER_CLAIM", status, sources, reason)
        node(requirement, label, "REQUIREMENT", status, sources, reason)
        cited = set(submission["evidence_ids"]) if submission else set()
        answer_status = "PENDING"
        if submission:
            answer_status = "SUPPORTED" if supported and all(grounded[pid] & cited for pid in present) else "UNKNOWN"
        node(answer, f"最终答案依据 · {label}", "ANSWER", answer_status, set(sources) & cited, reason)
        for pid in present:
            link(f"prop:{pid}", assessment, group.operator)
        link(assessment, claim, "ASSESSES")
        link(claim, requirement, "SATISFIES")
        link(requirement, answer, "ANSWER_BASIS")
        if not supported:
            frontier.append({"requirement_id": requirement, "question": label, "gap": gap})
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
        elif node["kind"] in {"PROPOSITION", "ANSWER_CLAIM", "REQUIREMENT", "ANSWER"} and before == after:
            preserved.append(row)
    return impacts, preserved


def other_outcome_changes(control: dict, intervention: dict, direct_impacts: list[dict]) -> list[dict]:
    """报告两条轨迹实际出现、但无法沿工作图追到被屏蔽来源的状态变化。"""
    reachable = {item["id"] for item in direct_impacts}
    left = {node["id"]: node for node in control["nodes"]}
    rows = []
    for node in intervention["nodes"]:
        id = node["id"]
        if id in reachable or id not in left or node["kind"] not in {"ANSWER_CLAIM", "REQUIREMENT", "ANSWER"}:
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
