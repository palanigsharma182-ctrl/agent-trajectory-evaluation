import json
from pathlib import Path

import pytest

from research_agent.models import Fixture, ModelAction, OfflineReview, RunConfig
from research_agent.policy import MockClient
from research_agent.report import render_report
from research_agent.runtime import ResearchRuntime
from research_agent.store import KnowledgeStore


ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path, mode="paragraph", max_tool_calls=7):
    fixture = Fixture.model_validate_json((ROOT / "fixtures/qasper_paper.json").read_text(encoding="utf-8"))
    review = OfflineReview.model_validate_json((ROOT / "fixtures/qasper_offline_review.json").read_text(encoding="utf-8"))
    store = KnowledgeStore(tmp_path / "research.db")
    snapshot = store.publish(fixture, review.assessments)
    clients = {branch: MockClient(fixture.demo) for branch in ("original", "control", "intervention")}
    return ResearchRuntime(fixture, store, snapshot,
                           RunConfig(retrieval=mode, max_tool_calls=max_tool_calls), clients)


@pytest.mark.parametrize("mode,first_kind", [
    ("paragraph", "read"), ("unit", "search"), ("dependency", "search")])
def test_replay_forks_before_first_content_exposure_and_keeps_siblings(tmp_path, mode, first_kind):
    runtime = build(tmp_path, mode)
    target = runtime.fixture.demo.ablation_paragraph_id
    report = runtime.run_experiment(tmp_path / "checkpoints.db", lambda _: None,
                                    intervention_target=target)
    assert report["intervention"]["status"] == "INTERVENTION_APPLIED"
    assert report["intervention"]["prefix_equal"]
    original, control, masked = report["runs"]
    assert original["run_id"] == control["parent_run_id"] == masked["parent_run_id"]
    assert original["answer"] == control["answer"]
    assert original["graph"] == control["graph"]
    assert original["information_exposures"] == control["information_exposures"]
    first = report["intervention"]["first_exposure"]
    assert first and all(item["paragraph_id"] == target for item in first)
    assert first[0]["observation_kind"] == first_kind
    call_id = report["intervention"]["tool_call_id"]

    def at_fork(run):
        return next(e["observation"] for e in run["events"]
                    if e["event_type"] == "tool_observation" and e["tool_call_id"] == call_id)

    left, right = at_fork(control), at_fork(masked)
    assert left["kind"] == right["kind"] == first_kind
    if first_kind == "read":
        assert left["paragraph_ids"] == right["paragraph_ids"]
        assert right["paragraphs"] == [p for p in left["paragraphs"] if p["id"] != target]
    else:
        assert right["hits"] == [h for h in left["hits"] if h["paragraph_id"] != target]
    assert target in right["unavailable"]
    def post_observation_graph(run):
        return next(e["graph"] for e in run["events"] if e["event_type"] == "state_update"
                    and e.get("phase") == "observation" and e["step"] == first[0]["step"])

    assert post_observation_graph(control) != post_observation_graph(masked)
    assert any(e["event_type"] == "information_exposure" for e in control["events"])
    assert any(e["event_type"] == "information_exposure" for e in masked["events"])
    assert not any(row["paragraph_id"] == target and row["step"] == first[0]["step"]
                   for row in masked["information_exposures"])
    # 模型此前的请求及状态更新不能含已被遮蔽的原文。
    hidden_text = runtime.store.get_paragraph(runtime.snapshot, target).text
    hidden_props = [prop for prop in runtime.propositions if prop.paragraph_id == target]
    for event in masked["events"]:
        if event["step"] <= first[0]["step"] + 1 and event["event_type"] == "model_request":
            request = json.dumps(event["messages"], ensure_ascii=False)
            assert hidden_text not in request
            assert all(prop.quote not in request and prop.statement not in request for prop in hidden_props)


@pytest.mark.parametrize("target,budget,reason", [
    ("nonexistent-paragraph", 7, "target_not_exposed"),
    ("s11p5", 2, "target_not_exposed"),
    (None, 7, "target_not_requested"),
])
def test_inapplicable_intervention_still_saves_and_renders_original(tmp_path, target, budget, reason):
    runtime = build(tmp_path, max_tool_calls=budget)
    saved = []
    report = runtime.run_experiment(tmp_path / "checkpoints.db", saved.append,
                                    intervention_target=target)
    assert report["intervention"]["status"] == "INTERVENTION_NOT_APPLICABLE"
    assert report["intervention"]["reason"] == reason
    assert len(saved) == len(report["runs"]) == 1
    assert saved[0]["run_id"] == report["runs"][0]["run_id"]
    assert report["runs"][0]["termination_reason"] == (
        "tool_budget_exhausted" if budget == 2 else "finished")
    assert "<html" in render_report(report).lower()


def test_alternative_read_records_primary_proposition_from_actual_source(tmp_path):
    runtime = build(tmp_path)
    obs = runtime.read(["s16p0"])
    state = {"step": 2, "action": {"tool_call_id": "tool-002"}, "information_exposures": []}
    rows = runtime._observation_exposures(obs, state)
    assert {row["subject_id"] for row in rows if row["subject_type"] == "PROPOSITION"} >= {
        "m_two_way_ablation", "s_ablation_control"}
    assert all(row["paragraph_id"] == "s16p0" for row in rows)


def test_identical_quote_in_raw_read_is_exposure_without_extracted_unit(tmp_path):
    runtime = build(tmp_path)
    prop = next(item for item in runtime.propositions if item.paragraph_id == "s11p5")
    obs = {"kind": "read", "paragraphs": [{"id": "other-source", "text": prop.quote,
                                            "propositions": []}], "unavailable": []}
    state = {"step": 1, "action": {"tool_call_id": "tool-001"}, "information_exposures": []}
    rows = runtime._observation_exposures(obs, state)
    assert any(row["subject_type"] == "PROPOSITION" and row["subject_id"] == prop.id
               and row["paragraph_id"] == "other-source" for row in rows)


@pytest.mark.parametrize("read_ids,reason", [
    (["s16p0"], "target_content_already_exposed_via_alternative_source"),
    (["s11p4", "s16p0"], "target_content_in_sibling_observation"),
])
def test_alternative_source_makes_paragraph_mask_inapplicable(tmp_path, read_ids, reason):
    runtime = build(tmp_path)
    query = "two rounds of ablation testing"
    assert set(read_ids).issubset({hit["paragraph_id"] for hit in runtime.search(query)["hits"]})
    # 同批替代来源只通过原文暴露该命题，工具没有给它附带提取后的命题。
    runtime.reviewed_propositions_by_source.pop("s16p0")
    appendix, = runtime.read(["s16p0"])["paragraphs"]
    assert not any(prop["id"] == "m_two_way_ablation" for prop in appendix["propositions"])

    class AlternativeClient(MockClient):
        def choose(self, messages, step):
            if step == 1:
                return ModelAction(name="search", arguments={"query": query},
                                   action_reason="寻找消融段落。")
            if step == 2:
                return ModelAction(name="read", arguments={"paragraph_ids": read_ids},
                                   source_ids=read_ids, action_reason="读取消融证据。")
            return ModelAction(name="finish", arguments={"answer": "Evidence was found in the appendix.",
                                                         "evidence_ids": ["s16p0"]},
                               source_ids=["s16p0"], action_reason="仅陈述已经读到的证据。")

    runtime.clients = {branch: AlternativeClient(runtime.fixture.demo)
                       for branch in ("original", "control", "intervention")}
    report = runtime.run_experiment(tmp_path / "checkpoints.db", lambda _: None,
                                    intervention_target="s11p4")
    assert {event["observation"]["kind"] for event in report["runs"][0]["events"]
            if event["event_type"] == "tool_observation"} == {"search", "read", "finish"}
    assert any(row["subject_id"] == "m_two_way_ablation"
               for row in report["runs"][0]["information_exposures"])
    assert report["intervention"]["status"] == "INTERVENTION_NOT_APPLICABLE"
    assert report["intervention"]["reason"] == reason
    assert report["intervention"]["first_exposure"][0]["paragraph_id"] == read_ids[0]
    assert len(report["runs"]) == 1


def test_last_budgeted_read_has_no_followup_decision_to_replay(tmp_path):
    runtime = build(tmp_path, max_tool_calls=2)
    result = runtime.run_experiment(tmp_path / "checkpoints.db", lambda _: None,
                                    intervention_target=runtime.fixture.demo.result_paragraph_id)
    assert result["intervention"]["status"] == "INTERVENTION_NOT_APPLICABLE"
    assert result["intervention"]["reason"] == "no_followup_model_turn"
    assert result["runs"][0]["termination_reason"] == "tool_budget_exhausted"
    assert len(result["runs"]) == 1


def test_read_uses_pinned_snapshot_propositions_not_mutated_fixture(tmp_path):
    original = build(tmp_path)
    modified = original.fixture.model_copy(deep=True)
    prop = next(item for item in modified.propositions if item.paragraph_id == "s11p5")
    original_statement = prop.statement
    prop.statement = "This modified statement was never published."
    runtime = ResearchRuntime(modified, original.store, original.snapshot,
                              original.config, original.clients)
    found = runtime.read(["s11p5"])
    statements = {item["statement"] for item in found["paragraphs"][0]["propositions"]}
    assert original_statement in statements
    assert prop.statement not in statements
