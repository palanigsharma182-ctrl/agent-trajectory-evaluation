import json
from pathlib import Path

from research_agent.models import Fixture, OfflineReview, RunConfig
from research_agent.policy import MockClient
from research_agent.report import render_report
from research_agent.runtime import ResearchRuntime
from research_agent.store import KnowledgeStore
from research_agent.working import projection


ROOT = Path(__file__).resolve().parents[1]


def build(tmp_path, mode="paragraph"):
    fixture = Fixture.model_validate_json((ROOT / "fixtures/qasper_paper.json").read_text(encoding="utf-8"))
    review = OfflineReview.model_validate_json((ROOT / "fixtures/qasper_offline_review.json").read_text(encoding="utf-8"))
    store = KnowledgeStore(tmp_path / "research.db")
    snapshot = store.publish(fixture, review.assessments)
    clients = {branch: MockClient(fixture.demo) for branch in ("original", "control", "intervention")}
    return ResearchRuntime(fixture, store, snapshot, RunConfig(retrieval=mode), clients)


def status(run, id):
    return next(item["status"] for item in run["graph"]["nodes"] if item["id"] == id)


def test_checkpoint_replay_removes_only_the_target_observation(tmp_path):
    runtime = build(tmp_path)
    report = runtime.run_experiment(tmp_path / "checkpoints.db", lambda run: None)
    original, control, masked = report["runs"]
    intervention = report["intervention"]
    assert intervention["prefix_equal"]
    assert original["answer"] == control["answer"]
    assert control["parent_run_id"] == masked["parent_run_id"] == original["run_id"]
    assert control["checkpoint_id"] == masked["checkpoint_id"]
    assert control["termination_reason"] == "finished"
    assert masked["termination_reason"] == "finished_with_gaps"
    assert status(control, "req:attribution") == "SUPPORTED"
    assert status(masked, "req:attribution") == "UNKNOWN"
    assert status(masked, "req:result") == "SUPPORTED"
    assert "s11p5" in control["citations"] and "s11p5" not in masked["citations"]
    assert {"claim:attribution", "req:attribution", "answer:attribution"}.issubset(
        {item["id"] for item in masked["impacts"]}
    )
    assert "claim:result" in {item["id"] for item in masked["preserved"]}

    def checkpoint_observation(run):
        return next(event["observation"] for event in run["events"]
                    if event["event_type"] == "tool_observation" and event["tool_call_id"] == intervention["tool_call_id"])

    left, right = checkpoint_observation(control), checkpoint_observation(masked)
    assert left["paragraph_ids"] == right["paragraph_ids"]
    assert right["unavailable"] == ["s11p5"]
    assert right["paragraphs"] == [item for item in left["paragraphs"] if item["id"] != "s11p5"]
    masked_model_inputs = [event["messages"] for event in masked["events"] if event["event_type"] == "model_request"]
    assert runtime.store.get_paragraph(runtime.snapshot, "s11p5").text not in json.dumps(masked_model_inputs)
    page = render_report(report)
    assert 'href="#source-s11p5"' in page and 'id="source-s11p5"' in page
    assert "intervention_analysis" in page
    assert len(page.encode()) < 2_000_000


def test_reviewed_appendix_can_restate_an_identical_source_span(tmp_path):
    runtime = build(tmp_path)
    found = runtime.read(["s11p5", "s16p0"])
    assert {prop["id"] for prop in found["paragraphs"][1]["propositions"]} == {"m_two_way_ablation", "s_ablation_control"}
    assert all(prop["source"]["paragraph_id"] == "s16p0" for prop in found["paragraphs"][1]["propositions"])
    graph = projection(runtime.fixture, [found], runtime.assessments)
    assert status({"graph": graph}, "req:attribution") == "SUPPORTED"


def test_three_retrieval_modes_share_tool_and_expand_real_edge(tmp_path):
    paragraph = build(tmp_path, "paragraph")
    unit = build(tmp_path, "unit")
    dependency = build(tmp_path, "dependency")
    assert paragraph.search("augmented fitness")["hits"]
    seeds = unit.search("augmented fitness")["hits"]
    expanded = dependency.search("augmented fitness")["hits"]
    assert expanded[:len(seeds)] == seeds
    assert any(hit.get("expanded_from") and hit["id"] == "m_two_way_ablation" for hit in expanded)
