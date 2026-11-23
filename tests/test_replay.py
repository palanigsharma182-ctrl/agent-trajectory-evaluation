import json
from pathlib import Path

from research_agent.models import Fixture, ModelAction, OfflineReview, RunConfig, Task
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
    report = runtime.run_experiment(tmp_path / "checkpoints.db", lambda run: None, intervention_target="s11p5")
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


def test_revised_assessment_supersedes_old_support_and_independent_disagreement_is_kept(tmp_path):
    runtime = build(tmp_path)
    evidence = [runtime.read(["s11p4", "s11p5", "s16p0"])]
    original = next(item for item in runtime.assessments if item["claim_id"] == "m_two_way_ablation")
    revision = {**original, "id": "revision:ablation", "revision": 2,
                "status": "REFUTED", "source_ids": ["s11p4"], "reason": "Updated assessment"}
    independent = {**original, "id": "independent:ablation", "assessor": "another-reviewer",
                   "source_ids": ["s16p0"], "reason": "Independent assessment"}

    revised = {"graph": projection(runtime.fixture, evidence, [*runtime.assessments, revision])}
    assert status(revised, "prop:m_two_way_ablation") == "REFUTED"
    assert status(revised, "req:attribution") != "SUPPORTED"
    assert f"assessment:{original['id']}" not in {node["id"] for node in revised["graph"]["nodes"]}

    disputed = {"graph": projection(runtime.fixture, evidence, [*runtime.assessments, revision, independent])}
    assert status(disputed, "prop:m_two_way_ablation") == "CONFLICTED"
    assert status(disputed, "req:attribution") == "CONFLICTED"
    checks = {node["id"]: node["source_ids"] for node in disputed["graph"]["nodes"]}
    assert checks["assessment:revision:ablation"] == ["s11p4"]
    assert checks["assessment:independent:ablation"] == ["s16p0"]


class QuestionClient:
    name = "scripted-online"
    uses_mock_fixture = False

    def __init__(self, question, *, support="SUPPORTED"):
        self.question = question
        self.support = support

    def choose(self, messages, step):
        command = {"action_reason": "Check the result relevant to the question."}
        state = json.loads(messages[-1]["content"])
        requirement = {"id": "bleu-gain", "question": self.question}
        if step == 1:
            assert messages[1]["content"] == self.question
            assert state["task_requirements"] == []
            return ModelAction(**command, name="search", arguments={"query": "BLEU FLOPS"}, requirements=[requirement])
        assert state["task_requirements"] == [requirement]
        if step == 2:
            if "unresolved_frontier" in state:
                assert state["unresolved_frontier"][0]["requirement_id"] == "req:bleu-gain"
            return ModelAction(**command, name="read", arguments={"paragraph_ids": ["s11p1"]})
        observation = json.loads(messages[-2]["content"])
        if not observation["paragraphs"]:
            return ModelAction(**command, name="finish", arguments={"answer": "Evidence unavailable.", "evidence_ids": []})
        paragraph = observation["paragraphs"][0]
        prop = next(item for item in paragraph["propositions"] if item["id"] == "r_mobile_bleu")
        assessment = {"requirement_id": requirement["id"], "claim": prop["statement"], "status": self.support,
                      "evidence": [{"paragraph_id": paragraph["id"], "quote": prop["quote"]}],
                      "reason": "Assessment anchored to the paragraph."}
        return ModelAction(**command, name="finish", arguments={"answer": "The improvement was 9.7 BLEU.",
                                                               "evidence_ids": [paragraph["id"]]},
                           claim_assessments=[assessment])


def test_online_requirements_and_grounding_follow_only_branch_observations(tmp_path):
    setup = build(tmp_path)
    question = "What BLEU improvement was observed for mobile translation?"
    fixture = setup.fixture.model_copy(update={"task": Task(id="mobile-gain", question=question), "demo": None})
    clients = {branch: QuestionClient(question) for branch in ("original", "control", "intervention")}
    runtime = ResearchRuntime(fixture, setup.store, setup.snapshot, RunConfig(), clients)
    report = runtime.run_experiment(tmp_path / "checkpoints.db", lambda run: None, intervention_target="s11p1")
    original, control, masked = report["runs"]
    assert report["intervention"]["prefix_equal"]
    assert original["graph"] == control["graph"]
    assert original["information_exposures"] == control["information_exposures"]
    assert status(control, "req:bleu-gain") == "SUPPORTED"
    assert status(masked, "req:bleu-gain") == "UNKNOWN"
    assert status(control, "coverage:bleu-gain") == "COVERED"
    assert status(masked, "coverage:bleu-gain") == "NOT_COVERED"
    assert report["intervention"]["immediate_impacts"]
    assert "claim:bleu-gain" in {item["id"] for item in masked["impacts"]}
    assert not any(node["id"] in {"req:result", "req:attribution", "claim:bleu-gain"} for node in masked["graph"]["nodes"])
    assert not any(item["paragraph_id"] == "s11p1" for item in masked["information_exposures"])
    assert control["citations"] == ["s11p1"] and masked["citations"] == []
    assert control["answer"] == "The improvement was 9.7 BLEU."
    assert status(control, "answer:bleu-gain") == control["answer_correctness"] == "NOT_EVALUATED"


def test_fetched_evidence_can_leave_claim_support_unknown(tmp_path):
    setup = build(tmp_path)
    question = "What BLEU improvement was observed for mobile translation?"
    fixture = setup.fixture.model_copy(update={"task": Task(id="mobile-gain", question=question), "demo": None})
    model = QuestionClient(question, support="UNKNOWN")
    runtime = ResearchRuntime(fixture, setup.store, setup.snapshot, RunConfig(frontier=False), {"original": model})
    original, = runtime.run_experiment(tmp_path / "checkpoints.db", lambda run: None)["runs"]
    assert status(original, "coverage:bleu-gain") == "COVERED"
    assert status(original, "claim:bleu-gain") == "UNKNOWN"
    assert original["answer_correctness"] == "NOT_EVALUATED"
    assert original["termination_reason"] == "finished_with_gaps"


def test_online_assessment_rejects_quote_absent_from_the_read_paragraph(tmp_path):
    setup = build(tmp_path)
    question = "What BLEU improvement was observed for mobile translation?"
    fixture = setup.fixture.model_copy(update={"task": Task(id="mobile-gain", question=question), "demo": None})

    class UnverifiableClient(QuestionClient):
        def choose(self, messages, step):
            action = super().choose(messages, step)
            if step == 3:
                action.claim_assessments[0].evidence[0].quote = "The improvement was 9.7 BLEU."
            return action

    runtime = ResearchRuntime(fixture, setup.store, setup.snapshot, RunConfig(), {"original": UnverifiableClient(question)})
    original, = runtime.run_experiment(tmp_path / "checkpoints.db", lambda run: None)["runs"]
    assert original["termination_reason"] == "model_error"
    assert status(original, "req:bleu-gain") == "UNKNOWN"
    assert original["answer"] is None
