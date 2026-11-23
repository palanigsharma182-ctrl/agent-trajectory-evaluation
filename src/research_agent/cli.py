import argparse
import json
from pathlib import Path

from .models import Fixture, OfflineReview, RunConfig
from .policy import ExchangeClient, MockClient
from .report import render_report
from .runtime import ResearchRuntime
from .store import KnowledgeStore


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="单篇 QASPER 论文的问答与观测干预重放")
    parser.add_argument("--fixture", type=Path, default=Path("fixtures/qasper_paper.json"))
    parser.add_argument("--offline-review", type=Path, default=Path("fixtures/qasper_offline_review.json"))
    parser.add_argument("--output", type=Path, required=True, help="全新的输出目录")
    parser.add_argument("--model", choices=("mock", "exchange"), default="mock")
    parser.add_argument("--retrieval", choices=("paragraph", "unit", "dependency"), default="paragraph")
    parser.add_argument("--mask-paragraph", help="重放时屏蔽该段落的首次内容曝光")
    parser.add_argument("--no-frontier", action="store_true")
    parser.add_argument("--max-tool-calls", type=int, default=7)
    parser.add_argument("--exchange-timeout", type=float, default=300)
    args = parser.parse_args()

    data = json.loads(args.fixture.read_text(encoding="utf-8"))
    if args.model == "exchange":
        data.pop("demo", None)
        data["task"].pop("result_requirement", None)
        data["task"].pop("attribution_requirement", None)
    fixture = Fixture.model_validate(data)
    if args.model == "mock" and fixture.demo is None:
        parser.error("Mock 运行需要带 demo 的 fixture")
    intervention_target = args.mask_paragraph
    if args.model == "mock" and intervention_target is None:
        intervention_target = fixture.demo.ablation_paragraph_id
    offline = OfflineReview.model_validate_json(args.offline_review.read_text(encoding="utf-8"))
    config = RunConfig(retrieval=args.retrieval, frontier=not args.no_frontier, max_tool_calls=args.max_tool_calls)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    branches = ("original", "control", "intervention")
    if args.model == "exchange":
        # 交接目录只使用槽位号，模型请求里不注入当前分支的实验标签。
        clients = {branch: ExchangeClient(out / "exchange" / f"slot-{number}", args.exchange_timeout)
                   for number, branch in enumerate(branches, start=1)}
    else:
        clients = {branch: MockClient(fixture.demo) for branch in branches}
    store = KnowledgeStore(out / "research.db")
    try:
        snapshot_id = store.publish(fixture, offline.assessments)
        offline_data = {"snapshot_id": snapshot_id, "paper_id": fixture.paper.id, "paper_title": fixture.paper.title,
                        "paragraph_count": len(fixture.paper.paragraphs), "proposition_count": len(fixture.propositions),
                        "relation_count": len(fixture.relations), "offline_assessor": offline.assessor,
                        "assessment_count": len(offline.assessments)}
        write_json(out / "knowledge_snapshot.json", offline_data)
        print(json.dumps({"event": "knowledge_published", **offline_data}, ensure_ascii=False), flush=True)

        def save_run(run: dict) -> None:
            branch = run["branch"]
            store.record_run(snapshot_id, run)
            write_json(out / f"{branch}.json", run)
            (out / f"{branch}.trace.jsonl").write_text(
                "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in run["events"]), encoding="utf-8")
            print(json.dumps({"event": "branch_completed", "branch": branch, "run_id": run["run_id"],
                              "termination_reason": run["termination_reason"]}, ensure_ascii=False), flush=True)

        runtime = ResearchRuntime(fixture, store, snapshot_id, config, clients)
        report = runtime.run_experiment(out / "checkpoints.db", save_run, intervention_target=intervention_target)
        report["model"] = clients["original"].name
        report["knowledge"] = offline_data
        write_json(out / "experiment.json", report)
        (out / "report.html").write_text(render_report(report), encoding="utf-8")
        print(json.dumps({"event": "experiment_completed", "prefix_equal": report["intervention"].get("prefix_equal"),
                          "intervention_status": report["intervention"]["status"],
                          "report": str(out / "report.html")}, ensure_ascii=False), flush=True)
    finally:
        store.engine.dispose()


if __name__ == "__main__":
    main()
