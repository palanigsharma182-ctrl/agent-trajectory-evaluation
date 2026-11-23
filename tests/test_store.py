import json
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from research_agent.models import Assessment, Fixture
from research_agent.store import KnowledgeStore


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "qasper_paper.json"


def _fixture() -> Fixture:
    return Fixture.model_validate(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _assessment() -> Assessment:
    return Assessment(
        id="review-1",
        claim_id="claim-1",
        status="SUPPORTED",
        proposition_ids=["r_mobile_bleu"],
        source_ids=["s11p1"],
        reason="Direct result evidence",
        scope="paper",
        assessor="offline",
    )


def test_publish_search_and_version(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    fixture = _fixture()
    snapshot_id = store.publish(fixture, [_assessment()])
    assert store.publish(fixture, [_assessment()]) == snapshot_id
    assert store.get_paragraph(snapshot_id, "s11p1").id == "s11p1"
    assert store.get_proposition(snapshot_id, "r_mobile_bleu").paragraph_id == "s11p1"
    assert store.get_assessments(snapshot_id) == [_assessment()]
    assert len(store.get_relations(snapshot_id)) == len(fixture.relations)
    assert any(row["id"] == "s11p1" for row in store.search(snapshot_id, "BLEU", "paragraph", 50))
    assert any(row["id"] == "r_mobile_bleu" for row in store.search(snapshot_id, "BLEU", "proposition", 50))
    assert store.search("missing", "BLEU", "paragraph") == []

    revised = fixture.model_copy(update={"paper": fixture.paper.model_copy(update={"title": "Revised title"})})
    next_id = store.publish(revised, [_assessment()])
    assert next_id != snapshot_id
    with sqlite3.connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 2

    store.record_run(snapshot_id, {"run_id": "branch-1", "events": [], "answer": "done"})
    with sqlite3.connect(store.path) as database:
        report = database.execute("SELECT report FROM runs WHERE run_id='branch-1'").fetchone()[0]
    assert json.loads(report)["answer"] == "done"
    assert KnowledgeStore(store.path).get_assessments(snapshot_id) == [_assessment()]


def test_invalid_reference_and_failed_insert_leave_no_snapshot(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    fixture = _fixture()
    invalid = _assessment().model_copy(update={"source_ids": ["not-a-paragraph"]})
    with pytest.raises(ValueError, match="unknown source"):
        store.publish(fixture, [invalid])

    duplicate = fixture.model_copy(update={"relations": fixture.relations + fixture.relations[:1]})
    with pytest.raises(IntegrityError):
        store.publish(duplicate, [_assessment()])
    with sqlite3.connect(store.path) as database:
        assert database.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM paragraph_fts").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM proposition_fts").fetchone()[0] == 0
