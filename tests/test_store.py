import json
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError

import research_agent.store as store_module
from research_agent.models import Assessment, Fixture, Paragraph, Proposition
from research_agent.store import KnowledgeStore


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "qasper_paper.json"


def _fixture() -> Fixture:
    return Fixture.model_validate(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _assessment(fixture: Fixture | None = None) -> Assessment:
    fixture = fixture or _fixture()
    proposition = next(item for item in fixture.propositions if item.id == "r_mobile_bleu")
    return Assessment(
        id="review-1",
        claim_id="r_mobile_bleu",
        status="SUPPORTED",
        proposition_ids=["r_mobile_bleu"],
        proposition_hashes={"r_mobile_bleu": proposition.content_hash()},
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
    next_id = store.publish(revised, [_assessment(revised)])
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
        assert database.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'paragraph_fts_%'").fetchone()[0] == 0
        assert database.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'proposition_fts_%'").fetchone()[0] == 0


def test_assessment_must_bind_the_current_proposition_content(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    fixture = _fixture()
    review = _assessment(fixture)
    revised_proposition = next(item for item in fixture.propositions if item.id == review.claim_id).model_copy(
        update={"statement": "A revised interpretation of the same paragraph"}
    )
    revised = fixture.model_copy(update={"propositions": [
        revised_proposition if item.id == revised_proposition.id else item for item in fixture.propositions
    ]})
    with pytest.raises(ValueError, match="stale proposition hashes"):
        store.publish(revised, [review])
    with pytest.raises(ValueError, match="missing or stale proposition hashes"):
        store.publish(fixture, [review.model_copy(update={"proposition_hashes": {}})])
    with pytest.raises(ValueError, match="unknown claim"):
        store.publish(fixture, [review.model_copy(update={"claim_id": "missing"})])
    with pytest.raises(ValueError, match="missing or stale proposition hashes"):
        store.publish(fixture, [review.model_copy(update={"proposition_ids": [
            "r_mobile_bleu", "r_big_flops"
        ]})])

    revised_review = review.model_copy(update={"proposition_hashes": {
        revised_proposition.id: revised_proposition.content_hash()
    }})
    snapshot_id = store.publish(revised, [revised_review])
    assert store.get_assessments(snapshot_id) == [revised_review]
    with sqlite3.connect(store.path) as database:
        database.execute(
            "UPDATE assessments SET proposition_hashes = '{}' WHERE snapshot_id = ? AND id = ?",
            (snapshot_id, revised_review.id),
        )
    assert store.get_assessments(snapshot_id) == []


def test_old_snapshot_bm25_scores_are_stable_after_new_snapshot(tmp_path: Path) -> None:
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    fixture = _fixture()
    first = store.publish(fixture, [_assessment(fixture)])
    before = {mode: store.search(first, "BLEU", mode, 50) for mode in ("paragraph", "proposition")}
    assert before["paragraph"] and before["proposition"]

    # New documents have an extremely different BLEU frequency. FTS5 must not
    # change the IDF, scores, or ranking of the original snapshot.
    added_paragraph = Paragraph(id="newp0", section="Later edition", text="BLEU " * 100)
    added_proposition = Proposition(
        id="new_result", kind="RESULT", statement="BLEU " * 60,
        paragraph_id="newp0", quote="BLEU",
    )
    revised = fixture.model_copy(update={
        "paper": fixture.paper.model_copy(update={"paragraphs": fixture.paper.paragraphs + [added_paragraph]}),
        "propositions": fixture.propositions + [added_proposition],
    })
    second = store.publish(revised, [_assessment(revised)])
    assert second != first
    assert store.search(second, "BLEU", "paragraph", 50)
    assert store.search(second, "BLEU", "proposition", 50)
    assert {mode: store.search(first, "BLEU", mode, 50) for mode in before} == before
    reopened = KnowledgeStore(store.path)
    assert {mode: reopened.search(first, "BLEU", mode, 50) for mode in before} == before


def test_migration_rebuilds_legacy_fts_and_does_not_trust_unbound_review(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    config = Config()
    config.set_main_option("script_location", str(Path(store_module.__file__).parent / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{path}")
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "0001")
    engine.dispose()
    snapshot_id = "a" * 24
    with sqlite3.connect(path) as database:
        database.execute(
            "INSERT INTO snapshots (id, content_hash, paper_id, title, abstract, source_url, source_split) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (snapshot_id, "b" * 64, "paper", "Title", "Abstract", "source", "train"),
        )
        database.execute("INSERT INTO paragraphs VALUES (?, ?, ?, ?)", (snapshot_id, "p1", "Results", "BLEU gain"))
        database.execute(
            "INSERT INTO propositions VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_id, "r1", "RESULT", "BLEU gain", "p1", "BLEU gain"),
        )
        database.execute(
            "INSERT INTO assessments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (snapshot_id, "review-1", "r1", "SUPPORTED", '["r1"]', '["p1"]', "old review", "paper", "offline", 1),
        )
        database.execute(
            "INSERT INTO paragraph_fts (snapshot_id, paper_id, paragraph_id, text, section) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, "paper", "p1", "BLEU gain", "Results"),
        )
        database.execute(
            "INSERT INTO proposition_fts (snapshot_id, paper_id, proposition_id, paragraph_id, statement, quote) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_id, "paper", "r1", "p1", "BLEU gain", "BLEU gain"),
        )
    store = KnowledgeStore(path)
    assert [row["id"] for row in store.search(snapshot_id, "BLEU", "paragraph")] == ["p1"]
    assert [row["id"] for row in store.search(snapshot_id, "BLEU", "proposition")] == ["r1"]
    assert store.get_assessments(snapshot_id) == []
    with sqlite3.connect(path) as database:
        assert database.execute("SELECT proposition_hashes FROM assessments").fetchone()[0] == "{}"
        assert database.execute("SELECT count(*) FROM sqlite_master WHERE name = 'paragraph_fts'").fetchone()[0] == 0
