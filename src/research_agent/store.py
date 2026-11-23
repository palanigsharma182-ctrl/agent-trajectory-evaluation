"""Versioned, offline SQLite store for one-paper research snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine

from .models import Assessment, Fixture, Paragraph, Proposition, Relation


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_ids(items: list[object], label: str) -> set[str]:
    ids = [str(getattr(item, "id")) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {label} id")
    return set(ids)


class KnowledgeStore:
    """Publish and query immutable snapshots of one paper at a time."""

    def __init__(self, path: Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{self.path}")

        @event.listens_for(self.engine, "connect")
        def _enable_foreign_keys(dbapi_connection: object, _: object) -> None:
            dbapi_connection.execute("PRAGMA foreign_keys=ON")  # type: ignore[attr-defined]

        config = Config()
        config.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
        config.set_main_option("sqlalchemy.url", str(self.engine.url))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    @staticmethod
    def _validate(fixture: Fixture, assessments: list[Assessment]) -> None:
        paper = fixture.paper
        if not paper.id or not paper.source_url or not paper.source_split:
            raise ValueError("paper id, source_url and source_split are required")
        paragraph_ids = _unique_ids(paper.paragraphs, "paragraph")
        proposition_ids = _unique_ids(fixture.propositions, "proposition")
        _unique_ids(assessments, "assessment")
        paragraphs = {paragraph.id: paragraph for paragraph in paper.paragraphs}

        for proposition in fixture.propositions:
            if proposition.paragraph_id not in paragraph_ids:
                raise ValueError(f"unknown paragraph for proposition {proposition.id}")
            if proposition.quote and proposition.quote not in paragraphs[proposition.paragraph_id].text:
                raise ValueError(f"quote is absent from paragraph for proposition {proposition.id}")
        for relation in fixture.relations:
            if relation.source_id not in proposition_ids or relation.target_id not in proposition_ids:
                raise ValueError("relation refers to an unknown proposition")
        if fixture.experiment is not None:
            for identifier in fixture.experiment.setting_ids + fixture.experiment.result_ids:
                if identifier not in proposition_ids:
                    raise ValueError(f"experiment refers to unknown proposition {identifier}")
        valid_sources = paragraph_ids
        for assessment in assessments:
            if any(identifier not in proposition_ids for identifier in assessment.proposition_ids):
                raise ValueError(f"assessment {assessment.id} refers to an unknown proposition")
            if any(identifier not in valid_sources for identifier in assessment.source_ids):
                raise ValueError(f"assessment {assessment.id} refers to an unknown source")

    @staticmethod
    def _payload(fixture: Fixture, assessments: list[Assessment]) -> dict:
        paper = fixture.paper.model_dump(mode="json")
        paper["paragraphs"] = sorted(paper["paragraphs"], key=lambda row: row["id"])
        return {
            "paper": paper,
            "propositions": sorted((item.model_dump(mode="json") for item in fixture.propositions), key=lambda row: row["id"]),
            "relations": sorted((item.model_dump(mode="json") for item in fixture.relations), key=lambda row: (row["source_id"], row["target_id"], row["kind"])),
            "experiment": fixture.experiment.model_dump(mode="json") if fixture.experiment is not None else None,
            "assessments": sorted((item.model_dump(mode="json") for item in assessments), key=lambda row: row["id"]),
        }

    def publish(self, fixture: Fixture, assessments: list[Assessment]) -> str:
        """Atomically publish a paper version; identical content reuses its snapshot ID."""
        self._validate(fixture, assessments)
        payload = self._payload(fixture, assessments)
        digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()
        snapshot_id = digest[:24]
        paper = fixture.paper

        with self.engine.begin() as connection:
            inserted = connection.exec_driver_sql(
                "INSERT OR IGNORE INTO snapshots (id, content_hash, paper_id, title, abstract, source_url, source_split) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (snapshot_id, digest, paper.id, paper.title, paper.abstract, paper.source_url, paper.source_split),
            )
            if inserted.rowcount == 0:
                existing = connection.exec_driver_sql(
                    "SELECT content_hash FROM snapshots WHERE id = ?", (snapshot_id,)
                ).scalar_one_or_none()
                if existing != digest:
                    raise ValueError("snapshot ID collision")
                return snapshot_id
            self._insert_paragraphs(connection, snapshot_id, fixture)
            self._insert_propositions(connection, snapshot_id, fixture)
            for relation in fixture.relations:
                connection.exec_driver_sql(
                    "INSERT INTO relations (snapshot_id, source_id, target_id, kind) VALUES (?, ?, ?, ?)",
                    (snapshot_id, relation.source_id, relation.target_id, relation.kind),
                )
            if fixture.experiment is not None:
                experiment = fixture.experiment
                connection.exec_driver_sql(
                    "INSERT INTO experiments (snapshot_id, id, setting_ids, result_ids) VALUES (?, ?, ?, ?)",
                    (snapshot_id, experiment.id, _json(experiment.setting_ids), _json(experiment.result_ids)),
                )
            for assessment in assessments:
                connection.exec_driver_sql(
                    "INSERT INTO assessments (snapshot_id, id, claim_id, status, proposition_ids, source_ids, reason, scope, assessor, revision) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (snapshot_id, assessment.id, assessment.claim_id, assessment.status,
                     _json(assessment.proposition_ids), _json(assessment.source_ids), assessment.reason,
                     assessment.scope, assessment.assessor, assessment.revision),
                )
        return snapshot_id

    @staticmethod
    def _insert_paragraphs(connection: Connection, snapshot_id: str, fixture: Fixture) -> None:
        for paragraph in fixture.paper.paragraphs:
            connection.exec_driver_sql(
                "INSERT INTO paragraphs (snapshot_id, id, section, text) VALUES (?, ?, ?, ?)",
                (snapshot_id, paragraph.id, paragraph.section, paragraph.text),
            )
            connection.exec_driver_sql(
                "INSERT INTO paragraph_fts (snapshot_id, paper_id, paragraph_id, text, section) VALUES (?, ?, ?, ?, ?)",
                (snapshot_id, fixture.paper.id, paragraph.id, paragraph.text, paragraph.section),
            )

    @staticmethod
    def _insert_propositions(connection: Connection, snapshot_id: str, fixture: Fixture) -> None:
        for proposition in fixture.propositions:
            connection.exec_driver_sql(
                "INSERT INTO propositions (snapshot_id, id, kind, statement, paragraph_id, quote) VALUES (?, ?, ?, ?, ?, ?)",
                (snapshot_id, proposition.id, proposition.kind, proposition.statement, proposition.paragraph_id, proposition.quote),
            )
            connection.exec_driver_sql(
                "INSERT INTO proposition_fts (snapshot_id, paper_id, proposition_id, paragraph_id, statement, quote) VALUES (?, ?, ?, ?, ?, ?)",
                (snapshot_id, fixture.paper.id, proposition.id, proposition.paragraph_id, proposition.statement, proposition.quote),
            )

    def get_paragraph(self, snapshot_id: str, paragraph_id: str) -> Paragraph:
        with self.engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT id, section, text FROM paragraphs WHERE snapshot_id = ? AND id = ?",
                (snapshot_id, paragraph_id),
            ).mappings().one_or_none()
        if row is None:
            raise KeyError((snapshot_id, paragraph_id))
        return Paragraph.model_validate(dict(row))

    def get_proposition(self, snapshot_id: str, proposition_id: str) -> Proposition:
        with self.engine.connect() as connection:
            row = connection.exec_driver_sql(
                "SELECT id, kind, statement, paragraph_id, quote FROM propositions WHERE snapshot_id = ? AND id = ?",
                (snapshot_id, proposition_id),
            ).mappings().one_or_none()
        if row is None:
            raise KeyError((snapshot_id, proposition_id))
        return Proposition.model_validate(dict(row))

    def search(self, snapshot_id: str, query: str, mode: Literal["paragraph", "proposition"], limit: int = 5) -> list[dict]:
        if mode not in ("paragraph", "proposition"):
            raise ValueError("mode must be paragraph or proposition")
        if limit < 1:
            return []
        # FTS syntax is built from quoted literal terms, never from raw query operators.
        words = re.findall(r"\w+", query, flags=re.UNICODE)
        if not words:
            return []
        expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
        table = "paragraph_fts" if mode == "paragraph" else "proposition_fts"
        id_column = "paragraph_id" if mode == "paragraph" else "proposition_id"
        with self.engine.connect() as connection:
            rows = connection.exec_driver_sql(
                f"SELECT {id_column} AS id, paragraph_id, -bm25({table}) AS score "
                f"FROM {table} WHERE {table} MATCH ? AND snapshot_id = ? "
                "AND paper_id = (SELECT paper_id FROM snapshots WHERE id = ?) "
                f"ORDER BY bm25({table}), id LIMIT ?",
                (expression, snapshot_id, snapshot_id, limit),
            ).mappings().all()
        return [{"id": row["id"], "score": float(row["score"]), "paragraph_id": row["paragraph_id"]} for row in rows]

    def get_relations(self, snapshot_id: str) -> list[Relation]:
        with self.engine.connect() as connection:
            rows = connection.exec_driver_sql(
                "SELECT source_id, target_id, kind FROM relations WHERE snapshot_id = ? ORDER BY source_id, target_id, kind",
                (snapshot_id,),
            ).mappings().all()
        return [Relation.model_validate(dict(row)) for row in rows]

    def get_assessments(self, snapshot_id: str) -> list[Assessment]:
        with self.engine.connect() as connection:
            rows = connection.exec_driver_sql(
                "SELECT id, claim_id, status, proposition_ids, source_ids, reason, scope, assessor, revision "
                "FROM assessments WHERE snapshot_id = ? ORDER BY id",
                (snapshot_id,),
            ).mappings().all()
        return [Assessment.model_validate({**dict(row), "proposition_ids": json.loads(row["proposition_ids"]),
                                           "source_ids": json.loads(row["source_ids"])}) for row in rows]

    def record_run(self, snapshot_id: str, report: dict) -> None:
        """Store one completed branch report against an existing snapshot."""
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("report.run_id is required")
        with self.engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO runs (snapshot_id, run_id, report) VALUES (?, ?, ?) ",
                (snapshot_id, run_id, _json(report)),
            )
