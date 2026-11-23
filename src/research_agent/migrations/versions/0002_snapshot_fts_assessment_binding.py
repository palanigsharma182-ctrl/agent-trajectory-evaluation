"""Pin FTS statistics to each snapshot and bind assessments to proposition content.

Revision ID: 0002
Revises: 0001
"""

import re

from alembic import op
import sqlalchemy as sa


revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _table(kind: str, snapshot_id: str) -> str:
    if re.fullmatch(r"[0-9a-f]{24}", snapshot_id) is None:
        raise ValueError(f"unsafe snapshot ID: {snapshot_id!r}")
    return f"{kind}_fts_{snapshot_id}"


def upgrade() -> None:
    op.add_column("assessments", sa.Column("proposition_hashes", sa.Text(), nullable=False, server_default="{}"))
    connection = op.get_bind()
    ids = connection.exec_driver_sql("SELECT id FROM snapshots ORDER BY id").scalars().all()
    for snapshot_id in ids:
        paragraph_table = _table("paragraph", snapshot_id)
        proposition_table = _table("proposition", snapshot_id)
        op.execute(f"CREATE VIRTUAL TABLE {paragraph_table} USING fts5(paragraph_id UNINDEXED, text, section, tokenize='unicode61')")
        op.execute(f"CREATE VIRTUAL TABLE {proposition_table} USING fts5(proposition_id UNINDEXED, paragraph_id UNINDEXED, statement, quote, tokenize='unicode61')")
        connection.exec_driver_sql(
            f"INSERT INTO {paragraph_table} (paragraph_id, text, section) "
            "SELECT id, text, section FROM paragraphs WHERE snapshot_id = ? ORDER BY id",
            (snapshot_id,),
        )
        connection.exec_driver_sql(
            f"INSERT INTO {proposition_table} (proposition_id, paragraph_id, statement, quote) "
            "SELECT id, paragraph_id, statement, quote FROM propositions WHERE snapshot_id = ? ORDER BY id",
            (snapshot_id,),
        )
    op.execute("DROP TABLE proposition_fts")
    op.execute("DROP TABLE paragraph_fts")


def downgrade() -> None:
    op.execute("CREATE VIRTUAL TABLE paragraph_fts USING fts5(snapshot_id UNINDEXED, paper_id UNINDEXED, paragraph_id UNINDEXED, text, section, tokenize='unicode61')")
    op.execute("CREATE VIRTUAL TABLE proposition_fts USING fts5(snapshot_id UNINDEXED, paper_id UNINDEXED, proposition_id UNINDEXED, paragraph_id UNINDEXED, statement, quote, tokenize='unicode61')")
    connection = op.get_bind()
    connection.exec_driver_sql(
        "INSERT INTO paragraph_fts (snapshot_id, paper_id, paragraph_id, text, section) "
        "SELECT p.snapshot_id, s.paper_id, p.id, p.text, p.section "
        "FROM paragraphs p JOIN snapshots s ON s.id = p.snapshot_id"
    )
    connection.exec_driver_sql(
        "INSERT INTO proposition_fts (snapshot_id, paper_id, proposition_id, paragraph_id, statement, quote) "
        "SELECT p.snapshot_id, s.paper_id, p.id, p.paragraph_id, p.statement, p.quote "
        "FROM propositions p JOIN snapshots s ON s.id = p.snapshot_id"
    )
    ids = connection.exec_driver_sql("SELECT id FROM snapshots ORDER BY id").scalars().all()
    for snapshot_id in ids:
        op.execute(f"DROP TABLE {_table('proposition', snapshot_id)}")
        op.execute(f"DROP TABLE {_table('paragraph', snapshot_id)}")
    op.drop_column("assessments", "proposition_hashes")
