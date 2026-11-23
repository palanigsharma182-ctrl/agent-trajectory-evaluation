"""Initial immutable paper snapshots and two FTS5 indexes.

Revision ID: 0001
Revises:
"""

from alembic import op
import sqlalchemy as sa


revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "snapshots",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False, unique=True),
        sa.Column("paper_id", sa.String(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("abstract", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_split", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
    )
    op.create_index("ix_snapshots_paper_id", "snapshots", ["paper_id"])
    op.create_table(
        "paragraphs",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("section", sa.Text(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "id"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "propositions",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("paragraph_id", sa.String(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "id"),
        sa.ForeignKeyConstraint(["snapshot_id", "paragraph_id"], ["paragraphs.snapshot_id", "paragraphs.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "relations",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "source_id", "target_id", "kind"),
        sa.ForeignKeyConstraint(["snapshot_id", "source_id"], ["propositions.snapshot_id", "propositions.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["snapshot_id", "target_id"], ["propositions.snapshot_id", "propositions.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "experiments",
        sa.Column("snapshot_id", sa.String(), primary_key=True),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("setting_ids", sa.Text(), nullable=False),
        sa.Column("result_ids", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "assessments",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("claim_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("proposition_ids", sa.Text(), nullable=False),
        sa.Column("source_ids", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("assessor", sa.String(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "id"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
    )
    op.create_table(
        "runs",
        sa.Column("snapshot_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("report", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("snapshot_id", "run_id"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["snapshots.id"], ondelete="CASCADE"),
    )
    op.execute("CREATE VIRTUAL TABLE paragraph_fts USING fts5(snapshot_id UNINDEXED, paper_id UNINDEXED, paragraph_id UNINDEXED, text, section, tokenize='unicode61')")
    op.execute("CREATE VIRTUAL TABLE proposition_fts USING fts5(snapshot_id UNINDEXED, paper_id UNINDEXED, proposition_id UNINDEXED, paragraph_id UNINDEXED, statement, quote, tokenize='unicode61')")


def downgrade() -> None:
    op.execute("DROP TABLE proposition_fts")
    op.execute("DROP TABLE paragraph_fts")
    op.drop_table("runs")
    op.drop_table("assessments")
    op.drop_table("experiments")
    op.drop_table("relations")
    op.drop_table("propositions")
    op.drop_table("paragraphs")
    op.drop_index("ix_snapshots_paper_id", table_name="snapshots")
    op.drop_table("snapshots")
