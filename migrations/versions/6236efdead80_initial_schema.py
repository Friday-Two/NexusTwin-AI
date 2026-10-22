"""Initial database schema.

Revision ID: 6236efdead80
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "6236efdead80"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "services",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_services_name", "services", ["name"])
    op.create_index("ix_services_category", "services", ["category"])

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("preference_key", sa.String(length=100), nullable=False),
        sa.Column("preference_value", sa.Text(), nullable=False),
        sa.Column("is_sensitive", sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_user_preferences_user_id", "user_preferences", ["user_id"])

    op.create_table(
        "negotiation_sessions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("service_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("budget_limit", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["service_id"], ["services.id"]),
    )
    op.create_index(
        "ix_negotiation_sessions_status",
        "negotiation_sessions",
        ["status"],
    )

    op.create_table(
        "negotiation_offers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("agent_role", sa.String(length=30), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["negotiation_sessions.id"]),
    )
    op.create_index(
        "ix_negotiation_offers_session_id",
        "negotiation_offers",
        ["session_id"],
    )

    op.create_table(
        "execution_records",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("external_reference", sa.String(length=255), nullable=True),
        sa.Column("result_details", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["negotiation_sessions.id"]),
        sa.UniqueConstraint("session_id"),
    )

    op.create_table(
        "verification_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("execution_id", sa.Integer(), nullable=False),
        sa.Column("verified", sa.Boolean(), nullable=False),
        sa.Column("details", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["execution_id"], ["execution_records.id"]),
    )
    op.create_index(
        "ix_verification_results_execution_id",
        "verification_results",
        ["execution_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_verification_results_execution_id", table_name="verification_results")
    op.drop_table("verification_results")
    op.drop_table("execution_records")
    op.drop_index("ix_negotiation_offers_session_id", table_name="negotiation_offers")
    op.drop_table("negotiation_offers")
    op.drop_index("ix_negotiation_sessions_status", table_name="negotiation_sessions")
    op.drop_table("negotiation_sessions")
    op.drop_index("ix_user_preferences_user_id", table_name="user_preferences")
    op.drop_table("user_preferences")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
    op.drop_index("ix_services_category", table_name="services")
    op.drop_index("ix_services_name", table_name="services")
    op.drop_table("services")
