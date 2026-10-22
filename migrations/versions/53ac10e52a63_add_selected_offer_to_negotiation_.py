"""Add selected offer to negotiation session."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "53ac10e52a63"
down_revision: str | None = "e462be3a3155"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("negotiation_sessions") as batch_op:
        batch_op.add_column(
            sa.Column("selected_offer_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_negotiation_sessions_selected_offer_id",
            "negotiation_offers",
            ["selected_offer_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("negotiation_sessions") as batch_op:
        batch_op.drop_constraint(
            "fk_negotiation_sessions_selected_offer_id",
            type_="foreignkey",
        )
        batch_op.drop_column("selected_offer_id")
