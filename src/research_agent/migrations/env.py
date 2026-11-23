"""Alembic environment for the embedded knowledge store."""

from alembic import context
from sqlalchemy import engine_from_config, pool


def run_migrations_online() -> None:
    configuration = context.config
    supplied_connection = configuration.attributes.get("connection")
    if supplied_connection is not None:
        context.configure(connection=supplied_connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()
        return

    engine = engine_from_config(
        configuration.get_section(configuration.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
