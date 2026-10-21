"""
MaintenAI— TimescaleDB Initialization Script

Connects via asyncpg, applies schema.sql DDL, and confirms TimescaleDB
hypertable creation for real-time telemetry storage.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg

from factorypulse.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


async def init_database(dsn: str | None = None) -> bool:
    """Execute TimescaleDB schema and verify hypertable."""
    settings = get_settings()
    db_url = dsn or f"postgresql://{settings.db.user}:{settings.db.password}@{settings.db.host}:{settings.db.port}/{settings.db.db}"

    logger.info("Connecting to TimescaleDB at %s:%s/%s...", settings.db.host, settings.db.port, settings.db.db)
    conn = await asyncpg.connect(db_url)
    try:
        sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        logger.info("Executing schema DDL from %s...", SCHEMA_SQL_PATH)
        await conn.execute(sql)

        # Verify hypertable
        hypertable = await conn.fetchrow(
            "SELECT hypertable_schema, hypertable_name FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_readings';"
        )
        if hypertable:
            logger.info("Hypertable verified: %s.%s", hypertable["hypertable_schema"], hypertable["hypertable_name"])
        else:
            logger.warning("Hypertable 'sensor_readings' not listed in timescaledb_information.hypertables")

        # Verify seeded machines
        count = await conn.fetchval("SELECT count(*) FROM machines;")
        logger.info("Seeded machines total: %d", count)
        return True
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(init_database())


if __name__ == "__main__":
    main()
