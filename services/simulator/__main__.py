"""
MaintenAI— Simulator Service Entrypoint

Usage:
    python -m services.simulator --machines 20 --rate 200
"""

from __future__ import annotations

import asyncio
import logging

import click
import httpx

from services.simulator.engine import FleetSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def run_simulation(machines: int, target_url: str, interval_sec: float) -> None:
    simulator = FleetSimulator(num_machines=machines)
    logger.info("Started Fleet Simulator with %d machines streaming to %s", machines, target_url)

    async with httpx.AsyncClient(timeout=5.0) as client:
        async for batch in simulator.stream_telemetry(interval_sec=interval_sec):
            try:
                resp = await client.post(target_url, json={"readings": batch})
                if resp.status_code not in (200, 201, 202):
                    logger.warning("Ingestion endpoint returned %s", resp.status_code)
            except Exception as e:
                logger.debug("Ingestion target not reachable: %s (simulation continuing)", e)


@click.command()
@click.option("--machines", default=10, help="Number of machines to simulate")
@click.option("--rate", default=1000, help="Interval in milliseconds between telemetry batches")
@click.option(
    "--target-url",
    default="http://localhost:8003/api/v1/telemetry/batch",
    help="Ingestion API endpoint",
)
def main(machines: int, rate: int, target_url: str) -> None:
    """Run simulator CLI."""
    interval_sec = rate / 1000.0
    asyncio.run(run_simulation(machines, target_url, interval_sec))


if __name__ == "__main__":
    main()
