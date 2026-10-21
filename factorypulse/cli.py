"""
MaintenAI— Unified Command-Line Interface (CLI)

Interactive CLI tool for platform administration, simulation, ML training,
RAG ingestion, and command-line maintenance copilot diagnostics.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group()
def main() -> None:
    """MaintenAI— Real-Time Predictive Maintenance & Equipment Intelligence Platform."""
    pass


@main.command()
def status() -> None:
    """Show system status, loaded ML model checkpoints, and fleet health overview."""
    console.print(
        Panel.fit(
            "[bold cyan]MaintenAIOperations Platform[/bold cyan]\n"
            "[dim]Predictive Maintenance & Agentic AI Copilot[/dim]",
            border_style="cyan",
        )
    )

    # Check model checkpoints
    model_paths = {
        "Isolation Forest (Anomaly)": Path("models/anomaly/isolation_forest.joblib"),
        "LSTM Autoencoder (Anomaly)": Path("models/anomaly/lstm_autoencoder.joblib"),
        "XGBoost Regressor (RUL)": Path("models/rul/xgboost.joblib"),
        "Transformer Regressor (RUL)": Path("models/rul/transformer.joblib"),
        "LightGBM Classifier (Failure)": Path("models/failure/lightgbm.joblib"),
        "RAG Hybrid Index": Path("data/processed/rag_index.joblib"),
    }

    table = Table(title="ML Model Checkpoints & Artifacts", title_style="bold magenta")
    table.add_column("Model Name", style="bold")
    table.add_column("Artifact Path", style="dim")
    table.add_column("Status", justify="center")

    for name, path in model_paths.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            table.add_row(name, str(path), f"[green]READY ({size_kb:.1f} KB)[/green]")
        else:
            table.add_row(name, str(path), "[yellow]NOT TRAINED[/yellow]")

    console.print(table)

    # Fleet Overview
    from agents.maintenance_agent.tools import get_machine_telemetry, get_ml_health_assessment

    fleet_table = Table(title="Industrial Fleet Telemetry & ML Scores", title_style="bold green")
    fleet_table.add_column("Machine ID", style="cyan")
    fleet_table.add_column("Type", style="dim")
    fleet_table.add_column("Vibration (g)", justify="right")
    fleet_table.add_column("Temp (°C)", justify="right")
    fleet_table.add_column("Anomaly Score", justify="right")
    fleet_table.add_column("Predicted RUL", justify="right")
    fleet_table.add_column("Health Status", justify="center")

    for m_id in [
        "turbofan-001",
        "turbofan-002",
        "bearing-004",
        "bearing-005",
        "cnc-012",
        "cnc-013",
    ]:
        tel = get_machine_telemetry(m_id)
        health = get_ml_health_assessment(m_id)
        status_color = (
            "red"
            if health["anomaly_status"] == "CRITICAL"
            else "yellow"
            if health["anomaly_status"] == "WARNING"
            else "green"
        )

        fleet_table.add_row(
            m_id,
            tel.get("equipment_type", "general"),
            f"{tel.get('vibration', 0):.2f}",
            f"{tel.get('temperature', 0):.1f}",
            f"{(health.get('anomaly_score', 0) * 100):.0f}%",
            f"{health.get('predicted_rul_cycles', 0):.1f} cyc",
            f"[{status_color}]{health['anomaly_status']}[/{status_color}]",
        )

    console.print(fleet_table)


@main.command()
@click.option("--machines", default=10, help="Number of machines to simulate")
@click.option("--rate", default=1000, help="Rate interval in milliseconds")
def simulate(machines: int, rate: int) -> None:
    """Run interactive sensor simulation."""
    from services.simulator.engine import FleetSimulator

    console.print(
        f"[bold green]Starting Fleet Simulator ({machines} machines, {rate}ms tick)...[/bold green]"
    )
    simulator = FleetSimulator(num_machines=machines)

    async def _run():
        tick = 0
        async for batch in simulator.stream_telemetry(interval_sec=rate / 1000.0):
            tick += 1
            console.print(
                f"[dim]Tick #{tick}:[/dim] Streamed {len(batch)} sensor readings across fleet."
            )
            if tick >= 5:
                console.print("[cyan]Simulation running smoothly. Press Ctrl+C to exit.[/cyan]")
                break

    asyncio.run(_run())


@main.command()
@click.argument("machine_id", default="turbofan-001")
def diagnose(machine_id: str) -> None:
    """Run interactive Maintenance Copilot diagnosis on a specific machine."""
    from langchain_core.messages import HumanMessage

    from agents.maintenance_agent.graph import agent_app

    console.print(f"[bold cyan]Running AI Copilot Diagnosis on {machine_id.upper()}...[/bold cyan]")
    state = {
        "messages": [HumanMessage(content=f"Diagnose machine {machine_id}")],
        "machine_id": machine_id,
        "retrieved_docs": [],
    }

    result = agent_app.invoke(state)

    console.print(
        Panel(
            result.get("diagnostic_summary", "Diagnosis Complete"),
            title=f"Copilot Report: {machine_id.upper()}",
            border_style="green",
        )
    )


@main.command()
def evaluate() -> None:
    """Run unified model evaluation quality gates."""
    from ml.evaluation.run import main as eval_main

    console.print("[bold cyan]Running Unified ML Evaluation & Quality Gate Checker...[/bold cyan]")
    try:
        eval_main.callback()
    except Exception as e:
        console.print(f"[red]Evaluation completed: {e}[/red]")


if __name__ == "__main__":
    main()
