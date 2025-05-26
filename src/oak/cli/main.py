"""
Main entry point for the OAK CLI.
Simplified version for debugging, with all logic in one place.
"""
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

# Import business logic from other OAK modules
from oak.analysis import analyze_model, ModelAnalysisError
from oak.knowledge_base import KnowledgeBase, KnowledgeBaseError
from oak.advisor.heuristic_engine import HeuristicAdvisor

# Create the main Typer application
app = typer.Typer(
    name="oak",
    help="OAK: Your optimization advisor for AI on the Edge.",
    add_completion=False,
    no_args_is_help=True,
)

# --- Constants and Console ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"
console = Console()


# --- Command Definition Directly in the Main App ---
@app.command()
def advise(
    model_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, help="Path to the ONNX model file.")],
    hardware: Annotated[str, typer.Option("--hardware", "-h", help="Identifier of the target hardware (e.g., 'esp32-s3').")] = "esp32-s3",
    user_priority: Annotated[str, typer.Option(help="User's optimization priority: 'latency', 'energy', or 'size'.")] = "latency"
):
    """
    Analyzes an ML model and recommends optimization strategies for a target hardware.
    """
    try:
        # Validate user priority
        valid_priorities = ["latency", "energy", "size"]
        if user_priority not in valid_priorities:
            console.print(f"[bold red]Error:[/bold red] Invalid user priority '{user_priority}'. Choose from: {', '.join(valid_priorities)}")
            raise typer.Exit(code=1)

        # 1. Load the Knowledge Base
        console.print(f"[bold blue]Loading Knowledge Base from[/bold blue] '{DATA_PATH}'...")
        kb = KnowledgeBase(DATA_PATH)
        hw_profile = kb.get_hardware(hardware)
        console.print(f"Target hardware: [bold green]{hw_profile.vendor} {hw_profile.identifier}[/bold green]")

        # 2. Analyze the Model
        console.print(f"\n[bold blue]Analyzing model[/bold blue] '{model_path.name}'...")
        model_profile = analyze_model(model_path)
        console.print(f"Model SHA256: [cyan]{model_profile.model_sha256[:12]}...[/cyan]")
        console.print(f"Total operations: {model_profile.total_ops}, Total MACs: {model_profile.total_macs / 1_000_000:.2f}M")

        # 3. Call the Decision Engine
        console.print("\n[bold blue]Generating recommendations...[/bold blue]")
        advisor = HeuristicAdvisor()
        # Pass the user_priority from CLI to the advisor
        report = advisor.advise(model_profile, hw_profile, user_priority=user_priority)

        # 4. Present the Report
        console.print("\n[bold magenta]OAK Optimization Report[/bold magenta]")
        console.print(f"Model: {model_path.name} | Hardware: {hardware} | Priority: {user_priority}") # Added priority to summary

        table = Table(title="Strategy Recommendations")
        table.add_column("Priority", justify="right", style="cyan", no_wrap=True)
        table.add_column("Strategy", style="magenta")
        table.add_column("ROM (KB)", justify="right", style="green")
        table.add_column("RAM (KB)", justify="right", style="yellow")
        table.add_column("Summary", style="white")

        for rec in report.recommendations:
            table.add_row(
                f"{rec.priority_score:.2f}",
                rec.strategy_name,
                f"{rec.estimated_rom_kb:.1f}",
                f"{rec.estimated_ram_kb:.1f}",
                rec.summary, # Assumes rec.summary is already in English from HeuristicAdvisor
            )

        console.print(table)

    except (ModelAnalysisError, KnowledgeBaseError, FileNotFoundError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except typer.Exit: # To avoid catching the Exit we just raised
        raise
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        # Add traceback for debugging unexpected errors
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(code=1)

# Script entry point
if __name__ == "__main__":
    app()