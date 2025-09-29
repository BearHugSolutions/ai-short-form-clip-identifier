import typer
from pathlib import Path
from rich.console import Console
import asyncio
import json
from typing import List, Optional
from dataclasses import dataclass, asdict
import inquirer
import importlib.util
import sys

# --- Dynamic Module Loading ---
def load_module_from_path(module_name: str, file_path: Path):
    """Dynamically loads a module from a given file path."""
    if not file_path.exists():
        raise ImportError(f"Could not find module file at {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load module from {file_path}")

try:
    current_dir = Path(__file__).parent
    hybrid_module_path = current_dir / "03-hybrid_clip_identification.py"
    hybrid_module = load_module_from_path("hybrid_module", hybrid_module_path)
    
except ImportError as e:
    print(f"Error loading a required module: {e}")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv()

# --- Data Structures ---
@dataclass
class TranscriptDirectory:  
    """Represents a valid transcript directory with chunked content"""
    path: Path
    name: str
    chunks_count: int
    has_timestamps: bool
    has_video: bool
    
    def __str__(self):
        status_indicators = []
        status_indicators.append(f"{self.chunks_count} chunks")
        status_indicators.append("✓ timestamps" if self.has_timestamps else "✗ timestamps")
        status_indicators.append("✓ video" if self.has_video else "✗ video")
        return f"{self.name} ({', '.join(status_indicators)})"

# --- CLI Setup ---
app = typer.Typer(
    name="run-hybrid-extraction",
    help="Runs the hybrid multi-layer pipeline to extract content clips from a transcript.",
    add_completion=False,
)
console = Console()

# --- Directory Discovery Functions ---
def find_transcript_directories(search_dir: Path = Path(".")) -> List[TranscriptDirectory]:
    """Find valid transcript directories with chunked content"""
    ignore_dirs = {
        "venv", ".venv", "env", ".env", "baml_client", "baml_src", 
        "__pycache__", ".git", ".gitignore", "node_modules", ".DS_Store"
    }
    transcript_dirs = []
    for item in search_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.') or item.name in ignore_dirs:
            continue
        chunks_dir = item / "chunks"
        if not chunks_dir.is_dir():
            continue
        chunk_files = list(chunks_dir.glob("chunk_*.txt"))
        if not chunk_files:
            continue
        timestamp_files = list(item.glob("*_with_timestamps.txt"))
        video_extensions = [".mp4", ".webm", ".mkv", ".avi"]
        has_video = any(list(item.glob(f"*{ext}")) for ext in video_extensions)
        transcript_dirs.append(TranscriptDirectory(
            path=item, name=item.name, chunks_count=len(chunk_files),
            has_timestamps=len(timestamp_files) > 0, has_video=has_video
        ))
    return sorted(transcript_dirs, key=lambda x: x.name)

def select_transcript_directory_with_inquirer(transcript_dirs: List[TranscriptDirectory]) -> Optional[TranscriptDirectory]:
    """Let user select transcript directory using inquirer"""
    if not transcript_dirs:
        console.print("[bold red]No valid transcript directories found![/bold red]")
        return None
    if len(transcript_dirs) == 1:
        return transcript_dirs[0]
    
    choices = [(str(td), td) for td in transcript_dirs]
    choices.append(("Exit", None))
    
    try:
        questions = [inquirer.List('transcript_dir', message="Select transcript directory to process", choices=choices, carousel=True)]
        answers = inquirer.prompt(questions)
        return answers['transcript_dir'] if answers else None
    except (KeyboardInterrupt, Exception):
        console.print("\n[yellow]Inquirer failed. Falling back to text selection.[/yellow]")
        return select_transcript_directory_fallback(transcript_dirs)

def select_transcript_directory_fallback(transcript_dirs: List[TranscriptDirectory]) -> Optional[TranscriptDirectory]:
    """Fallback text-based selection if inquirer fails"""
    for i, td in enumerate(transcript_dirs, 1):
        console.print(f"  {i}. {td}")
    while True:
        try:
            choice = input(f"\nSelect a directory (1-{len(transcript_dirs)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']: return None
            index = int(choice) - 1
            if 0 <= index < len(transcript_dirs): return transcript_dirs[index]
        except (ValueError, KeyboardInterrupt):
            return None

async def main_async(transcript_dir: Path, prompt: str, target_clip_count: int):
    """Asynchronous main function to run the hybrid pipeline."""
    console.rule("[bold cyan]Starting Hybrid Clip Extraction[/bold cyan]")
    
    pipeline = hybrid_module.HybridExtractionPipeline()
    
    # The pipeline now handles all logic, including saving files.
    final_clips = await pipeline.run_hybrid_pipeline(
        transcript_dir=transcript_dir,
        user_query=prompt,
        target_clip_count=target_clip_count
    )
    
    console.rule(f"[bold green]Pipeline Complete[/bold green]")
    
    if not final_clips:
        console.print("No valid clips were generated.")
        console.print("[]")
    else:
        # Output final JSON to console for piping or review
        results_list = [asdict(clip) for clip in final_clips]
        console.rule("[bold green]Final Composed Clips (JSON Output)[/bold green]")
        console.print(json.dumps(results_list, indent=2))

@app.command()
def main(
    transcript_dir: Optional[Path] = typer.Option(None, "--transcript-dir", "-d", help="Path to the transcript directory. If not provided, will show selection menu."),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The user's prompt defining the desired content."),
    target_clips: int = typer.Option(15, "--target-clips", "-c", help="Target number of clips to extract."),
    no_inquirer: bool = typer.Option(False, "--no-inquirer", help="Disable interactive UI and use text-based selection."),
):
    """Identifies and extracts content clips using the hybrid multi-layer pipeline."""
    
    selected_dir_path = transcript_dir
    if selected_dir_path is None:
        console.print("[bold cyan]Hybrid Clip Extraction - Directory Selection[/bold cyan]")
        transcript_dirs = find_transcript_directories()
        if no_inquirer:
            selected_dir_obj = select_transcript_directory_fallback(transcript_dirs)
        else:
            selected_dir_obj = select_transcript_directory_with_inquirer(transcript_dirs)
        
        if selected_dir_obj is None:
            console.print("No directory selected. Exiting.")
            raise typer.Exit(code=1)
        
        selected_dir_path = selected_dir_obj.path
        console.print(f"\n[green]Selected:[/green] {selected_dir_obj}")
    
    if not selected_dir_path.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory {selected_dir_path} not found or is not a directory.")
        raise typer.Exit(code=1)
    
    console.print(f"\n[bold blue]Processing:[/bold blue] {selected_dir_path.name}")
    console.print(f"[bold blue]Prompt:[/bold blue] \"{prompt}\"")
    console.print(f"[bold blue]Target clips:[/bold blue] {target_clips}")
    
    asyncio.run(main_async(selected_dir_path, prompt, target_clips))

if __name__ == "__main__":
    app()
