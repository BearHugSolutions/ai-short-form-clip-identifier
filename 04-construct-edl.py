import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import typer
from rich.console import Console
import inquirer

# --- Data Structures ---
@dataclass
class TranscriptDirectory:
    """Represents a valid transcript directory with clips content"""
    path: Path
    name: str
    clips_folders: List[str]
    has_video: bool
    video_filename: Optional[str] = None
    
    def __str__(self):
        status_indicators = []
        status_indicators.append(f"{len(self.clips_folders)} clips folders")
        status_indicators.append("✓ video" if self.has_video else "✗ video")
        if self.video_filename:
            status_indicators.append(f"({self.video_filename})")
        return f"{self.name} ({', '.join(status_indicators)})"

@dataclass
class ClipsFolder:
    """Represents a clips folder with its JSON files"""
    path: Path
    name: str
    json_files: List[str]
    
    def __str__(self):
        return f"{self.name} ({len(self.json_files)} JSON files)"

# --- CLI Setup ---
app = typer.Typer(
    name="edl-generator",
    help="Interactive EDL generator that creates CMX 3600 EDL files from clips JSON data.",
    add_completion=False,
)
console = Console()

# --- Core EDL Functions ---
def to_timecode(td: timedelta, fps: float) -> str:
    """Converts a timedelta object to HH:MM:SS:FF timecode format."""
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    frames = int((seconds - int(seconds)) * fps)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{frames:02d}"

def parse_timestamp(ts: str) -> timedelta:
    """Parses HH:MM:SS string into a timedelta object."""
    h, m, s = map(int, ts.split(':'))
    return timedelta(hours=h, minutes=m, seconds=s)

def check_overlaps(clips):
    """Check for overlapping clips and report them."""
    overlaps = []
    for i, clip1 in enumerate(clips):
        start1 = parse_timestamp(clip1['start_timestamp'])
        end1 = parse_timestamp(clip1['end_timestamp'])
        
        for j, clip2 in enumerate(clips[i+1:], i+1):
            start2 = parse_timestamp(clip2['start_timestamp'])
            end2 = parse_timestamp(clip2['end_timestamp'])
            
            # Check if clips overlap
            if start1 < end2 and start2 < end1:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                overlap_duration = overlap_end - overlap_start
                
                overlaps.append({
                    'clip1': {'title': clip1['title'], 'index': i},
                    'clip2': {'title': clip2['title'], 'index': j},
                    'overlap_duration': overlap_duration.total_seconds()
                })
    
    return overlaps

def create_edl(json_path: str, output_path: str, video_filename: str, fps: float = 29.97):
    """
    Generates a CMX 3600 EDL file from a JSON file containing video clips.
    Handles overlapping source clips by placing them sequentially on the record timeline.

    Args:
        json_path (str): Path to the input clips JSON file.
        output_path (str): Path to save the generated .edl file.
        video_filename (str): The base name of the source video file for the EDL title.
        fps (float): Frames per second of the source video.
    """
    try:
        with open(json_path, 'r') as f:
            clips = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error reading JSON file:[/bold red] {e}")
        return False

    console.print(f"[bold blue]Processing {len(clips)} clips...[/bold blue]")
    
    # Check for overlaps and report them
    overlaps = check_overlaps(clips)
    if overlaps:
        console.print(f"\n[yellow]📊 Found {len(overlaps)} overlapping clip pairs:[/yellow]")
        for overlap in overlaps:
            console.print(f"  • '{overlap['clip1']['title'][:40]}...' overlaps with")
            console.print(f"    '{overlap['clip2']['title'][:40]}...' ({overlap['overlap_duration']:.1f}s)")
        console.print("  [green]✅ EDL will handle overlaps by using the same source reel multiple times[/green]\n")
    else:
        console.print("[green]✅ No overlapping clips detected[/green]\n")

    # Sort clips by start time to ensure chronological order in the EDL
    clips.sort(key=lambda x: parse_timestamp(x['start_timestamp']))

    edl_events = []
    record_timeline = timedelta(hours=1)  # Start record timeline at 01:00:00:00
    
    # Use a short, 8-char max reel name from the video filename
    reel_name = os.path.splitext(video_filename)[0][:8].upper()
    console.print(f"[bold cyan]Using reel name:[/bold cyan] {reel_name}")
    console.rule("[bold cyan]Processing Clips[/bold cyan]")

    for i, clip in enumerate(clips):
        event_num = i + 1
        title = clip.get('title', 'Untitled Clip')
        
        source_in_td = parse_timestamp(clip['start_timestamp'])
        source_out_td = parse_timestamp(clip['end_timestamp'])
        clip_duration = source_out_td - source_in_td

        record_out_td = record_timeline + clip_duration

        source_in_tc = to_timecode(source_in_td, fps)
        source_out_tc = to_timecode(source_out_td, fps)
        record_in_tc = to_timecode(record_timeline, fps)
        record_out_tc = to_timecode(record_out_td, fps)

        console.print(f"[dim]Event {event_num:03d}:[/dim] {title[:50]}")
        console.print(f"  [dim]Source:[/dim] {source_in_tc} → {source_out_tc} [dim](duration: {clip_duration.total_seconds():.1f}s)[/dim]")
        console.print(f"  [dim]Record:[/dim] {record_in_tc} → {record_out_tc}")
        console.print()

        # EDL Event Format (CMX 3600)
        # Format: Event# ReelName EditType Track Source_In Source_Out Record_In Record_Out
        event = (
            f"{event_num:03d}  {reel_name:<8} B     C        "
            f"{source_in_tc} {source_out_tc} {record_in_tc} {record_out_tc}"
        )
        
        # Add a comment with the clip title for readability
        comment = f"* FROM CLIP: {title}"
        
        edl_events.append(event)
        edl_events.append(comment)
        edl_events.append("")  # Blank line for readability

        # Update the record timeline for the next clip
        record_timeline = record_out_td

    # Calculate total program duration
    total_program_duration = record_timeline - timedelta(hours=1)
    console.rule("[bold green]EDL Summary[/bold green]")
    console.print(f"[bold green]Total program duration:[/bold green] {total_program_duration}")

    # Assemble the final EDL content
    header = f"TITLE: {video_filename} Highlights"
    fcm_line = "FCM: NON-DROP FRAME"  # Assuming non-drop frame timecode
    
    edl_content = "\n".join([header, fcm_line, ""] + edl_events)

    try:
        with open(output_path, 'w') as f:
            f.write(edl_content)
        
        console.print(f"[bold green]✅ Successfully created EDL file at:[/bold green] {output_path}")
        console.print(f"[bold blue]📺 Total clips:[/bold blue] {len(clips)}")
        console.print(f"[bold blue]⏱️  Program duration:[/bold blue] {total_program_duration}")
        
        # Show a sample of the EDL content
        console.rule("[bold cyan]EDL Preview[/bold cyan]")
        lines = edl_content.split('\n')[:15]  # Show first 15 lines
        for line in lines:
            console.print(f"[dim]{line}[/dim]")
        if len(edl_content.split('\n')) > 15:
            console.print("[dim]...[/dim]")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Error writing EDL file:[/bold red] {e}")
        return False

# --- Directory Discovery Functions ---
def find_transcript_directories(search_dir: Path = Path(".")) -> List[TranscriptDirectory]:
    """Find valid transcript directories with clips content"""
    ignore_dirs = {
        "venv", ".venv", "env", ".env", "baml_client", "baml_src", 
        "__pycache__", ".git", ".gitignore", "node_modules", ".DS_Store"
    }
    transcript_dirs = []
    
    for item in search_dir.iterdir():
        if not item.is_dir() or item.name.startswith('.') or item.name in ignore_dirs:
            continue
            
        # Look for clips folders
        clips_folders = []
        for potential_clips_dir in item.iterdir():
            if potential_clips_dir.is_dir() and "clips" in potential_clips_dir.name.lower():
                # Check if it has JSON files
                json_files = list(potential_clips_dir.glob("*.json"))
                if json_files:
                    clips_folders.append(potential_clips_dir.name)
        
        if not clips_folders:
            continue
            
        # Check for video files
        video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(item.glob(f"*{ext}")))
        
        video_filename = video_files[0].name if video_files else None
        
        transcript_dirs.append(TranscriptDirectory(
            path=item, 
            name=item.name, 
            clips_folders=sorted(clips_folders),
            has_video=len(video_files) > 0,
            video_filename=video_filename
        ))
    
    return sorted(transcript_dirs, key=lambda x: x.name)

def find_clips_folders(transcript_dir: Path) -> List[ClipsFolder]:
    """Find clips folders within a transcript directory"""
    clips_folders = []
    
    for item in transcript_dir.iterdir():
        if item.is_dir() and "clips" in item.name.lower():
            json_files = [f.name for f in item.glob("*.json")]
            if json_files:
                clips_folders.append(ClipsFolder(
                    path=item,
                    name=item.name,
                    json_files=sorted(json_files)
                ))
    
    return sorted(clips_folders, key=lambda x: x.name)

def select_transcript_directory_with_inquirer(transcript_dirs: List[TranscriptDirectory]) -> Optional[TranscriptDirectory]:
    """Let user select transcript directory using inquirer"""
    if not transcript_dirs:
        console.print("[bold red]No valid transcript directories found![/bold red]")
        return None
    if len(transcript_dirs) == 1:
        console.print(f"[green]Auto-selected only available directory:[/green] {transcript_dirs[0].name}")
        return transcript_dirs[0]
    
    choices = [(str(td), td) for td in transcript_dirs]
    choices.append(("Exit", None))
    
    try:
        questions = [inquirer.List('transcript_dir', message="Select transcript directory", choices=choices, carousel=True)]
        answers = inquirer.prompt(questions)
        return answers['transcript_dir'] if answers else None
    except (KeyboardInterrupt, Exception):
        console.print("\n[yellow]Inquirer failed. Falling back to text selection.[/yellow]")
        return select_transcript_directory_fallback(transcript_dirs)

def select_clips_folder_with_inquirer(clips_folders: List[ClipsFolder]) -> Optional[ClipsFolder]:
    """Let user select clips folder using inquirer"""
    if not clips_folders:
        console.print("[bold red]No clips folders found![/bold red]")
        return None
    if len(clips_folders) == 1:
        console.print(f"[green]Auto-selected only available clips folder:[/green] {clips_folders[0].name}")
        return clips_folders[0]
    
    choices = [(str(cf), cf) for cf in clips_folders]
    choices.append(("Back", None))
    
    try:
        questions = [inquirer.List('clips_folder', message="Select clips folder", choices=choices, carousel=True)]
        answers = inquirer.prompt(questions)
        return answers['clips_folder'] if answers else None
    except (KeyboardInterrupt, Exception):
        console.print("\n[yellow]Inquirer failed. Falling back to text selection.[/yellow]")
        return select_clips_folder_fallback(clips_folders)

def select_json_file_with_inquirer(clips_folder: ClipsFolder) -> Optional[str]:
    """Let user select JSON file using inquirer"""
    if len(clips_folder.json_files) == 1:
        console.print(f"[green]Auto-selected only available JSON file:[/green] {clips_folder.json_files[0]}")
        return clips_folder.json_files[0]
    
    choices = [(json_file, json_file) for json_file in clips_folder.json_files]
    choices.append(("Back", None))
    
    try:
        questions = [inquirer.List('json_file', message="Select JSON file", choices=choices, carousel=True)]
        answers = inquirer.prompt(questions)
        return answers['json_file'] if answers else None
    except (KeyboardInterrupt, Exception):
        console.print("\n[yellow]Inquirer failed. Falling back to text selection.[/yellow]")
        return select_json_file_fallback(clips_folder.json_files)

def select_transcript_directory_fallback(transcript_dirs: List[TranscriptDirectory]) -> Optional[TranscriptDirectory]:
    """Fallback text-based selection if inquirer fails"""
    console.print("\n[bold cyan]Available directories:[/bold cyan]")
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

def select_clips_folder_fallback(clips_folders: List[ClipsFolder]) -> Optional[ClipsFolder]:
    """Fallback text-based selection for clips folders"""
    console.print("\n[bold cyan]Available clips folders:[/bold cyan]")
    for i, cf in enumerate(clips_folders, 1):
        console.print(f"  {i}. {cf}")
    while True:
        try:
            choice = input(f"\nSelect a clips folder (1-{len(clips_folders)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit', 'back']: return None
            index = int(choice) - 1
            if 0 <= index < len(clips_folders): return clips_folders[index]
        except (ValueError, KeyboardInterrupt):
            return None

def select_json_file_fallback(json_files: List[str]) -> Optional[str]:
    """Fallback text-based selection for JSON files"""
    console.print("\n[bold cyan]Available JSON files:[/bold cyan]")
    for i, jf in enumerate(json_files, 1):
        console.print(f"  {i}. {jf}")
    while True:
        try:
            choice = input(f"\nSelect a JSON file (1-{len(json_files)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit', 'back']: return None
            index = int(choice) - 1
            if 0 <= index < len(json_files): return json_files[index]
        except (ValueError, KeyboardInterrupt):
            return None

@app.command()
def main(
    transcript_dir: Optional[Path] = typer.Option(None, "--transcript-dir", "-d", help="Path to the transcript directory"),
    clips_folder: Optional[str] = typer.Option(None, "--clips-folder", "-c", help="Name of the clips folder to use"),
    json_file: Optional[str] = typer.Option(None, "--json-file", "-j", help="Name of the JSON file to use"),
    fps: float = typer.Option(29.97, "--fps", "-f", help="Frames per second of the source video"),
    no_inquirer: bool = typer.Option(False, "--no-inquirer", help="Disable interactive UI and use text-based selection"),
):
    """Interactive EDL generator that creates CMX 3600 EDL files from clips JSON data."""
    
    console.rule("[bold cyan]EDL Generator - Interactive Mode[/bold cyan]")
    
    # Step 1: Select transcript directory
    selected_transcript_dir = transcript_dir
    if selected_transcript_dir is None:
        console.print("[bold blue]Step 1: Select Transcript Directory[/bold blue]")
        transcript_dirs = find_transcript_directories()
        if no_inquirer:
            selected_transcript_obj = select_transcript_directory_fallback(transcript_dirs)
        else:
            selected_transcript_obj = select_transcript_directory_with_inquirer(transcript_dirs)
        
        if selected_transcript_obj is None:
            console.print("[yellow]No directory selected. Exiting.[/yellow]")
            raise typer.Exit(code=1)
        
        selected_transcript_dir = selected_transcript_obj.path
        console.print(f"\n[green]Selected directory:[/green] {selected_transcript_obj.name}")
    
    if not selected_transcript_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory {selected_transcript_dir} not found")
        raise typer.Exit(code=1)
    
    # Step 2: Select clips folder
    selected_clips_folder = clips_folder
    if selected_clips_folder is None:
        console.print(f"\n[bold blue]Step 2: Select Clips Folder in {selected_transcript_dir.name}[/bold blue]")
        clips_folders = find_clips_folders(selected_transcript_dir)
        if no_inquirer:
            selected_clips_obj = select_clips_folder_fallback(clips_folders)
        else:
            selected_clips_obj = select_clips_folder_with_inquirer(clips_folders)
        
        if selected_clips_obj is None:
            console.print("[yellow]No clips folder selected. Exiting.[/yellow]")
            raise typer.Exit(code=1)
        
        selected_clips_folder = selected_clips_obj.name
        clips_folder_path = selected_clips_obj.path
        console.print(f"[green]Selected clips folder:[/green] {selected_clips_folder}")
    else:
        clips_folder_path = selected_transcript_dir / selected_clips_folder
        if not clips_folder_path.is_dir():
            console.print(f"[bold red]Error:[/bold red] Clips folder {clips_folder_path} not found")
            raise typer.Exit(code=1)
    
    # Step 3: Select JSON file
    selected_json_file = json_file
    if selected_json_file is None:
        console.print(f"\n[bold blue]Step 3: Select JSON File in {selected_clips_folder}[/bold blue]")
        clips_folder_obj = ClipsFolder(
            path=clips_folder_path,
            name=selected_clips_folder,
            json_files=[f.name for f in clips_folder_path.glob("*.json")]
        )
        if no_inquirer:
            selected_json_file = select_json_file_fallback(clips_folder_obj.json_files)
        else:
            selected_json_file = select_json_file_with_inquirer(clips_folder_obj)
        
        if selected_json_file is None:
            console.print("[yellow]No JSON file selected. Exiting.[/yellow]")
            raise typer.Exit(code=1)
        
        console.print(f"[green]Selected JSON file:[/green] {selected_json_file}")
    
    # Step 4: Find video file and generate paths
    video_extensions = [".mp4", ".webm", ".mkv", ".avi", ".mov"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(selected_transcript_dir.glob(f"*{ext}")))
    
    if not video_files:
        console.print(f"[bold red]Error:[/bold red] No video files found in {selected_transcript_dir}")
        raise typer.Exit(code=1)
    
    source_video = video_files[0].name
    json_path = clips_folder_path / selected_json_file
    edl_output_path = selected_transcript_dir / f"{os.path.splitext(source_video)[0]}_clips_{selected_clips_folder}.edl"
    
    # Step 5: Generate EDL
    console.rule(f"[bold green]Generating EDL[/bold green]")
    console.print(f"[bold blue]Source JSON:[/bold blue] {json_path}")
    console.print(f"[bold blue]Source Video:[/bold blue] {source_video}")
    console.print(f"[bold blue]Output EDL:[/bold blue] {edl_output_path}")
    console.print(f"[bold blue]FPS:[/bold blue] {fps}")
    console.print()
    
    success = create_edl(str(json_path), str(edl_output_path), source_video, fps)
    
    if success:
        console.rule("[bold green]Success! 🎬[/bold green]")
        console.print(f"[green]Your EDL file is ready at:[/green] {edl_output_path}")
        console.print("[dim]You can now import this EDL into your video editing software.[/dim]")
    else:
        console.rule("[bold red]Failed ❌[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()