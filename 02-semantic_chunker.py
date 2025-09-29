#!/usr/bin/env python3
"""
Semantic Chunker for YouTube Transcripts
Locally processes transcripts using semantic similarity to create meaningful chunks
Updated to work with files from 01-youtube_downloader.py and use inquirer for file selection
"""

import os
import re
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Any, Optional, Tuple, Dict
from dataclasses import dataclass

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import inquirer


@dataclass
class TranscriptSet:
    """Represents a complete set of downloaded files for a single video"""
    base_name: str
    transcript_file: str
    transcript_with_timestamps: Optional[str] = None
    video_file: Optional[str] = None
    
    def __str__(self):
        return f"{self.base_name} ({'✓' if self.video_file else '✗'} video, {'✓' if self.transcript_with_timestamps else '✗'} timestamps)"


@dataclass
class Chunk:
    """Represents a semantic chunk of text"""
    splits: List[str]
    is_triggered: bool = False
    triggered_score: Optional[float] = None
    token_count: Optional[int] = None
    
    @property
    def content(self) -> str:
        return " ".join(self.splits)


@dataclass
class ChunkStatistics:
    """Statistics about the chunking process"""
    total_documents: int
    total_chunks: int
    chunks_by_threshold: int
    chunks_by_max_chunk_size: int
    chunks_by_last_split: int
    min_token_size: int
    max_token_size: int
    chunks_by_similarity_ratio: float

    def __str__(self):
        return (
            f"Chunking Statistics:\n"
            f"  - Total Documents: {self.total_documents}\n"
            f"  - Total Chunks: {self.total_chunks}\n"
            f"  - Chunks by Threshold: {self.chunks_by_threshold}\n"
            f"  - Chunks by Max Chunk Size: {self.chunks_by_max_chunk_size}\n"
            f"  - Last Chunk: {self.chunks_by_last_split}\n"
            f"  - Minimum Token Size of Chunk: {self.min_token_size}\n"
            f"  - Maximum Token Size of Chunk: {self.max_token_size}\n"
            f"  - Similarity Chunk Ratio: {self.chunks_by_similarity_ratio:.2f}"
        )


class LocalSemanticChunker:
    """Local implementation of semantic chunking using sentence-transformers"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        threshold_adjustment: float = 0.01,
        dynamic_threshold: bool = True,
        window_size: int = 5,
        min_split_tokens: int = 100,
        max_split_tokens: int = 300,
        split_tokens_tolerance: int = 10,
        enable_statistics: bool = True,
    ):
        """
        Initialize the semantic chunker
        
        Args:
            model_name: Name of the sentence-transformers model
            threshold_adjustment: Step size for threshold optimization
            dynamic_threshold: Whether to dynamically find optimal threshold
            window_size: Size of the sliding window for context
            min_split_tokens: Minimum tokens per chunk
            max_split_tokens: Maximum tokens per chunk
            split_tokens_tolerance: Tolerance for token count optimization
            enable_statistics: Whether to collect and display statistics
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.threshold_adjustment = threshold_adjustment
        self.dynamic_threshold = dynamic_threshold
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.enable_statistics = enable_statistics
        self.DEFAULT_THRESHOLD = 0.5
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def tiktoken_length(self, text: str) -> int:
        """Count tokens using tiktoken"""
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def encode_documents(self, docs: List[str]) -> np.ndarray:
        """Encode documents into embeddings"""
        embeddings = self.model.encode(docs, convert_to_tensor=False, show_progress_bar=True)
        return np.array(embeddings)
    
    def calculate_similarity_scores(self, encoded_docs: np.ndarray) -> List[float]:
        """Calculate similarity scores using sliding window"""
        similarities = []
        for idx in range(1, len(encoded_docs)):
            window_start = max(0, idx - self.window_size)
            cumulative_context = np.mean(encoded_docs[window_start:idx], axis=0)
            
            # Cosine similarity
            curr_sim_score = np.dot(cumulative_context, encoded_docs[idx]) / (
                np.linalg.norm(cumulative_context) * np.linalg.norm(encoded_docs[idx]) + 1e-10
            )
            similarities.append(curr_sim_score)
        return similarities
    
    def find_split_indices(self, similarities: List[float], threshold: float) -> List[int]:
        """Find indices where to split based on similarity threshold"""
        split_indices = []
        for idx, score in enumerate(similarities):
            if score < threshold:
                split_indices.append(idx + 1)  # Split after the document at idx
        return split_indices
    
    def find_optimal_threshold(self, docs: List[str], similarities: List[float]) -> float:
        """Find optimal threshold using binary search"""
        token_counts = [self.tiktoken_length(doc) for doc in docs]
        cumulative_token_counts = np.cumsum([0] + token_counts)
        
        # Set initial bounds based on similarity distribution
        median_score = np.median(similarities)
        std_dev = np.std(similarities)
        
        low = max(0.0, float(median_score - std_dev))
        high = min(1.0, float(median_score + std_dev))
        
        iteration = 0
        calculated_threshold = 0.0
        
        while low <= high and iteration < 20:  # Prevent infinite loops
            calculated_threshold = (low + high) / 2
            split_indices = self.find_split_indices(similarities, calculated_threshold)
            
            # Calculate token counts for each split
            split_token_counts = [
                cumulative_token_counts[end] - cumulative_token_counts[start]
                for start, end in zip([0] + split_indices, split_indices + [len(token_counts)])
            ]
            
            median_tokens = np.median(split_token_counts) if split_token_counts else 0
            
            if (self.min_split_tokens - self.split_tokens_tolerance 
                <= median_tokens 
                <= self.max_split_tokens + self.split_tokens_tolerance):
                break
            elif median_tokens < self.min_split_tokens:
                high = calculated_threshold - self.threshold_adjustment
            else:
                low = calculated_threshold + self.threshold_adjustment
                
            iteration += 1
        
        return calculated_threshold
    
    def split_documents(self, docs: List[str], split_indices: List[int], similarities: List[float]) -> List[Chunk]:
        """Split documents into chunks based on split indices"""
        token_counts = [self.tiktoken_length(doc) for doc in docs]
        chunks = []
        current_split = []
        current_tokens_count = 0
        
        # Statistics tracking
        chunks_by_threshold = 0
        chunks_by_max_chunk_size = 0
        chunks_by_last_split = 0
        
        for doc_idx, doc in enumerate(docs):
            doc_token_count = token_counts[doc_idx]
            
            # Check if current index is a split point
            if doc_idx + 1 in split_indices:
                if (self.min_split_tokens 
                    <= current_tokens_count + doc_token_count 
                    < self.max_split_tokens):
                    
                    current_split.append(doc)
                    current_tokens_count += doc_token_count
                    
                    triggered_score = similarities[doc_idx] if doc_idx < len(similarities) else None
                    chunks.append(Chunk(
                        splits=current_split.copy(),
                        is_triggered=True,
                        triggered_score=triggered_score,
                        token_count=current_tokens_count,
                    ))
                    
                    current_split, current_tokens_count = [], 0
                    chunks_by_threshold += 1
                    continue
            
            # Check if adding current document exceeds max token limit
            if current_tokens_count + doc_token_count > self.max_split_tokens:
                if current_tokens_count >= self.min_split_tokens:
                    chunks.append(Chunk(
                        splits=current_split.copy(),
                        is_triggered=False,
                        triggered_score=None,
                        token_count=current_tokens_count,
                    ))
                    chunks_by_max_chunk_size += 1
                    current_split, current_tokens_count = [], 0
            
            current_split.append(doc)
            current_tokens_count += doc_token_count
        
        # Handle the last split
        if current_split:
            chunks.append(Chunk(
                splits=current_split.copy(),
                is_triggered=False,
                triggered_score=None,
                token_count=current_tokens_count,
            ))
            chunks_by_last_split += 1
        
        # Calculate statistics
        if self.enable_statistics:
            total_chunks = len(chunks)
            chunks_by_similarity_ratio = chunks_by_threshold / total_chunks if total_chunks else 0
            token_counts_chunks = [chunk.token_count for chunk in chunks if chunk.token_count]
            min_token_size = min(token_counts_chunks) if token_counts_chunks else 0
            max_token_size = max(token_counts_chunks) if token_counts_chunks else 0
            
            self.statistics = ChunkStatistics(
                total_documents=len(docs),
                total_chunks=total_chunks,
                chunks_by_threshold=chunks_by_threshold,
                chunks_by_max_chunk_size=chunks_by_max_chunk_size,
                chunks_by_last_split=chunks_by_last_split,
                min_token_size=min_token_size,
                max_token_size=max_token_size,
                chunks_by_similarity_ratio=chunks_by_similarity_ratio,
            )
        
        return chunks
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """Main chunking method"""
        print("Splitting text into sentences...")
        splits = self.split_text(text)
        
        # Handle case where text exceeds max tokens
        new_splits = []
        for split in splits:
            token_count = self.tiktoken_length(split)
            if token_count > self.max_split_tokens:
                # Further split long sentences
                words = split.split()
                current_chunk = []
                current_tokens = 0
                
                for word in words:
                    word_tokens = self.tiktoken_length(word)
                    if current_tokens + word_tokens > self.max_split_tokens and current_chunk:
                        new_splits.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_tokens = word_tokens
                    else:
                        current_chunk.append(word)
                        current_tokens += word_tokens
                
                if current_chunk:
                    new_splits.append(" ".join(current_chunk))
            else:
                new_splits.append(split)
        
        splits = [split for split in new_splits if split.strip()]
        
        print(f"Created {len(splits)} text segments")
        print("Generating embeddings...")
        encoded_splits = self.encode_documents(splits)
        
        print("Calculating similarity scores...")
        similarities = self.calculate_similarity_scores(encoded_splits)
        
        if self.dynamic_threshold:
            print("Finding optimal threshold...")
            threshold = self.find_optimal_threshold(splits, similarities)
            print(f"Optimal threshold: {threshold:.3f}")
        else:
            threshold = self.DEFAULT_THRESHOLD
        
        split_indices = self.find_split_indices(similarities, threshold)
        print(f"Found {len(split_indices)} split points")
        
        chunks = self.split_documents(splits, split_indices, similarities)
        
        if self.enable_statistics:
            print("\n" + str(self.statistics))
        
        return chunks


def find_youtube_downloader_files(directory: str = ".") -> List[TranscriptSet]:
    """Find files downloaded by 01-youtube_downloader.py based on naming schema"""
    # Look for transcript files: {base_name}_transcript.txt
    transcript_pattern = os.path.join(directory, "*_transcript.txt")
    transcript_files = glob.glob(transcript_pattern)
    
    transcript_sets = []
    
    for transcript_file in transcript_files:
        # Extract base name
        basename = os.path.basename(transcript_file)
        base_name = basename.replace("_transcript.txt", "")
        
        # Look for corresponding files
        transcript_with_timestamps = os.path.join(directory, f"{base_name}_with_timestamps.txt")
        if not os.path.exists(transcript_with_timestamps):
            transcript_with_timestamps = None
        
        # Look for video file (check multiple extensions)
        video_extensions = [".mp4", ".webm", ".mkv", ".avi"]
        video_file = None
        for ext in video_extensions:
            potential_video = os.path.join(directory, f"{base_name}_video{ext}")
            if os.path.exists(potential_video):
                video_file = potential_video
                break
        
        transcript_set = TranscriptSet(
            base_name=base_name,
            transcript_file=transcript_file,
            transcript_with_timestamps=transcript_with_timestamps,
            video_file=video_file
        )
        
        transcript_sets.append(transcript_set)
    
    return sorted(transcript_sets, key=lambda x: x.base_name)


def select_transcript_with_inquirer(transcript_sets: List[TranscriptSet]) -> Optional[TranscriptSet]:
    """Let user select which transcript set to process using inquirer"""
    if not transcript_sets:
        print("No transcript files found!")
        print("Make sure you have files downloaded by 01-youtube_downloader.py")
        return None
    
    if len(transcript_sets) == 1:
        print(f"Found one transcript set: {transcript_sets[0]}")
        return transcript_sets[0]
    
    # Create choices for inquirer
    choices = []
    for ts in transcript_sets:
        display_name = str(ts)
        choices.append((display_name, ts))
    
    # Add option to exit
    choices.append(("Exit", None))
    
    try:
        questions = [
            inquirer.List(
                'transcript_set',
                message="Select which transcript set to process",
                choices=choices,
                carousel=True
            ),
        ]
        
        answers = inquirer.prompt(questions)
        if answers is None or answers['transcript_set'] is None:
            return None
        
        return answers['transcript_set']
        
    except KeyboardInterrupt:
        print("\nCancelled by user")
        return None
    except Exception as e:
        print(f"Error with inquirer selection: {e}")
        print("Falling back to text-based selection...")
        return select_transcript_fallback(transcript_sets)


def select_transcript_fallback(transcript_sets: List[TranscriptSet]) -> Optional[TranscriptSet]:
    """Fallback text-based selection if inquirer fails"""
    print("Available transcript sets:")
    for i, ts in enumerate(transcript_sets, 1):
        print(f"  {i}. {ts}")
    
    while True:
        try:
            choice = input(f"\nSelect transcript set to process (1-{len(transcript_sets)}): ").strip()
            if choice.lower() in ['q', 'quit', 'exit']:
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(transcript_sets):
                return transcript_sets[index]
            else:
                print(f"Please enter a number between 1 and {len(transcript_sets)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled by user")
            return None


def create_output_directory(transcript_set: TranscriptSet) -> str:
    """Create output directory based on transcript set base name"""
    dir_name = f"{transcript_set.base_name}"
    
    # Create directory
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def save_chunks(chunks: List[Chunk], output_dir: str, transcript_set: TranscriptSet):
    """Save chunks to files and move original files to output directory"""
    print(f"Moving files to {output_dir}...")
    
    # Move original transcript files to output directory
    moved_files = []
    
    # Move main transcript file
    original_basename = os.path.basename(transcript_set.transcript_file)
    dest_transcript = os.path.join(output_dir, original_basename)
    shutil.move(transcript_set.transcript_file, dest_transcript)
    moved_files.append(original_basename)
    
    # Move transcript with timestamps if it exists
    if transcript_set.transcript_with_timestamps:
        timestamps_basename = os.path.basename(transcript_set.transcript_with_timestamps)
        dest_timestamps = os.path.join(output_dir, timestamps_basename)
        shutil.move(transcript_set.transcript_with_timestamps, dest_timestamps)
        moved_files.append(timestamps_basename)
    
    # Move video file if it exists
    if transcript_set.video_file:
        video_basename = os.path.basename(transcript_set.video_file)
        dest_video = os.path.join(output_dir, video_basename)
        shutil.move(transcript_set.video_file, dest_video)
        moved_files.append(video_basename)
        print(f"✓ Moved video file: {video_basename}")
    
    print(f"✓ Moved {len(moved_files)} original files: {', '.join(moved_files)}")
    
    # Save individual chunks
    chunks_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    
    chunk_metadata = []
    
    for i, chunk in enumerate(chunks, 1):
        # Save chunk content
        chunk_filename = f"chunk_{i:03d}.txt"
        chunk_path = os.path.join(chunks_dir, chunk_filename)
        
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(chunk.content)
        
        # Collect metadata
        metadata = {
            "chunk_id": i,
            "filename": chunk_filename,
            "token_count": chunk.token_count,
            "is_triggered": chunk.is_triggered,
            "triggered_score": float(chunk.triggered_score) if chunk.triggered_score is not None else None,
            "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        }
        chunk_metadata.append(metadata)
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "chunks_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "transcript_set": {
                "base_name": transcript_set.base_name,
                "had_video": transcript_set.video_file is not None,
                "had_timestamps": transcript_set.transcript_with_timestamps is not None
            },
            "chunks": chunk_metadata
        }, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(output_dir, "chunks_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Semantic Chunking Summary\n")
        f.write(f"={'='*50}\n\n")
        f.write(f"Original transcript set: {transcript_set.base_name}\n")
        f.write(f"Video file included: {'Yes' if transcript_set.video_file else 'No'}\n")
        f.write(f"Timestamps included: {'Yes' if transcript_set.transcript_with_timestamps else 'No'}\n")
        f.write(f"Total chunks created: {len(chunks)}\n")
        f.write(f"Chunks directory: {chunks_dir}\n\n")
        
        f.write("Chunk Details:\n")
        f.write("-" * 20 + "\n")
        for i, chunk in enumerate(chunks, 1):
            f.write(f"Chunk {i:03d}: {chunk.token_count} tokens")
            if chunk.is_triggered:
                f.write(f" (triggered by similarity: {chunk.triggered_score:.3f})")
            f.write(f"\n  Preview: {chunk.content[:100]}...\n\n")
    
    print(f"✓ Saved {len(chunks)} chunks to {chunks_dir}")
    print(f"✓ Metadata saved to {metadata_path}")
    print(f"✓ Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Chunker for YouTube Transcripts (works with 01-youtube_downloader.py files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --model BAAI/bge-base-en-v1.5 --max-tokens 500
  %(prog)s --directory /path/to/transcripts --min-tokens 50

This script looks for files downloaded by 01-youtube_downloader.py:
  - {base_name}_transcript.txt (main transcript)
  - {base_name}_with_timestamps.txt (timestamped transcript)
  - {base_name}_video.{ext} (video file)
        """
    )
    
    parser.add_argument('--directory', '-d', default='.', 
                       help='Directory to search for transcript files (default: current directory)')
    parser.add_argument('--model', '-m', default='BAAI/bge-m3',
                       help='Sentence transformer model to use (default: BAAI/bge-m3)')
    parser.add_argument('--min-tokens', type=int, default=100,
                       help='Minimum tokens per chunk (default: 100)')
    parser.add_argument('--max-tokens', type=int, default=300,
                       help='Maximum tokens per chunk (default: 300)')
    parser.add_argument('--window-size', type=int, default=5,
                       help='Window size for similarity calculation (default: 5)')
    parser.add_argument('--no-dynamic', action='store_true',
                       help='Disable dynamic threshold optimization')
    parser.add_argument('--no-inquirer', action='store_true',
                       help='Disable inquirer UI and use text-based selection')
    
    args = parser.parse_args()
    
    print("Semantic Chunker for YouTube Transcripts")
    print("Works with files from 01-youtube_downloader.py")
    print("=" * 50)
    
    # Find transcript sets from youtube downloader
    transcript_sets = find_youtube_downloader_files(args.directory)
    if not transcript_sets:
        print(f"No transcript files found in {args.directory}")
        print("Make sure you have files downloaded by 01-youtube_downloader.py with the naming pattern:")
        print("  - {base_name}_transcript.txt")
        print("  - {base_name}_with_timestamps.txt (optional)")
        print("  - {base_name}_video.{ext} (optional)")
        sys.exit(1)
    
    print(f"Found {len(transcript_sets)} transcript set(s)")
    
    # Select transcript set to process
    if args.no_inquirer:
        selected_set = select_transcript_fallback(transcript_sets)
    else:
        selected_set = select_transcript_with_inquirer(transcript_sets)
    
    if not selected_set:
        print("No transcript set selected. Exiting.")
        sys.exit(1)
    
    # Read transcript content
    print(f"\nReading transcript: {selected_set.transcript_file}")
    try:
        with open(selected_set.transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    if not content.strip():
        print("File is empty!")
        sys.exit(1)
    
    print(f"Loaded {len(content)} characters of text")
    
    # Initialize chunker
    print(f"\nInitializing semantic chunker...")
    chunker = LocalSemanticChunker(
        model_name=args.model,
        dynamic_threshold=not args.no_dynamic,
        window_size=args.window_size,
        min_split_tokens=args.min_tokens,
        max_split_tokens=args.max_tokens,
    )
    
    # Perform chunking
    print("\nStarting semantic chunking...")
    try:
        chunks = chunker.chunk_text(content)
    except Exception as e:
        print(f"Error during chunking: {e}")
        sys.exit(1)
    
    if not chunks:
        print("No chunks were created!")
        sys.exit(1)
    
    # Create output directory and save results
    output_dir = create_output_directory(selected_set)
    print(f"\nSaving results to: {output_dir}")
    
    try:
        save_chunks(chunks, output_dir, selected_set)
    except Exception as e:
        print(f"Error saving chunks: {e}")
        sys.exit(1)
    
    print(f"\n🎉 Completed! Check the '{output_dir}' directory for results.")
    print(f"   📁 Original files moved to: {output_dir}")
    print(f"   📄 Chunks saved to: {output_dir}/chunks/")


if __name__ == "__main__":
    main()