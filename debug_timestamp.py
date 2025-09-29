#!/usr/bin/env python3
"""
Debug script to test timestamp matching functionality.
Run this to see how your chunks compare to the timestamped transcript.
"""

import re
from pathlib import Path
from typing import List, Tuple
from difflib import SequenceMatcher

def normalize_text(text: str) -> str:
    """Normalize text for more flexible matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\w\s']", '', text)
    return text.strip()

def load_test_data(transcript_dir: Path):
    """Load chunks and timestamped lines for testing."""
    
    # Load chunks
    chunks_dir = transcript_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    chunks_text = [p.read_text() for p in chunk_files]
    
    # Load timestamped lines
    timestamp_files = list(transcript_dir.glob("*_with_timestamps.txt"))
    timestamp_file = timestamp_files[0]
    timestamped_lines: List[Tuple[str, str]] = []
    time_pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)")
    
    with open(timestamp_file, "r") as f:
        for line in f:
            match = time_pattern.match(line)
            if match:
                timestamped_lines.append((match.group(1), match.group(2).strip()))
    
    return chunks_text, timestamped_lines

def test_chunk_matching(chunk_idx: int, transcript_dir: Path):
    """Test matching for a specific chunk."""
    
    chunks_text, timestamped_lines = load_test_data(transcript_dir)
    
    if chunk_idx >= len(chunks_text):
        print(f"Error: Only {len(chunks_text)} chunks available, requested chunk {chunk_idx}")
        return
    
    chunk_text = chunks_text[chunk_idx]
    
    print(f"=== Testing Chunk #{chunk_idx + 1} ===")
    print(f"Chunk length: {len(chunk_text)} chars, {len(chunk_text.split())} words")
    print(f"Chunk text (first 200 chars):\n'{chunk_text[:200]}...'")
    print(f"\nNormalized chunk:\n'{normalize_text(chunk_text)[:200]}...'")
    
    # Test first 10 words of chunk
    chunk_words = normalize_text(chunk_text).split()
    search_phrase = " ".join(chunk_words[:10])
    print(f"\nSearch phrase (first 10 words):\n'{search_phrase}'")
    
    print(f"\n=== Searching in transcript ===")
    found_matches = []
    
    # Search through timestamped lines
    for i in range(len(timestamped_lines) - 15):
        window_lines = [line for _, line in timestamped_lines[i:i + 15]]
        window_text = normalize_text(" ".join(window_lines))
        
        # Check substring match
        if search_phrase in window_text:
            timestamp = timestamped_lines[i][0]
            found_matches.append((timestamp, "exact"))
            print(f"✅ EXACT match found at {timestamp}")
            print(f"   Context: '{window_text[:100]}...'")
            break
        
        # Check fuzzy match
        window_words = window_text.split()
        if len(window_words) >= 10:
            window_start = " ".join(window_words[:10])
            similarity = SequenceMatcher(None, search_phrase, window_start).ratio()
            
            if similarity > 0.7:
                timestamp = timestamped_lines[i][0]
                found_matches.append((timestamp, f"fuzzy-{similarity:.2f}"))
                if len(found_matches) == 1:  # Only show first good fuzzy match
                    print(f"🔍 FUZZY match found at {timestamp} (similarity: {similarity:.2f})")
                    print(f"   Expected: '{search_phrase}'")
                    print(f"   Found:    '{window_start}'")
                    print(f"   Context:  '{window_text[:100]}...'")
    
    if not found_matches:
        print("❌ No matches found!")
        print("\nFirst few timestamped lines for reference:")
        for i, (ts, line) in enumerate(timestamped_lines[:5]):
            print(f"  [{ts}] {line}")
            print(f"       → '{normalize_text(line)}'")
    
    print(f"\nTotal matches found: {len(found_matches)}")
    print("=" * 50)

if __name__ == "__main__":
    # Update this path to your transcript directory
    transcript_dir = Path("transcript__Zh7miqMQPI")
    
    # Test first few chunks
    for chunk_idx in [0, 1, 31, 32]:  # Test chunks that might be in your results
        try:
            test_chunk_matching(chunk_idx, transcript_dir)
            print("\n")
        except Exception as e:
            print(f"Error testing chunk {chunk_idx}: {e}")