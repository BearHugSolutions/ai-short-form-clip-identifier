import re
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

def _normalize_text(text: str) -> str:
    """
    Internal helper to normalize text for more flexible matching.
    Converts to lowercase, removes extra whitespace and most punctuation.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"[^\w\s']", '', text)
    return text.strip()

def find_timestamp_for_quote(
    quote: str, 
    timestamped_lines: List[Tuple[str, str]],
    search_after_ts: Optional[str] = None,
    error_tolerance: float = 0.7
) -> str:
    """
    Finds the best matching timestamp for a given quote using fuzzy matching.

    Args:
        quote: The text quote to search for.
        timestamped_lines: A list of (timestamp, text) tuples.
        search_after_ts: Optional timestamp to start the search from.
        error_tolerance: Minimum similarity ratio for a match (0.0 to 1.0).

    Returns:
        The best matching timestamp string or "00:00:00" if no good match is found.
    """
    normalized_quote = _normalize_text(quote)
    search_phrase = " ".join(normalized_quote.split()[:15]) # Use first 15 words

    best_match_score = 0
    best_timestamp = "00:00:00"

    start_index = 0
    if search_after_ts:
        for i, (ts, _) in enumerate(timestamped_lines):
            if ts == search_after_ts:
                start_index = i
                break
    
    # Use a sliding window of 5 lines for context
    window_size = 5
    for i in range(start_index, len(timestamped_lines) - window_size + 1):
        window_lines = [line for _, line in timestamped_lines[i:i + window_size]]
        window_text = _normalize_text(" ".join(window_lines))
        
        # First, try a faster substring check
        if search_phrase in window_text:
            return timestamped_lines[i][0]

        # If no exact match, use fuzzy matching on the start of the window
        window_start = " ".join(window_text.split()[:20])
        similarity = SequenceMatcher(None, search_phrase, window_start).ratio()
        
        if similarity > best_match_score and similarity >= error_tolerance:
            best_match_score = similarity
            best_timestamp = timestamped_lines[i][0]

    return best_timestamp

def validate_quote_timing(start_ts: str, end_ts: str) -> bool:
    """
    Validates that the start and end timestamps are logical.
    - Both timestamps must be found (not "00:00:00").
    - End timestamp must be after the start timestamp.
    """
    if start_ts == "00:00:00" or end_ts == "00:00:00":
        return False
    
    # Simple string comparison works for HH:MM:SS format
    return end_ts > start_ts
