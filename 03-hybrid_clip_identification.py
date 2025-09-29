#!/usr/bin/env python3
"""
Hybrid Multi-Layer Clip Extraction Pipeline
Blends top-down content strategy with bottom-up semantic precision
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import re
import importlib.util
import sys

# BAML imports
from baml_client.async_client import b
from baml_py.errors import BamlValidationError
from baml_client.types import ContentMap, SectionStrategy, PrecisionClip

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
    timestamp_module_path = current_dir / "03-timestamp_extraction.py"
    timestamp_module = load_module_from_path("timestamp_module", timestamp_module_path)
except ImportError as e:
    print(f"Error loading timestamp extraction module: {e}")
    sys.exit(1)


@dataclass
class MultiLayerChunks:
    """Container for different granularities of chunks"""
    coarse_chunks: List[str]  # 800-1500 tokens each (~5-10 min content)
    fine_chunks: List[str]    # 100-300 tokens each (your current chunks)
    coarse_to_fine_mapping: Dict[int, List[int]]  # Maps coarse chunk indices to fine chunk indices

@dataclass
class ThemeExtractionPlan:
    """Plan for extracting clips around specific themes"""
    theme: str
    target_coarse_sections: List[int]
    target_fine_chunks: List[int]
    max_clips_expected: int
    extraction_priority: str  # HIGH/MEDIUM/LOW

@dataclass
class ExtractedClip:
    """Final clip with all metadata"""
    title: str
    summary: str
    theme: str
    start_quote: str
    end_quote: str
    start_timestamp: str
    end_timestamp: str
    source_chunks: List[int]
    duration_seconds: int
    hook_strength: float
    closure_strength: float
    confidence_score: float

class HybridExtractionPipeline:
    """Multi-layer extraction pipeline combining strategy and precision"""

    def __init__(self):
        self.failure_log = []

    async def create_multi_layer_chunks(
        self,
        transcript_dir: Path,
        coarse_min_tokens: int = 800,
        coarse_max_tokens: int = 1500
    ) -> MultiLayerChunks:
        """Create both coarse and fine-grained chunks"""
        print("🧩 Creating multi-layer semantic chunks...")

        # Load existing fine chunks (from your current process)
        fine_chunks = self._load_existing_chunks(transcript_dir)

        # Create coarse chunks by combining fine chunks
        coarse_chunks = []
        coarse_to_fine_mapping = {}

        current_coarse = []
        current_tokens = 0
        current_fine_indices = []

        for i, fine_chunk in enumerate(fine_chunks):
            chunk_tokens = self._estimate_tokens(fine_chunk)

            # Check if adding this chunk would exceed max tokens
            if (current_tokens + chunk_tokens > coarse_max_tokens and
                current_tokens >= coarse_min_tokens):

                # Finalize current coarse chunk
                coarse_chunks.append(" ".join(current_coarse))
                coarse_to_fine_mapping[len(coarse_chunks) - 1] = current_fine_indices.copy()

                # Start new coarse chunk
                current_coarse = [fine_chunk]
                current_tokens = chunk_tokens
                current_fine_indices = [i]
            else:
                current_coarse.append(fine_chunk)
                current_tokens += chunk_tokens
                current_fine_indices.append(i)

        # Don't forget the last coarse chunk
        if current_coarse:
            coarse_chunks.append(" ".join(current_coarse))
            coarse_to_fine_mapping[len(coarse_chunks) - 1] = current_fine_indices

        print(f"   Created {len(coarse_chunks)} coarse chunks from {len(fine_chunks)} fine chunks")

        return MultiLayerChunks(
            coarse_chunks=coarse_chunks,
            fine_chunks=fine_chunks,
            coarse_to_fine_mapping=coarse_to_fine_mapping
        )

    async def phase1_content_mapping(
        self,
        chunks: MultiLayerChunks,
        user_query: str
    ) -> Optional[ContentMap]:
        """Phase 1: High-level content structure analysis using coarse chunks"""
        print("📋 Phase 1: Mapping content structure...")

        content_map = await self._call_baml_with_retry(
            b.MapContentStructure,
            context={"phase": "content_mapping"},
            coarse_chunks=chunks.coarse_chunks,
            user_query=user_query,
        )

        if not content_map:
            # Fallback returns None if BAML call fails completely
            print("   Failed to map content structure after retries.")
            return None

        print(f"   Identified {len(content_map.major_sections)} major content sections")
        return content_map

    async def phase2_extraction_strategy(
        self,
        content_map: ContentMap,
        user_query: str,
        target_clip_count: int = 15
    ) -> Optional[SectionStrategy]:
        """Phase 2: Create targeted extraction strategy"""
        print("🎯 Phase 2: Creating extraction strategy...")

        strategy = await self._call_baml_with_retry(
            b.CreateExtractionStrategy,
            context={"phase": "extraction_strategy"},
            content_map=content_map,
            user_query=user_query,
            target_clip_count=target_clip_count,
        )

        if not strategy:
            print("   Failed to create extraction strategy after retries.")
            return None

        print(f"   Strategy targets {len(strategy.extraction_priorities)} themes")
        return strategy

    async def phase3_theme_based_selection(
        self,
        chunks: MultiLayerChunks,
        strategy: SectionStrategy,
        user_query: str
    ) -> List[ThemeExtractionPlan]:
        """Phase 3: Select fine chunks based on theme priorities"""
        print("🎨 Phase 3: Theme-based chunk selection...")

        extraction_plans = []

        for theme_guide in strategy.extraction_priorities:
            theme = theme_guide.theme or 'general'
            target_sections = theme_guide.target_sections or []

            print(f"   Processing theme: {theme}")

            # Identify relevant coarse sections
            relevant_coarse_indices = []
            for i, section_summary in enumerate(strategy.target_sections):
                # Check if this coarse chunk is in target sections
                if section_summary in target_sections:
                    relevant_coarse_indices.append(i)

            if not relevant_coarse_indices and target_sections:
                 # Fallback if names don't match perfectly
                for i, coarse_chunk in enumerate(chunks.coarse_chunks):
                    if any(name.lower() in coarse_chunk.lower() for name in target_sections):
                        relevant_coarse_indices.append(i)

            # Get fine chunks from relevant coarse sections
            target_fine_chunks = []
            for coarse_idx in relevant_coarse_indices:
                target_fine_chunks.extend(
                    chunks.coarse_to_fine_mapping.get(coarse_idx, [])
                )

            # Evaluate fine chunks for this theme
            theme_matched_chunks = await self._evaluate_chunks_for_theme(
                chunks.fine_chunks,
                target_fine_chunks,
                theme,
                theme_guide.clip_characteristics or '',
                user_query
            )

            if theme_matched_chunks:
                plan = ThemeExtractionPlan(
                    theme=theme,
                    target_coarse_sections=relevant_coarse_indices,
                    target_fine_chunks=theme_matched_chunks,
                    max_clips_expected=min(len(theme_matched_chunks) // 2, 5),  # Conservative estimate
                    extraction_priority=theme_guide.priority or 'MEDIUM'
                )
                extraction_plans.append(plan)

        print(f"   Created {len(extraction_plans)} theme extraction plans")
        return extraction_plans

    async def phase4_precision_extraction(
        self,
        chunks: MultiLayerChunks,
        extraction_plans: List[ThemeExtractionPlan],
        user_query: str
    ) -> List[Dict]:
        """Phase 4: Extract precise clips from theme-matched content concurrently"""
        print("✂️ Phase 4: Precision clip extraction...")

        all_precision_clips = []
        tasks = []

        for plan in extraction_plans:
            print(f"   Queueing extraction for theme: {plan.theme}")

            chunk_groups = self._group_adjacent_chunks(plan.target_fine_chunks)

            for group in chunk_groups:
                if len(group) > 5:  # Don't process overly large groups
                    group = group[:5]

                combined_text = " ".join([chunks.fine_chunks[i] for i in group])

                # Create a task for each BAML call
                task = self._call_baml_with_retry(
                    b.ExtractPrecisionClips,
                    context={"theme": plan.theme, "chunks": group},
                    combined_chunks=combined_text,
                    target_theme=plan.theme,
                    user_query=user_query,
                    max_clips=min(3, plan.max_clips_expected),
                )
                tasks.append((task, group, plan))

        # Run all extraction tasks concurrently
        results = await asyncio.gather(*[t[0] for t in tasks])

        # Process the results
        for (clips, group, plan) in zip(results, [t[1] for t in tasks], [t[2] for t in tasks]):
            if clips:
                for clip in clips:
                    clip_data = {
                        'title': clip.title,
                        'summary': clip.summary,
                        'start_quote': clip.start_quote,
                        'end_quote': clip.end_quote,
                        'hook_strength': clip.hook_strength,
                        'closure_strength': clip.closure_strength,
                        'source_chunks': [i + 1 for i in group], # 1-based index
                        'theme': plan.theme,
                        'extraction_priority': plan.extraction_priority
                    }
                    all_precision_clips.append(clip_data)

        print(f"   Extracted {len(all_precision_clips)} precision clips")
        return all_precision_clips

    async def phase5_timestamp_resolution(
        self,
        precision_clips: List[Dict],
        timestamped_lines: List[Tuple[str, str]]
    ) -> List[ExtractedClip]:
        """Phase 5: Resolve exact timestamps and finalize clips"""
        print("🕐 Phase 5: Timestamp resolution and finalization...")

        final_clips = []

        for clip_data in precision_clips:
            print(f"   Resolving: {clip_data['title'][:40]}...")

            # Find timestamps using the centralized timestamp extraction module
            start_ts = timestamp_module.find_timestamp_for_quote(
                clip_data['start_quote'],
                timestamped_lines,
                error_tolerance=0.7
            )

            end_ts = timestamp_module.find_timestamp_for_quote(
                clip_data['end_quote'],
                timestamped_lines,
                search_after_ts=start_ts,
                error_tolerance=0.7
            )

            # Validate timing using the centralized function
            if not timestamp_module.validate_quote_timing(start_ts, end_ts):
                print(f"     Skipping due to invalid timing (start: {start_ts}, end: {end_ts})")
                self.failure_log.append({
                    "function": "validate_quote_timing",
                    "context": {"title": clip_data['title'], "start_ts": start_ts, "end_ts": end_ts},
                    "error": "Invalid or unordered timestamps."
                })
                continue

            # Calculate duration and confidence
            duration = self._calculate_duration(start_ts, end_ts)

            # Skip clips that are too short or too long for social media
            if not (5 <= duration <= 90):
                print(f"     Skipping due to duration ({duration}s)")
                continue

            confidence = self._calculate_confidence_score(clip_data, duration)

            final_clip = ExtractedClip(
                title=clip_data['title'],
                summary=clip_data['summary'],
                theme=clip_data['theme'],
                start_quote=clip_data['start_quote'],
                end_quote=clip_data['end_quote'],
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                source_chunks=clip_data['source_chunks'],
                duration_seconds=duration,
                hook_strength=clip_data.get('hook_strength', 0.7),
                closure_strength=clip_data.get('closure_strength', 0.7),
                confidence_score=confidence
            )

            final_clips.append(final_clip)

        # Sort by confidence and priority
        final_clips.sort(key=lambda x: (
            -x.confidence_score,
            -x.hook_strength,
            -x.closure_strength
        ))

        print(f"   Finalized {len(final_clips)} clips")
        return final_clips

    async def run_hybrid_pipeline(
        self,
        transcript_dir: Path,
        user_query: str,
        target_clip_count: int = 15
    ) -> List[ExtractedClip]:
        """Run the complete hybrid extraction pipeline"""
        print("🚀 Starting Hybrid Multi-Layer Extraction Pipeline")
        print("=" * 60)

        # Create multi-layer chunks
        chunks = await self.create_multi_layer_chunks(transcript_dir)

        # Phase 1: Content mapping using coarse chunks
        content_map = await self.phase1_content_mapping(chunks, user_query)
        if not content_map:
            return [] # Stop if phase 1 fails

        # Phase 2: Extraction strategy
        strategy = await self.phase2_extraction_strategy(content_map, user_query, target_clip_count)
        if not strategy:
            return [] # Stop if phase 2 fails

        # Phase 3: Theme-based selection using fine chunks
        extraction_plans = await self.phase3_theme_based_selection(chunks, strategy, user_query)

        # Phase 4: Precision extraction
        precision_clips = await self.phase4_precision_extraction(chunks, extraction_plans, user_query)

        # Phase 5: Timestamp resolution
        timestamped_lines = self._load_timestamped_transcript(transcript_dir)
        final_clips = await self.phase5_timestamp_resolution(precision_clips, timestamped_lines)

        # Keep only top clips
        if final_clips:
            final_clips = final_clips[:target_clip_count]

        # Save results
        self._save_hybrid_results(transcript_dir, final_clips, content_map, strategy)

        print("=" * 60)
        if final_clips:
            print(f"✅ Hybrid pipeline complete! Generated {len(final_clips)} clips")
            print(f"   Average confidence: {sum(c.confidence_score for c in final_clips) / len(final_clips):.2f}")
        else:
            print("✅ Hybrid pipeline complete! No valid clips were generated.")

        return final_clips

    # Helper methods
    def _load_existing_chunks(self, transcript_dir: Path) -> List[str]:
        """Load existing fine-grained chunks"""
        chunks_dir = transcript_dir / "chunks"
        chunk_files = sorted(chunks_dir.glob("chunk_*.txt"),
                           key=lambda p: int(p.stem.split("_")[1]))
        return [f.read_text(encoding='utf-8') for f in chunk_files]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text.split()) * 1.3  # Rough approximation

    async def _evaluate_chunks_for_theme(
        self,
        fine_chunks: List[str],
        target_chunk_indices: List[int],
        theme: str,
        theme_guide: str,
        user_query: str
    ) -> List[int]:
        """Evaluate fine chunks against a specific theme concurrently"""
        tasks = []
        valid_indices = []
        for chunk_idx in target_chunk_indices:
            if chunk_idx < len(fine_chunks):
                tasks.append(
                    self._call_baml_with_retry(
                        b.EvaluateChunkForTheme,
                        context={"theme": theme, "chunk": chunk_idx + 1},
                        chunk=fine_chunks[chunk_idx],
                        target_theme=theme,
                        theme_guide=theme_guide,
                        user_query=user_query,
                    )
                )
                valid_indices.append(chunk_idx)

        # Run all evaluation tasks concurrently
        results = await asyncio.gather(*tasks)

        # Filter for matched chunks
        matched_chunks = [
            chunk_idx for chunk_idx, evaluation in zip(valid_indices, results)
            if (evaluation and
                evaluation.matches_target_theme and
                evaluation.theme_strength > 0.6)
        ]

        return matched_chunks

    def _group_adjacent_chunks(self, chunk_indices: List[int]) -> List[List[int]]:
        """Group adjacent chunks for better context"""
        if not chunk_indices:
            return []

        groups = []
        current_group = [chunk_indices[0]]

        for i in range(1, len(chunk_indices)):
            if chunk_indices[i] == current_group[-1] + 1:  # Adjacent
                current_group.append(chunk_indices[i])
            else:
                groups.append(current_group)
                current_group = [chunk_indices[i]]

        groups.append(current_group)  # Don't forget the last group
        return groups

    def _calculate_duration(self, start_ts: str, end_ts: str) -> int:
        """Calculate duration in seconds"""
        def ts_to_seconds(ts: str) -> int:
            try:
                h, m, s = map(int, ts.split(':'))
                return h * 3600 + m * 60 + s
            except (ValueError, AttributeError):
                return 0
        
        start_seconds = ts_to_seconds(start_ts)
        end_seconds = ts_to_seconds(end_ts)
        
        if end_seconds > start_seconds:
            return end_seconds - start_seconds
        return 0


    def _calculate_confidence_score(self, clip_data: Dict, duration: int) -> float:
        """Calculate confidence score for clip quality"""
        base_score = 0.6

        # Boost for good duration (15-60 seconds ideal for social media)
        if 15 <= duration <= 60:
            base_score += 0.2
        elif 10 <= duration <= 75:
            base_score += 0.1

        # Boost for high hook/closure strength
        base_score += (clip_data.get('hook_strength', 0.5) * 0.15)
        base_score += (clip_data.get('closure_strength', 0.5) * 0.15)

        # Boost for high-priority themes
        if clip_data.get('extraction_priority') == 'HIGH':
            base_score += 0.1

        return min(1.0, base_score)

    def _load_timestamped_transcript(self, transcript_dir: Path) -> List[Tuple[str, str]]:
        """Load timestamped transcript"""
        timestamp_files = list(transcript_dir.glob("*_with_timestamps.txt"))
        if not timestamp_files:
            return []

        timestamped_lines = []
        time_pattern = re.compile(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)")

        with open(timestamp_files[0], "r", encoding='utf-8') as f:
            for line in f:
                match = time_pattern.match(line)
                if match:
                    timestamped_lines.append((match.group(1), match.group(2).strip()))

        return timestamped_lines

    def _save_hybrid_results(
        self,
        transcript_dir: Path,
        clips: List[ExtractedClip],
        content_map: Optional[ContentMap],
        strategy: Optional[SectionStrategy]
    ):
        """Save results with hybrid pipeline metadata"""
        clips_dir = transcript_dir / "clips_hybrid"
        clips_dir.mkdir(exist_ok=True)

        # Save individual clips
        for i, clip in enumerate(clips):
            clip_dict = asdict(clip)
            safe_title = re.sub(r'[^\w\s-]', '', clip.title)[:50]
            safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()

            filename = f"clip_{i+1:02d}_{safe_title}.json"

            with open(clips_dir / filename, 'w', encoding='utf-8') as f:
                json.dump(clip_dict, f, indent=2, ensure_ascii=False)

        # Save combined results
        all_clips_data = [asdict(clip) for clip in clips]
        with open(clips_dir / "all_clips_hybrid.json", 'w', encoding='utf-8') as f:
            json.dump(all_clips_data, f, indent=2, ensure_ascii=False)

        # Save pipeline metadata
        metadata = {
            "content_map": content_map.model_dump() if content_map else None,
            "extraction_strategy": strategy.model_dump() if strategy else None,
            "total_clips": len(clips),
            "average_duration": (sum(c.duration_seconds for c in clips) / len(clips)) if clips else 0,
            "themes_covered": list(set(c.theme for c in clips)),
            "failure_log": self.failure_log
        }

        with open(clips_dir / "hybrid_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def _call_baml_with_retry(self, baml_function, context: Dict, **kwargs):
        """Call BAML function with retry logic"""
        last_exception = None
        for attempt in range(3):
            try:
                # The baml_function is already an awaitable, so we just pass it along
                return await baml_function(**kwargs)
            except BamlValidationError as e:
                last_exception = e
                if attempt < 2:
                    await asyncio.sleep(0.5)

        self.failure_log.append({
            "function": baml_function.__name__,
            "context": context,
            "error": str(last_exception)
        })
        return None
