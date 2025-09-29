// content_strategy.baml
// Multi-layer approach to balance global strategy with token efficiency
// VERSION: 2.0 - Optimized for token efficiency

// Schema for high-level content mapping
class ContentMap {
  major_sections SectionSummary[] @description("3-6 major thematic sections of the content")
  overall_structure string @description("High-level narrative flow (e.g., 'intro -> announcements -> demos -> integration')")
  clip_potential_score float @description("Overall assessment of content's clip-worthiness (0.0-1.0)")
  // REMOVED: `recommended_approach` - This can be inferred by the next LLM call from the structured data.
}

class SectionSummary {
  section_name string @description("Descriptive name for this content section")
  key_themes string[] @description("2-3 main themes/topics in this section")
  clip_density string @description("LOW/MEDIUM/HIGH - how many clips this section could yield")
  priority_for_query string @description("HIGH/MEDIUM/LOW - relevance to user query")
  // REMOVED: `approximate_chunk_range` - The LLM has to guess this. Better to calculate it in code if needed.
}

// Schema for focused section analysis
class SectionStrategy {
  target_sections string[] @description("Names of sections to focus extraction on")
  extraction_priorities ThemeGuide[] @description("Specific themes to target within sections")
  clip_count_estimate int @description("Realistic number of clips expected from this strategy")
  // REMOVED: `success_criteria` - The `clip_characteristics` in ThemeGuide is more specific and actionable.
}

class ThemeGuide {
  theme string @description("Specific theme to extract (e.g., 'product announcements')")
  target_sections string[] @description("Which sections likely contain this theme")
  clip_characteristics string @description("What to look for in clips for this theme")
  priority string @description("HIGH/MEDIUM/LOW priority for this theme based on relevance to the user query.")
}

// Function 1: High-level content mapping using coarse chunks
function MapContentStructure(coarse_chunks: string[], user_query: string) -> ContentMap {
  client CustomGPT5Mini
  prompt #"
    You are a content strategist analyzing a long-form video/podcast. You need to identify the major structural sections and assess clip potential.

    USER QUERY: "{{user_query}}"

    CONTENT SECTIONS (these are large chunks representing ~5-10 minutes each):
    {% for chunk in coarse_chunks %}
    === SECTION {{ loop.index0 }} ===
    {{ chunk }}
    
    {% endfor %}

    INSTRUCTIONS:
    1. Identify 3-6 major thematic sections of this content.
    2. For each section, assess its relevance to the user query and its potential clip density.
    3. Describe the overall narrative flow of the content.

    Focus on structure and themes, not specific quotes or timestamps.

    {{ ctx.output_format }}
  "#
}

// Function 2: Create targeted extraction strategy
function CreateExtractionStrategy(content_map: ContentMap, user_query: string, target_clip_count: int) -> SectionStrategy {
  client CustomGPT5Mini
  prompt #"
    You are a video editor planning clip extraction. Based on the content analysis, create a focused strategy.

    USER QUERY: "{{user_query}}"
    TARGET CLIPS: {{target_clip_count}}

    CONTENT ANALYSIS:
    {{content_map}}

    INSTRUCTIONS:
    1. Select the most promising sections for clip extraction.
    2. Define specific themes/topics to target within those sections.
    3. For each theme, provide guidance on what makes a good clip and assign a priority.
    4. Be realistic about clip counts - better to get fewer high-quality clips.

    Focus on strategy, not implementation details.

    {{ ctx.output_format }}
  "#
}

// Schema for theme-focused chunk evaluation (replaces pure similarity)
// OPTIMIZED: Removed all fields not used by the Python script.
class ThemeEvaluation {
  matches_target_theme bool @description("Does this chunk contain the target theme content?")
  theme_strength float @description("How strongly does this chunk represent the theme? (0.0-1.0)")
  // REMOVED: `clip_potential`, `needs_context`, `reasoning`. These are not used by the orchestration code
  // and removing them provides the biggest token savings for the entire pipeline.
}

// Function 3: Evaluate chunks against specific themes (replaces EvaluateSeedChunk)
function EvaluateChunkForTheme(chunk: string, target_theme: string, theme_guide: string, user_query: string) -> ThemeEvaluation {
  client CustomGPT5Mini
  prompt #"
    You are a content curator looking for specific thematic content. Your output must be lean and efficient.

    TARGET THEME: "{{target_theme}}"
    THEME GUIDANCE: {{theme_guide}}
    USER QUERY: "{{user_query}}"

    CHUNK TO EVALUATE:
    ---
    {{chunk}}
    ---

    INSTRUCTIONS:
    1. Assess if this chunk definitively contains content about the TARGET THEME.
    2. Provide a strict score from 0.0 to 1.0 for how strongly it represents the theme.

    Be strict and fast. Your goal is a simple yes/no check with a confidence score.

    {{ ctx.output_format }}
  "#
}

// Schema for precise clip boundaries within identified content
class PrecisionClip {
  start_quote string @description("Exact quote where clip should start (8-15 words)")
  end_quote string @description("Exact quote where clip should end (8-15 words)")
  title string @description("Engaging title for this clip")
  summary string @description("One-sentence summary of clip content")
  hook_strength float @description("How attention-grabbing is the opening? (0.0-1.0)")
  closure_strength float @description("How satisfying is the ending? (0.0-1.0)")
  // REMOVED: `theme` - The calling code already knows the theme, so the LLM doesn't need to return it.
}

// Function 4: Extract precise clips from theme-matched content
function ExtractPrecisionClips(
  combined_chunks: string, 
  target_theme: string, 
  user_query: string,
  max_clips: int
) -> PrecisionClip[] {
  client CustomGPT5Mini
  prompt #"
    You are a precision video editor. Extract short-form clips (15-90 seconds) from this content.

    TARGET THEME: "{{target_theme}}"
    USER QUERY: "{{user_query}}"
    MAX CLIPS: {{max_clips}}

    CONTENT TO EXTRACT FROM:
    ---
    {{combined_chunks}}
    ---

    INSTRUCTIONS:
    1. Find 1-{{max_clips}} potential clips that strongly match the target theme.
    2. For each clip, identify EXACT start and end quotes (be very specific).
    3. Prioritize clips with strong hooks and natural endings.
    4. Each clip should be self-contained and engaging.
    5. Start/end quotes should be 8-15 words for reliable matching.
    6. Provide a catchy title and a one-sentence summary for each.

    Remember: These will become TikTok/Instagram clips, so they need immediate engagement.

    {{ ctx.output_format }}
  "#
}
