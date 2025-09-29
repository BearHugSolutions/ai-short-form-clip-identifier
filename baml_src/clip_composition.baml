// schema for the first step: Evaluating the core chunk
class SeedEvaluation {
  is_relevant bool @description("Is this chunk, on its own, highly engaging and relevant to the user's query?")
  relevance_score float @description("A score from 0.0 to 1.0 for the chunk's relevance. Be strict.")
  reasoning string @description("A step-by-step analysis of why this chunk is or is not a strong candidate.")
}

// schema for the second step: Deciding whether to add context
class ContextDecision {
  should_include bool @description("Is this context absolutely essential to understand the main chunk? Be very conservative.")
  inclusion_score float @description("A score from 0.0 to 1.0. A score above 0.8 indicates critical context. Lower scores mean it's likely unnecessary.")
  reasoning string @description("Explain why this context is or is not critical for the main chunk.")
}

// schema for the final output from the LLM - purely creative content
class FinalClip {
  title string @description("A compelling, catchy title for the final composed clip.")
  summary string @description("A brief summary of the clip's key takeaway.")
  justification string @description("The final rationale for why this combination of chunks is a great clip that fulfills the user's prompt.")
}

// Function for Step 1: Is the core chunk relevant?
function EvaluateSeedChunk(seed_chunk: string, user_prompt: string) -> SeedEvaluation {
  client CustomGPT5Mini
  prompt #"
    You are a sharp-witted content strategist. Your task is to evaluate a single transcript chunk to see if it serves its purpose to quickly inform, engage, or entertain.

    The user is looking for content about: "{{user_prompt}}"

    Evaluate the following SEED CHUNK:
    ---
    {{seed_chunk}}
    ---

    INSTRUCTIONS:
    1.  **Reasoning:** Analyze the chunk. Is it self-contained? Does it have a strong hook or a key insight? Is it emotionally resonant or surprising? Most importantly, how well does it match the user's prompt?
    2.  **Decision & Score:** Based on your reasoning, decide if it's relevant (`is_relevant`) and provide a strict `relevance_score`. A clip must be excellent to be considered relevant.

    Output ONLY in the specified JSON format.
    {{ ctx.output_format }}
  "#
}

// Function for Step 2: Does the surrounding context add critical value?
function EvaluateContextChunk(seed_chunk: string, context_chunk: string, context_position: "BEFORE" | "AFTER", user_prompt: string) -> ContextDecision {
  client CustomGPT5Mini
  prompt #"
    You are a discerning film editor. You must decide if adding a contextual chunk is ESSENTIAL for a core clip's success. Be extremely conservative. Most of the time, you will say NO.

    The user's goal is to find clips about: "{{user_prompt}}"
    The SEED CHUNK we are building around is:
    ---
    {{seed_chunk}}
    ---

    We are considering adding the following CONTEXT CHUNK which comes {{context_position}} the seed chunk:
    ---
    {{context_chunk}}
    ---

    INSTRUCTIONS:
    1.  **Reasoning:** Does the context chunk provide CRITICAL setup (if BEFORE) or a vital CONCLUSION (if AFTER)? If the seed chunk works perfectly fine without it, the context is not needed. Does adding it make the clip significantly better or just longer?
    2.  **Decision & Score:** Based on this strict analysis, decide if it `should_include` and provide an `inclusion_score`. Only include context if the score is very high (e.g., >0.8).

    Output ONLY in the specified JSON format.
    {{ ctx.output_format }}
  "#
}

// Function for Step 3: Polish and finalize the composed clip.
function FinalizeClip(full_text: string, user_prompt: string) -> FinalClip {
  client CustomGPT5Mini
  prompt #"
    You are an Executive Producer. You have the final text for a video clip. Your job is to give it a title, a summary, and a final justification.

    The user's original request was: "{{user_prompt}}"

    COMPOSED CLIP TEXT:
    ---
    {{full_text}}
    ---

    INSTRUCTIONS:
    1.  **Title & Summary:** Write a catchy `title` and a concise `summary`.
    2.  **Justification:** Write a final `justification` explaining why this clip is a great fit for the user's request.

    Output ONLY in the specified JSON format.
    {{ ctx.output_format }}
  "#
}

