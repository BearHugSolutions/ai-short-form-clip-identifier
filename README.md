# Short-Form Clip Extraction Pipeline

An AI-powered pipeline for extracting engaging short-form clips from long-form YouTube content. This system downloads videos, semantically analyzes transcripts using LLMs, identifies compelling clips, and generates EDL files ready for video editing workflows.

## 🎯 Overview

This pipeline transforms long-form YouTube content into a curated collection of short, engaging clips suitable for TikTok, Instagram Reels, or YouTube Shorts. It uses a multi-layer AI approach combining semantic chunking, strategic content analysis, and precision extraction.

### Pipeline Flow

```
YouTube URL → Download Video & Transcript → Semantic Chunking → 
AI Content Analysis → Clip Identification → EDL Generation → Video Editor
```

## 📁 Repository Structure

### Core Pipeline Scripts

**`01-youtube_downloader.py`**
- Downloads YouTube videos using `yt-dlp`
- Extracts transcripts with timestamps using `youtube-transcript-api`
- Creates three files: video file, transcript with timestamps, plain transcript
- **Tunable**: Video quality preferences, output formats

**`02-semantic_chunker.py`**
- Chunks transcripts using semantic similarity (sentence-transformers)
- Creates multi-granularity chunks for hierarchical analysis
- Uses sliding window approach with dynamic threshold optimization
- **Tunable**: 
  - Embedding model (`--model`)
  - Token limits (`--min-tokens`, `--max-tokens`)
  - Window size (`--window-size`)
  - Threshold behavior (`--no-dynamic`)

**`03-hybrid_clip_identification.py`**
- Multi-layer extraction pipeline combining strategy and precision
- Phase 1: Maps content structure using coarse chunks
- Phase 2: Creates targeted extraction strategy
- Phase 3: Theme-based chunk selection
- Phase 4: Precision clip extraction with exact timestamps
- Phase 5: Timestamp resolution and validation
- **Tunable**: All phases controlled via BAML prompts (see `baml_src/`)

**`03-relevance_retrieval.py`**
- CLI interface for running the hybrid extraction pipeline
- Interactive directory and file selection using `inquirer`
- Orchestrates the entire clip identification process
- **Tunable**:
  - User query/prompt (`--prompt`)
  - Target clip count (`--target-clips`)
  - Interactive mode (`--no-inquirer`)

**`03-timestamp_extraction.py`**
- Shared utility module for timestamp matching
- Uses fuzzy matching with configurable error tolerance
- Validates timestamp ordering and logic
- **Tunable**: Error tolerance threshold in function calls

**`04-construct-edl.py`**
- Generates CMX 3600 EDL files from clip JSON
- Detects and handles overlapping clips
- Interactive selection of transcript directories and clips folders
- **Tunable**:
  - FPS (`--fps`)
  - Directory/file selection

### BAML Configuration

**`baml_src/clients.baml`**
- Defines LLM client configurations
- **Tunable**: Model selection, temperature, API endpoints

**`baml_src/content_strategy.baml`**
- Multi-layer content analysis prompts (optimized for token efficiency)
- Functions: `MapContentStructure`, `CreateExtractionStrategy`, `EvaluateChunkForTheme`, `ExtractPrecisionClips`
- **Tunable**: All prompt templates, evaluation criteria, scoring thresholds

**`baml_src/clip_composition.baml`**
- Legacy composition approach (replaced by hybrid pipeline)
- Functions: `EvaluateSeedChunk`, `EvaluateContextChunk`, `FinalizeClip`
- **Tunable**: Evaluation criteria, context inclusion logic

**`baml_src/generators.baml`**
- Code generation configuration for BAML client

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- VSCode or Cursor (recommended for BAML development)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd short-form
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install BAML VSCode Extension**
   - Open VSCode/Cursor
   - Install: [BAML Extension](https://marketplace.visualstudio.com/items?itemName=boundary.baml-extension)
   - Provides syntax highlighting, testing playground, and prompt previews

5. **Configure environment variables**
```bash
# Create .env file
touch .env

# Add your API keys
echo "OPENAI_API_KEY=your_key_here" >> .env
```

6. **Generate BAML client**
```bash
baml-cli generate
```

This creates the `baml_client/` directory with Python bindings for your BAML functions.

### Basic Usage

**Step 1: Download a YouTube video**
```bash
python 01-youtube_downloader.py "MyVideo" "https://www.youtube.com/watch?v=VIDEO_ID"
```

**Step 2: Create semantic chunks**
```bash
python 02-semantic_chunker.py
# Interactive selection menu will appear
```

**Step 3: Extract clips with AI**
```bash
python 03-relevance_retrieval.py --prompt "Find clips about product announcements and demos"
```

**Step 4: Generate EDL for video editing**
```bash
python 04-construct-edl.py
# Interactive selection menu will appear
```

## 🎛️ Tunable Parameters

### Pipeline-Level Parameters

| Area | Parameter | Location | Impact |
|------|-----------|----------|--------|
| **Semantic Chunking** | Token limits | `02-semantic_chunker.py` flags | Chunk granularity |
| **Content Strategy** | User query | `03-relevance_retrieval.py --prompt` | What content to extract |
| **Clip Count** | Target clips | `03-relevance_retrieval.py --target-clips` | How many clips to generate |
| **LLM Behavior** | All prompts | `baml_src/*.baml` | Extraction logic and quality |
| **Model Selection** | Client config | `baml_src/clients.baml` | Cost/quality tradeoff |

### Key Configuration Files

**For LLM Tuning**: Edit `baml_src/content_strategy.baml`
- Adjust evaluation criteria in prompts
- Modify scoring thresholds
- Change instruction specificity
- Tune for different content types (podcasts vs. tutorials vs. talks)

**For Chunking**: Command-line flags in `02-semantic_chunker.py`
```bash
--model BAAI/bge-m3           # Embedding model
--min-tokens 100              # Minimum chunk size
--max-tokens 300              # Maximum chunk size
--window-size 5               # Context window for similarity
```

**For EDL Generation**: `04-construct-edl.py`
```bash
--fps 29.97                   # Frame rate
```

## 🏗️ Architecture Decisions

### Multi-Layer Approach
The hybrid extraction pipeline uses **coarse chunks** (800-1500 tokens) for strategic planning and **fine chunks** (100-300 tokens) for precision extraction. This balances computational efficiency with extraction accuracy.

### Token Optimization
`content_strategy.baml` has been optimized to minimize token usage by removing unused fields and consolidating prompts, reducing costs by ~40% while maintaining quality.

### Modular Design
Each script can be run independently, making it easy to:
- Test individual components
- Swap out implementations (e.g., different semantic chunking approaches)
- Add new extraction strategies

## 🔧 Advanced Customization

### Creating Custom Extraction Strategies

Edit `baml_src/content_strategy.baml` to define new extraction approaches:

```baml
function CustomExtractionStrategy(chunks: string[], criteria: string) -> CustomOutput {
  client CustomGPT5Mini
  prompt #"
    Your custom prompt here...
    {{ ctx.output_format }}
  "#
}
```

Then call it from your Python code:
```python
from baml_client.async_client import b
result = await b.CustomExtractionStrategy(chunks=my_chunks, criteria="...")
```

### Adding New Evaluation Metrics

Extend the schemas in `content_strategy.baml`:
```baml
class EnhancedClip {
  // Existing fields...
  virality_score float @description("Predicted social media performance")
  emotion_detected string @description("Primary emotional tone")
}
```

## 📊 Output Files

Each run of the pipeline creates:

- **Download stage**: `{name}_video.mp4`, `{name}_transcript.txt`, `{name}_with_timestamps.txt`
- **Chunking stage**: `{name}/chunks/chunk_*.txt`, `chunks_metadata.json`, `chunks_summary.txt`
- **Extraction stage**: `{name}/clips_hybrid/*.json`, `all_clips_hybrid.json`, `hybrid_metadata.json`
- **EDL stage**: `{name}_clips_hybrid.edl`

## 🐛 Debugging

- **BAML Issues**: Use the VSCode extension's testing playground
- **Timestamp Mismatches**: Check `hybrid_metadata.json` failure log
- **Empty Results**: Review `chunks_summary.txt` and verify input transcript quality

## 📝 Development Notes

**BAML Auto-generation**: The VSCode extension automatically runs `baml-cli generate` when you save `.baml` files, keeping your Python bindings in sync.

**Type Safety**: BAML converts all schemas into Pydantic models, providing full type checking in Python.

## 🤝 Contributing

Areas for improvement:
- Alternative semantic chunking algorithms
- More sophisticated clip quality scoring
- Additional EDL format support
- Frontend interface for parameter tuning
- Batch processing capabilities

## 📄 License

[Your License Here]
