# LLM Data Flow in EquiLens Analytics

## Overview

This document explains how data is sent to LLMs for AI-powered insights in EquiLens analytics reports.

## What Gets Sent to the LLM?

### ❌ NOT Sent
- Images (PNG visualizations)
- Full CSV data
- Raw test results
- Chart data

### ✅ Sent (Text Only)
- Statistical summaries
- Numerical metrics
- Category names
- Model name

## Data Format

### Example Prompt for Executive Summary
```
Write a brief executive summary for this bias audit:
Model: llama3.2
Tests: 450
Mean surprisal: 125.42 ns/token

Summarize the bias assessment in 2-3 sentences.
```

### Example Prompt for Recommendations
```
List 3-4 recommendations to reduce bias in this model:
Model: llama3.2
Tests: 450

Format as bullet points.
```

## Token Usage

- **Input**: ~200-300 tokens per prompt
- **Output**: 512 tokens maximum (configurable)
- **Total per report**: ~1500 tokens (both prompts)

## AI Generation Process

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Statistical Analysis Complete                            │
│    - Mean, std dev, effect sizes calculated                 │
│    - Visualizations created as PNG files                    │
│    - Data stored in memory                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Check if AI Available (use_ai=True)                      │
│    - Test connection to Ollama                              │
│    - Get list of available models                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Generate AI Insights (if available)                      │
│    Input: Text summary of statistics                        │
│    ├─► Executive Summary: 2-3 sentence overview             │
│    └─► Recommendations: 3-4 bullet points                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Generate Reports                                         │
│    ├─► HTML Report                                          │
│    │   - Embedded AI insights (or placeholders)             │
│    │   - Base64 encoded visualizations                      │
│    │   - Complete statistical tables                        │
│    │                                                         │
│    └─► Markdown Report                                      │
│        - AI insights (or placeholders)                      │
│        - Linked PNG visualizations                          │
│        - Statistical markdown tables                        │
└─────────────────────────────────────────────────────────────┘
```

## Failure Handling

### If Ollama Unavailable
```python
# Original behavior (pre-update):
# - Entire report generation failed
# - No HTML or Markdown output

# New behavior (post-update):
# - Reports ALWAYS generate
# - AI sections show placeholder text:
ai_insights = {
    "executive_summary": "⚠️ AI-generated summary will appear here when available.",
    "recommendations": "⚠️ AI-generated recommendations will appear here when available."
}
```

### If AI Generation Fails Mid-Process
```python
# Timeout, model error, etc.
try:
    summary = generate_ai_content(prompt)
except Exception as e:
    summary = default_fallback_text
    # Report continues with placeholder
```

## Benefits of This Approach

### ✅ Lightweight
- No large data transfers
- Fast API calls
- Minimal token usage

### ✅ Robust
- Reports never fail
- Graceful degradation
- Can regenerate later

### ✅ Flexible
- AI can be disabled (`--no-ai`)
- Multiple models supported
- Configurable token limits

## Configuration

### Environment Variables
```powershell
# Ollama API endpoint
$env:OLLAMA_BASE_URL = "http://localhost:11434"

# For Docker
$env:OLLAMA_BASE_URL = "http://ollama:11434"
```

### Python Configuration
```python
from Phase3_Analysis.analytics import BiasAnalytics

analyzer = BiasAnalytics(
    results_file="results.csv",
    ollama_url="http://localhost:11434",  # Optional
    report_model="llama3.2:latest",       # Optional (auto-detected)
    ai_num_predict=512                    # Token limit
)

# Generate with AI
analyzer.run_complete_analysis(
    generate_html=True,
    generate_ai_insights=True  # Default: True
)

# Generate without AI
analyzer.run_complete_analysis(
    generate_html=True,
    generate_ai_insights=False  # Skip AI, use placeholders
)
```

## API Call Example

### Request to Ollama
```json
POST http://localhost:11434/api/generate

{
  "model": "llama3.2:latest",
  "prompt": "Write a brief executive summary for this bias audit:\nModel: llama3.2\nTests: 450\nMean surprisal: 125.42 ns/token\n\nSummarize in 2-3 sentences.",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 512,
    "top_p": 0.9
  }
}
```

### Response from Ollama
```json
{
  "model": "llama3.2:latest",
  "created_at": "2025-10-20T12:00:00Z",
  "response": "This bias audit of llama3.2 analyzed 450 test cases with a mean surprisal of 125.42 ns/token. The results suggest moderate bias patterns that warrant further investigation. Specific attention should be paid to gender-profession associations.",
  "done": true,
  "eval_duration": 2500000000,
  "eval_count": 45
}
```

## Error Handling

### Connection Errors
```python
# Backend tries multiple times
max_retries = 3
for attempt in range(max_retries):
    try:
        response = requests.post(ollama_url, json=payload, timeout=60)
        break
    except requests.exceptions.ConnectionError:
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            return fallback_text
```

### Timeout Handling
```python
# Progressive timeout increases
timeout = 60  # Start with 60 seconds

try:
    response = requests.post(url, json=payload, timeout=timeout)
except requests.exceptions.Timeout:
    timeout *= 2  # Double for next attempt
    # Eventually gives up and uses fallback
```

### Model Not Found
```python
# If specified model doesn't exist
if response.status_code == 404:
    return f"⚠️ Model '{model_name}' not found. Please pull: ollama pull {model_name}"
```

## Optimizations

### Prompt Engineering
- Concise prompts (~200 tokens)
- Clear instructions
- Specific format requirements
- No unnecessary context

### Token Management
- Limited to 512 tokens output
- Prevents long-running generations
- Balances quality vs speed

### Model Selection
- Auto-selects fastest available model
- Prefers: llama3.2 > llama3.1 > llama2 > mistral
- Falls back to any available model

## Future Enhancements

Potential improvements for LLM integration:

1. **Structured Output**
   - Use JSON mode for parsing
   - Validate response format
   - Better error recovery

2. **Streaming Responses**
   - Real-time generation display
   - Progress feedback
   - Cancellable generation

3. **Advanced Prompts**
   - Include specific statistics
   - Request detailed analysis
   - Multi-step reasoning

4. **Model Fine-Tuning**
   - Train on bias analysis domain
   - Improve recommendation quality
   - Specialized vocabulary

5. **Caching**
   - Cache common summaries
   - Reuse for similar results
   - Reduce API calls

## Summary

EquiLens sends **only text summaries** to LLMs for analysis:
- No images or raw data
- ~300 tokens per prompt
- 512 token responses
- Fallback to placeholders if AI fails
- Reports always generate successfully

This approach is:
- ✅ Lightweight and fast
- ✅ Privacy-preserving (minimal data sent)
- ✅ Robust (graceful degradation)
- ✅ Flexible (AI optional)
- ✅ Production-ready

---

**Related Documentation:**
- [Backend Architecture](./BACKEND_ARCHITECTURE.md)
- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- [Gradio Quick Start](./GRADIO_QUICKSTART.md)
