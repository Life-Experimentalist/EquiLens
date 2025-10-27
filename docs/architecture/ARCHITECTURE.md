# EquiLens System Architecture

This document provides detailed architecture diagrams showing how EquiLens components interact with Ollama and LLM models.

## Table of Contents
- [System Architecture Overview](#system-architecture-overview)
- [Model Interaction Architecture](#model-interaction-architecture)
- [Request-Response Flow](#request-response-flow)
- [Deployment Architecture](#deployment-architecture)

---

## System Architecture Overview

This diagram shows the complete EquiLens system architecture and component interactions:

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI["CLI Interface<br/>(Typer + Rich)"]
        WebUI["Web UI<br/>(Gradio)"]
        TUI["Terminal UI<br/>(Rich TUI)"]
    end

    subgraph "EquiLens Core"
        Manager["EquiLens Manager<br/>(Orchestration)"]

        subgraph "Phase 1: Corpus Generation"
            CorpusGen["Corpus Generator<br/>generate_corpus.py"]
            WordLists["Word Lists<br/>(JSON Schema)"]
            Templates["Prompt Templates<br/>(4 variants)"]
        end

        subgraph "Phase 2: Model Auditing"
            Auditor["Bias Auditor<br/>audit_model.py"]
            EnhancedAuditor["Enhanced Auditor<br/>enhanced_audit_model.py"]
            ProgressTracker["Progress Tracker<br/>(Resume Support)"]
            RetryQueue["Retry Queue<br/>(Error Recovery)"]
        end

        subgraph "Phase 3: Analysis"
            Analyzer["Results Analyzer<br/>analyze_results.py"]
            EnhancedAnalyzer["Enhanced Analyzer<br/>enhanced_analyzer.py"]
            Visualizer["Visualization Engine<br/>(Matplotlib + Seaborn)"]
        end
    end

    subgraph "Ollama Service Layer"
        OllamaAPI["Ollama API Server<br/>:11434/api/generate"]
        ModelManager["Model Manager<br/>(Pull/Load Models)"]

        subgraph "LLM Runtime"
            ModelEngine["Model Inference Engine<br/>(GPU Accelerated)"]
            TokenCounter["Token Counter<br/>(eval_count)"]
            LogProbCalc["Log-Probability Calculator<br/>(surprisal_score)"]
        end

        ModelCache["Model Cache<br/>(/root/.ollama)"]
    end

    subgraph "Data Storage"
        CorpusFiles["Corpus CSV Files<br/>(Phase1/corpus/)"]
        ResultsFiles["Results CSV Files<br/>(results/)"]
        ProgressFiles["Progress JSON Files<br/>(session tracking)"]
        LogFiles["Log Files<br/>(logs/)"]
    end

    %% User interactions
    CLI --> Manager
    WebUI --> Manager
    TUI --> Manager

    %% Phase 1 flow
    Manager --> CorpusGen
    WordLists --> CorpusGen
    Templates --> CorpusGen
    CorpusGen --> CorpusFiles

    %% Phase 2 flow
    Manager --> Auditor
    Manager --> EnhancedAuditor
    CorpusFiles --> Auditor
    CorpusFiles --> EnhancedAuditor
    Auditor --> ProgressTracker
    Auditor --> RetryQueue
    EnhancedAuditor --> ProgressTracker
    EnhancedAuditor --> RetryQueue

    %% Ollama interactions
    Auditor -->|HTTP POST| OllamaAPI
    EnhancedAuditor -->|HTTP POST| OllamaAPI
    OllamaAPI --> ModelEngine
    ModelManager --> ModelCache
    ModelEngine --> TokenCounter
    ModelEngine --> LogProbCalc
    TokenCounter -->|eval_count| OllamaAPI
    LogProbCalc -->|eval_duration| OllamaAPI

    %% Results storage
    Auditor --> ResultsFiles
    Auditor --> ProgressFiles
    EnhancedAuditor --> ResultsFiles
    EnhancedAuditor --> ProgressFiles
    ProgressTracker --> ProgressFiles
    Manager --> LogFiles

    %% Phase 3 flow
    Manager --> Analyzer
    Manager --> EnhancedAnalyzer
    ResultsFiles --> Analyzer
    ResultsFiles --> EnhancedAnalyzer
    Analyzer --> Visualizer
    EnhancedAnalyzer --> Visualizer

```

---

## Model Interaction Architecture

This diagram details how EquiLens interacts with Ollama's API and the LLM model:

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant EquiLens as EquiLens Auditor
    participant Ollama as Ollama API<br/>:11434
    participant Model as LLM Model<br/>(llama2/phi3/etc)
    participant GPU as GPU Accelerator<br/>(CUDA/ROCm)

    Note over User,GPU: Audit Initialization Phase

    User->>EquiLens: Start Audit<br/>(model, corpus, workers)
    EquiLens->>Ollama: Check Service<br/>GET /api/tags
    Ollama-->>EquiLens: Available Models List

    alt Model Not Available
        EquiLens->>Ollama: Pull Model<br/>POST /api/pull
        Ollama->>Model: Download Model Weights
        Model-->>Ollama: Model Ready
        Ollama-->>EquiLens: Pull Complete
    end

    Note over User,GPU: Bias Testing Phase (Per Prompt)

    loop For Each Corpus Prompt
        EquiLens->>EquiLens: Load Prompt from Corpus<br/>(name, profession, trait)

        EquiLens->>Ollama: POST /api/generate<br/>{model, prompt, stream:false, options}

        Note over Ollama,GPU: Model Inference
        Ollama->>Model: Load Model to Memory
        Model->>GPU: Accelerated Inference

        GPU->>Model: Token Probabilities<br/>(per token)
        Model->>Model: Calculate Log-Probabilities<br/>(surprisal = -log P(token))
        Model->>Model: Generate Response Text<br/>(token by token)
        Model->>Model: Count Tokens<br/>(eval_count)
        Model->>Model: Measure Duration<br/>(eval_duration in ns)

        Model-->>Ollama: Response Package:<br/>- response text<br/>- eval_duration<br/>- eval_count<br/>- done: true

        Ollama-->>EquiLens: JSON Response

        EquiLens->>EquiLens: Calculate Surprisal Score<br/>surprisal = eval_duration / eval_count
        EquiLens->>EquiLens: Store Result<br/>(CSV + Progress JSON)

        alt Request Failed
            EquiLens->>EquiLens: Add to Retry Queue
            EquiLens->>EquiLens: Exponential Backoff
            EquiLens->>Ollama: Retry Request
        end
    end

    Note over User,GPU: Completion Phase

    EquiLens->>EquiLens: Generate Analysis Report<br/>(statistics, visualizations)
    EquiLens-->>User: Audit Complete<br/>(results CSV, PNG charts)

```

---

## Request-Response Flow

Detailed breakdown of a single API request showing data transformation:

```mermaid
graph LR
    subgraph "EquiLens Request Preparation"
        A1["Load Corpus Row<br/>name: Emily<br/>profession: engineer<br/>trait: analytical"]
        A2["Build Prompt Text<br/>'Emily, the engineer,<br/>is known for being<br/>very analytical.'"]
        A3["Prepare API Payload<br/>{<br/>model: 'llama2:latest'<br/>prompt: '...'<br/>stream: false<br/>options: {...}<br/>}"]

        A1 --> A2
        A2 --> A3
    end

    subgraph "Ollama API Processing"
        B1["POST Request<br/>http://localhost:11434<br/>/api/generate"]
        B2["Load Model<br/>to GPU Memory"]
        B3["Tokenize Prompt<br/>['Emily', ',', 'the', ...]"]

        A3 -->|HTTP POST| B1
        B1 --> B2
        B2 --> B3
    end

    subgraph "Model Inference"
        C1["Generate Tokens<br/>(Autoregressive)"]
        C2["Calculate Log-Prob<br/>per Token<br/>P(t|context)"]
        C3["Accumulate Metrics<br/>- eval_duration<br/>- eval_count<br/>- surprisal"]

        B3 --> C1
        C1 --> C2
        C2 --> C3
    end

    subgraph "Response Formation"
        D1["Build JSON Response<br/>{<br/>response: 'text...'<br/>eval_duration: ns<br/>eval_count: N<br/>done: true<br/>}"]
        D2["HTTP 200 OK<br/>Content-Type:<br/>application/json"]

        C3 --> D1
        D1 --> D2
    end

    subgraph "EquiLens Processing"
        E1["Parse Response<br/>Extract Fields"]
        E2["Calculate Surprisal<br/>= eval_duration /<br/>eval_count"]
        E3["Store to CSV<br/>+ sentence<br/>+ name_category<br/>+ trait_category<br/>+ surprisal_score<br/>+ timestamp"]

        D2 -->|Response| E1
        E1 --> E2
        E2 --> E3
    end

```

---

## Deployment Architecture

Shows how EquiLens can be deployed in different configurations:

```mermaid
graph TB
    subgraph "Deployment Option 1: Docker Compose (Recommended)"
        DC["Docker Compose<br/>Orchestration"]

        subgraph "Container: equilens-app"
            App["EquiLens Application<br/>Python 3.13"]
            AppVol["Volume Mount<br/>/workspace"]
        end

        subgraph "Container: equilens-ollama"
            Ollama["Ollama Service<br/>:11434"]
            OllamaVol["Persistent Volume<br/>ollama_data<br/>(/root/.ollama)"]
        end

        Network["Bridge Network<br/>172.20.0.0/16<br/>equilens-network"]

        DC --> App
        DC --> Ollama
        App --> Network
        Ollama --> Network
        App -.->|depends_on| Ollama
        App --> AppVol
        Ollama --> OllamaVol
    end

    subgraph "Deployment Option 2: Standalone (Local Development)"
        Local["Local Machine<br/>(Windows/macOS/Linux)"]

        subgraph "Python Environment"
            Venv["Virtual Environment<br/>(.venv)"]
            EquiLensLocal["EquiLens CLI<br/>uv run equilens"]
        end

        subgraph "Ollama Standalone"
            OllamaLocal["Ollama Service<br/>localhost:11434"]
            ModelsLocal["Models Directory<br/>~/.ollama"]
        end

        Local --> Venv
        Local --> OllamaLocal
        Venv --> EquiLensLocal
        OllamaLocal --> ModelsLocal
        EquiLensLocal -.->|HTTP| OllamaLocal
    end

    subgraph "GPU Acceleration (Optional)"
        NVIDIA["NVIDIA GPU<br/>(CUDA)"]
        AMD["AMD GPU<br/>(ROCm)"]
        Apple["Apple Silicon<br/>(Metal)"]

        Ollama -.->|Uses| NVIDIA
        Ollama -.->|Uses| AMD
        Ollama -.->|Uses| Apple
        OllamaLocal -.->|Uses| NVIDIA
        OllamaLocal -.->|Uses| AMD
        OllamaLocal -.->|Uses| Apple
    end

```

---

## Key Integration Points

### 1. **Ollama API Endpoints Used**

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/api/tags` | GET | List available models | Model inventory |
| `/api/pull` | POST | Download model | Pull progress |
| `/api/generate` | POST | Generate completion | Model response + metrics |
| `/api/show` | POST | Model details | Config, parameters |

### 2. **Request Payload Structure**

```json
{
  "model": "llama2:latest",
  "prompt": "Emily, the engineer, is known for being very analytical.",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 100
  }
}
```

### 3. **Response Metrics Extraction**

```json
{
  "response": "Generated text...",
  "eval_duration": 12345678900,  // nanoseconds
  "eval_count": 50,               // token count
  "done": true
}
```

**Surprisal Calculation:**
```python
surprisal_score = eval_duration / eval_count
# Result: average time per token (nanoseconds/token)
# Used as proxy for log-probability surprisal
```

### 4. **Concurrency Model**

- **Sequential Mode:** 1 worker, processes prompts one at a time
- **Concurrent Mode:** 2-10 workers using `ThreadPoolExecutor`
- **Dynamic Scaling:** Adjusts workers based on error rate
- **Retry Queue:** Failed requests queued for exponential backoff retry

### 5. **Data Flow Summary**

```
Corpus CSV → EquiLens Auditor → Ollama API → LLM Model → GPU
                  ↓                                ↓
           Progress JSON ←─────── Results CSV ←────┘
                  ↓
           Analyzer → Visualizations (PNG/HTML)
```

---

## Performance Characteristics

### Typical Request Timings (Llama2 7B)

- **Cold Start:** 5-10s (model load to GPU)
- **Warm Request:** 2-3s per prompt (token generation)
- **Sequential Processing:** ~40-60s for 20 prompts
- **Concurrent (3 workers):** ~20-30s for 20 prompts
- **Concurrent (5 workers):** ~15-20s for 20 prompts

### Resource Usage

- **GPU Memory:** 4-8 GB (depends on model size)
- **CPU:** Minimal (mostly I/O bound)
- **Disk:** ~4 GB per model (Llama2 7B)
- **Network:** Localhost only (no external calls)

---

## Error Handling & Recovery

```mermaid
graph TD
    A[Request to Ollama] --> B{Success?}
    B -->|Yes| C[Process Response]
    B -->|No| D{Retry Count < 3?}
    D -->|Yes| E[Exponential Backoff<br/>2^n seconds]
    D -->|No| F[Add to Retry Queue]
    E --> A
    F --> G[Process Other Prompts]
    G --> H{Batch Complete?}
    H -->|No| G
    H -->|Yes| I[Process Retry Queue]
    I --> A
    C --> J[Save to CSV + JSON]

```

---

## References

- **Ollama API Documentation:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **EquiLens Repository:** https://github.com/Life-Experimentalist/EquiLens
- **Model Context Protocol (MCP):** Used for enhanced interactions
- **Docker Compose Configuration:** `docker-compose.yml`
- **Main Auditor Implementation:** `src/Phase2_ModelAuditor/audit_model.py`
