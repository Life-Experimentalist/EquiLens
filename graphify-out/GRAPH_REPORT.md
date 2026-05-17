# Graph Report - src  (2026-05-18)

## Corpus Check
- Corpus is ~48,710 words - fits in a single context window. You may not need a graph.

## Summary
- 774 nodes · 1348 edges · 37 communities detected
- Extraction: 77% EXTRACTED · 23% INFERRED · 0% AMBIGUOUS · INFERRED: 316 edges (avg confidence: 0.61)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Core Audit Engine|Core Audit Engine]]
- [[_COMMUNITY_CLI Interface|CLI Interface]]
- [[_COMMUNITY_Bias Analytics & Stats|Bias Analytics & Stats]]
- [[_COMMUNITY_Statistical Methods|Statistical Methods]]
- [[_COMMUNITY_Docker & Infrastructure|Docker & Infrastructure]]
- [[_COMMUNITY_FastAPI Backend|FastAPI Backend]]
- [[_COMMUNITY_Enhanced Auditor|Enhanced Auditor]]
- [[_COMMUNITY_Gradio Web UI|Gradio Web UI]]
- [[_COMMUNITY_Service Launcher & Ports|Service Launcher & Ports]]
- [[_COMMUNITY_Configurable Auditor|Configurable Auditor]]
- [[_COMMUNITY_Corpus Generator|Corpus Generator]]
- [[_COMMUNITY_Database & Export|Database & Export]]
- [[_COMMUNITY_Ollama Config|Ollama Config]]
- [[_COMMUNITY_Dual Auditor Runner|Dual Auditor Runner]]
- [[_COMMUNITY_Config Validator|Config Validator]]
- [[_COMMUNITY_Module Group 15|Module Group 15]]
- [[_COMMUNITY_Module Group 16|Module Group 16]]
- [[_COMMUNITY_Module Group 17|Module Group 17]]
- [[_COMMUNITY_Module Group 18|Module Group 18]]
- [[_COMMUNITY_Module Group 19|Module Group 19]]
- [[_COMMUNITY_Module Group 20|Module Group 20]]
- [[_COMMUNITY_Module Group 21|Module Group 21]]
- [[_COMMUNITY_Module Group 22|Module Group 22]]
- [[_COMMUNITY_Module Group 23|Module Group 23]]
- [[_COMMUNITY_Module Group 24|Module Group 24]]
- [[_COMMUNITY_Module Group 25|Module Group 25]]
- [[_COMMUNITY_Module Group 26|Module Group 26]]
- [[_COMMUNITY_Module Group 27|Module Group 27]]
- [[_COMMUNITY_Module Group 28|Module Group 28]]
- [[_COMMUNITY_Module Group 29|Module Group 29]]
- [[_COMMUNITY_Module Group 30|Module Group 30]]
- [[_COMMUNITY_Module Group 31|Module Group 31]]
- [[_COMMUNITY_Module Group 32|Module Group 32]]
- [[_COMMUNITY_Module Group 33|Module Group 33]]
- [[_COMMUNITY_Module Group 34|Module Group 34]]
- [[_COMMUNITY_Module Group 35|Module Group 35]]
- [[_COMMUNITY_Module Group 36|Module Group 36]]

## God Nodes (most connected - your core abstractions)
1. `BiasAnalytics` - 71 edges
2. `EnhancedBiasAuditor` - 62 edges
3. `OllamaLogMonitor` - 61 edges
4. `EquiLensManager` - 60 edges
5. `ModelAuditor` - 39 edges
6. `DockerManager` - 28 edges
7. `GPUManager` - 26 edges
8. `JobDatabase` - 22 edges
9. `EquiLensClient` - 17 edges
10. `BiasAnalytics Class` - 14 edges

## Surprising Connections (you probably didn't know these)
- `find_interrupted_sessions()` --calls--> `load()`  [INFERRED]
  src\equilens\cli.py → src\equilens\telemetry.py
- `audit()` --calls--> `load()`  [INFERRED]
  src\equilens\cli.py → src\equilens\telemetry.py
- `audit()` --calls--> `EnhancedBiasAuditor`  [INFERRED]
  src\equilens\cli.py → src\Phase2_ModelAuditor\enhanced_audit_model.py
- `generate()` --calls--> `generate_corpus()`  [INFERRED]
  src\equilens\cli.py → src\Phase1_CorpusGenerator\generate_corpus.py
- `_run_auto_analytics()` --calls--> `BiasAnalytics`  [INFERRED]
  src\equilens\cli.py → src\Phase3_Analysis\analytics.py

## Hyperedges (group relationships)
- **EquiLens Service Launch Architecture (backend + frontend + ports)** — start_all_main, backend_server_main, gradio_app_main, core_ports_get_service_ports, core_ports_get_backend_port, core_ports_get_frontend_port [EXTRACTED 1.00]
- **Backend Job Lifecycle (create → execute → track → cancel)** — backend_api_app, backend_jobs_run_corpus_generation_job, backend_jobs_run_audit_job, backend_jobs_run_analysis_job, backend_jobs_cancel_job, backend_database_jobdatabase [EXTRACTED 1.00]
- **Core Orchestration Trio (Manager + Docker + GPU)** — core_manager_equilensmanager, core_docker_dockermanager, core_gpu_gpumanager, core_ollama_config_ollamaconfig [EXTRACTED 0.95]
- **Frontend-Backend HTTP Bridge** — gradio_app_equilensclient, backend_api_app, core_ports_get_backend_url [INFERRED 0.90]
- **EquiLens Public Package API** — equilens_init, core_manager_equilensmanager, core_gpu_gpumanager, core_docker_dockermanager [EXTRACTED 1.00]
- **Three-Phase Bias Auditing Pipeline (Phase1 CSV → Phase2 Audit → Phase3 Analysis)** — generate_corpus_AuditCorpusCSV, audit_model_ResultsCSV, analytics_BiasAnalytics, analyze_results_AdvancedAnalysisEngine [INFERRED 0.95]
- **Dual Bias Scoring: Logprobs (primary) and Timing Fallback** — audit_model_LogprobsScoring, audit_model_TimingFallback, analytics_DetectScoreMethod, enhanced_audit_EnhancedBiasAuditor [EXTRACTED 1.00]
- **Phase3 Statistical Analysis Suite (t-test, ANOVA, Cohen's d, CI)** — analytics_PerformStatisticalTests, analytics_CalculateEffectSizes, analytics_CohensD, analytics_CalculateConfidenceIntervals, analytics_MultiCategoryANOVA [EXTRACTED 1.00]
- **Corpus Generation Pipeline (word_lists.json → validate → generate → CSV)** — phase1_readme_WordListsJSON, test_config_Validate, generate_corpus_GenerateSingleCorpus, generate_corpus_AuditCorpusCSV [EXTRACTED 1.00]
- **Phase2 Audit Resilience Features (retry, checkpoint, graceful shutdown, pause)** — audit_model_RetryQueue, audit_model_ProgressJSON, audit_model_GracefulKiller, audit_model_PauseController, audit_model_DynamicConcurrencyScaling [EXTRACTED 1.00]
- **Enhanced Auditor Additional Metrics (sentiment, polarity, normalized surprisal, token count)** — enhanced_audit_SentimentScore, phase2_readme_NormalizedSurprisal, enhanced_audit_RepeatedSampling, enhanced_audit_CalibrationControls [EXTRACTED 0.95]
- **Report Generation Pipeline (stats → visualizations → AI insights → HTML/MD)** — analytics_PerformStatisticalTests, analytics_ViolinPlot, analytics_GenerateAIInsights, analytics_GenerateHTMLReport, analytics_GenerateMarkdownReport, analytics_Jinja2Template [EXTRACTED 1.00]

## Communities

### Community 0 - "Core Audit Engine"
Cohesion: 0.03
Nodes (62): AuditProgress, get_ollama_url(), GracefulKiller, main(), ModelAuditor, PauseController, Handle request results and manage dynamic concurrency scaling, Check system load and resource usage (+54 more)

### Community 1 - "CLI Interface"
Cohesion: 0.03
Nodes (87): audit(), audit_callback(), audit_run(), backend(), find_corpus_files(), find_interrupted_sessions(), format_file_size(), generate() (+79 more)

### Community 2 - "Bias Analytics & Stats"
Cohesion: 0.04
Nodes (44): BiasAnalytics, main(), Generate content using AI model with retry logic and comprehensive error handlin, Extract model name from filename., Generate AI-powered insights for the report with comprehensive error handling., Load and validate results data.          Returns:             bool: True if d, Generate comprehensive HTML report., Generate markdown report with embedded images (optionally AI-enhanced). (+36 more)

### Community 3 - "Statistical Methods"
Cohesion: 0.03
Nodes (77): BiasAnalytics Class, BiasAnalytics.calculate_confidence_intervals, BiasAnalytics.calculate_effect_sizes (Cohen's d), Cohen's d Effect Size Calculation, Comprehensive Dashboard Visualization, BiasAnalytics._detect_corpus_structure, BiasAnalytics._detect_score_method (logprobs vs timing), Effect Size Chart Visualization (+69 more)

### Community 4 - "Docker & Infrastructure"
Cohesion: 0.04
Nodes (40): DockerManager, Docker management and container orchestration, Test Ollama API connectivity, Manages Docker containers and services for EquiLens, Start EquiLens services with GPU detection, # TODO: Future enhancement - EquiLens app container support, # TODO: Future enhancement - EquiLens app container integration, # TODO: Future enhancement - full docker-compose with app container (+32 more)

### Community 5 - "FastAPI Backend"
Cohesion: 0.06
Nodes (57): cancel_job_endpoint(), create_job(), delete_job(), export_results(), get_html_report(), get_job(), get_job_logs(), get_system_status() (+49 more)

### Community 6 - "Enhanced Auditor"
Cohesion: 0.06
Nodes (26): EnhancedBiasAuditor, Save current progress with optional backup creation, Save two CSVs: one sanitized results file and one full responses file., Wrapper to save final sanitized and response CSVs and write a summary., Create a backup of progress file and maintain only 2 most recent backups, Keep only the 2 most recent backup files, Load progress from file, Prepare the final prompt with system instruction if not using system field (+18 more)

### Community 7 - "Gradio Web UI"
Cohesion: 0.07
Nodes (26): cancel_job_action(), EquiLensClient, get_job_status(), list_all_jobs(), list_results_action(), pull_model_action(), EquiLens Gradio Frontend  Pure frontend interface that communicates with the F, Pull an Ollama model. (+18 more)

### Community 8 - "Service Launcher & Ports"
Cohesion: 0.08
Nodes (31): main(), Backend launcher script for EquiLens API., Launch the EquiLens backend API., get_backend_port(), get_backend_url(), get_frontend_port(), get_service_ports(), main() (+23 more)

### Community 9 - "Configurable Auditor"
Cohesion: 0.09
Nodes (19): AuditProgress, ConfigurableEnhancedAuditor, get_safe_presets(), GracefulKiller, main(), Individual test result with timing information, Handle graceful shutdown, Predefined system instruction presets for different auditing scenarios (+11 more)

### Community 10 - "Corpus Generator"
Cohesion: 0.11
Nodes (25): generate_corpus(), generate_single_corpus(), interactive_select_comparison(), _preflight_check(), Generate corpus for a single comparison., Generates CSV file(s) of templated sentences based on word lists in a JSON file., Run strict validator before generating corpus. Abort if validation fails., Prompt user to select comparison(s) to generate. Returns list of comparison name (+17 more)

### Community 11 - "Database & Export"
Cohesion: 0.16
Nodes (21): FastAPI app (backend/api.py), get_connection() (thread-local SQLite), init_db(), JobDatabase (SQLite ORM), backend __init__ Package, cancel_job(), run_analysis_job(), run_audit_job() (+13 more)

### Community 12 - "Ollama Config"
Cohesion: 0.12
Nodes (12): get_ollama_url(), is_running_in_container(), OllamaConfig, Test if Ollama is accessible at given URL.          Args:             url: Ol, Check if Ollama is running as a Docker container.          Returns:, Intelligently determine the correct Ollama URL based on environment., Get detailed information about the detected environment.          Returns:, Clear cached URL to force re-detection (+4 more)

### Community 13 - "Dual Auditor Runner"
Cohesion: 0.47
Nodes (5): datetime_now(), main(), Run both auditors (stable and enhanced) and produce a comparison manifest  Thi, Instantiate auditor class and run its audit method.     Returns path to results, run_auditor()

### Community 14 - "Config Validator"
Cohesion: 1.0
Nodes (2): fail(), validate()

### Community 15 - "Module Group 15"
Cohesion: 1.0
Nodes (1): Main entry point for the EquiLens package

### Community 16 - "Module Group 16"
Cohesion: 1.0
Nodes (1): Path to the captured log file, or None if not started.

### Community 17 - "Module Group 17"
Cohesion: 1.0
Nodes (1): Log source type: 'docker', 'windows', or 'unknown'.

### Community 18 - "Module Group 18"
Cohesion: 1.0
Nodes (1): Create a new job entry.

### Community 19 - "Module Group 19"
Cohesion: 1.0
Nodes (1): Get job details by ID.

### Community 20 - "Module Group 20"
Cohesion: 1.0
Nodes (1): Update job status and details.

### Community 21 - "Module Group 21"
Cohesion: 1.0
Nodes (1): List jobs with optional filtering.

### Community 22 - "Module Group 22"
Cohesion: 1.0
Nodes (1): Add a log entry for a job.

### Community 23 - "Module Group 23"
Cohesion: 1.0
Nodes (1): Get logs for a specific job.

### Community 24 - "Module Group 24"
Cohesion: 1.0
Nodes (1): Delete a job and its logs.

### Community 25 - "Module Group 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Module Group 26"
Cohesion: 1.0
Nodes (1): Get dictionary of safe system instruction presets

### Community 27 - "Module Group 27"
Cohesion: 1.0
Nodes (1): Get dictionary of dangerous presets (for research/documentation)

### Community 28 - "Module Group 28"
Cohesion: 1.0
Nodes (1): Validate if a system instruction is likely safe for bias measurement

### Community 29 - "Module Group 29"
Cohesion: 1.0
Nodes (1): JobCreate Pydantic Model

### Community 30 - "Module Group 30"
Cohesion: 1.0
Nodes (1): JobResponse Pydantic Model

### Community 31 - "Module Group 31"
Cohesion: 1.0
Nodes (1): SystemStatus Pydantic Model

### Community 32 - "Module Group 32"
Cohesion: 1.0
Nodes (1): list_available_comparisons Function

### Community 33 - "Module Group 33"
Cohesion: 1.0
Nodes (1): get_current_comparison Function

### Community 34 - "Module Group 34"
Cohesion: 1.0
Nodes (1): Phase1 CorpusGenerator Package

### Community 35 - "Module Group 35"
Cohesion: 1.0
Nodes (1): CorpusGen README Documentation

### Community 36 - "Module Group 36"
Cohesion: 1.0
Nodes (1): Phase2 ModelAuditor Package

## Knowledge Gaps
- **217 isolated node(s):** `Backend launcher script for EquiLens API.`, `Launch the EquiLens backend API.`, `🌐 Launch the new Gradio web interface (connects to backend)`, `EquiLens Gradio Frontend  Pure frontend interface that communicates with the F`, `Client for communicating with EquiLens backend API.` (+212 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Module Group 15`** (2 nodes): `Main entry point for the EquiLens package`, `__main__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 16`** (1 nodes): `Path to the captured log file, or None if not started.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 17`** (1 nodes): `Log source type: 'docker', 'windows', or 'unknown'.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 18`** (1 nodes): `Create a new job entry.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 19`** (1 nodes): `Get job details by ID.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 20`** (1 nodes): `Update job status and details.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 21`** (1 nodes): `List jobs with optional filtering.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 22`** (1 nodes): `Add a log entry for a job.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 23`** (1 nodes): `Get logs for a specific job.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 24`** (1 nodes): `Delete a job and its logs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 25`** (1 nodes): `package_for_zenodo.ps1`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 26`** (1 nodes): `Get dictionary of safe system instruction presets`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 27`** (1 nodes): `Get dictionary of dangerous presets (for research/documentation)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 28`** (1 nodes): `Validate if a system instruction is likely safe for bias measurement`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 29`** (1 nodes): `JobCreate Pydantic Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 30`** (1 nodes): `JobResponse Pydantic Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 31`** (1 nodes): `SystemStatus Pydantic Model`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 32`** (1 nodes): `list_available_comparisons Function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 33`** (1 nodes): `get_current_comparison Function`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 34`** (1 nodes): `Phase1 CorpusGenerator Package`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 35`** (1 nodes): `CorpusGen README Documentation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Module Group 36`** (1 nodes): `Phase2 ModelAuditor Package`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EquiLensManager` connect `CLI Interface` to `Bias Analytics & Stats`, `Docker & Infrastructure`?**
  _High betweenness centrality (0.238) - this node is a cross-community bridge._
- **Why does `audit()` connect `CLI Interface` to `Core Audit Engine`, `Corpus Generator`, `Enhanced Auditor`?**
  _High betweenness centrality (0.232) - this node is a cross-community bridge._
- **Why does `BiasAnalytics` connect `Bias Analytics & Stats` to `CLI Interface`, `Docker & Infrastructure`?**
  _High betweenness centrality (0.192) - this node is a cross-community bridge._
- **Are the 42 inferred relationships involving `BiasAnalytics` (e.g. with `Modern CLI interface for EquiLens using Typer and Rich  A comprehensive comman` and `Robust corpus discovery:       1) use explicit user_path (if provided)       2`) actually correct?**
  _`BiasAnalytics` has 42 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `EnhancedBiasAuditor` (e.g. with `Modern CLI interface for EquiLens using Typer and Rich  A comprehensive comman` and `Robust corpus discovery:       1) use explicit user_path (if provided)       2`) actually correct?**
  _`EnhancedBiasAuditor` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 50 inferred relationships involving `OllamaLogMonitor` (e.g. with `TqdmLoggingHandler` and `AuditProgress`) actually correct?**
  _`OllamaLogMonitor` has 50 INFERRED edges - model-reasoned connections that need verification._
- **Are the 45 inferred relationships involving `EquiLensManager` (e.g. with `Modern CLI interface for EquiLens using Typer and Rich  A comprehensive comman` and `Robust corpus discovery:       1) use explicit user_path (if provided)       2`) actually correct?**
  _`EquiLensManager` has 45 INFERRED edges - model-reasoned connections that need verification._