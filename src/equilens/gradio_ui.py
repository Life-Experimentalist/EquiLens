"""
Enhanced Gradio GUI for EquiLens - AI Bias Detection Platform

A comprehensive, professional web interface using Gradio that replaces the TUI.
Provides full functionality for bias detection workflows with real-time progress tracking.
"""

import subprocess
from pathlib import Path

import gradio as gr
from gradio.themes import Soft

from equilens.core.manager import EquiLensManager
from equilens.core.ollama_config import get_ollama_url

# # Helper to get Ollama API URL
# def get_ollama_url() -> str:
#     """Return the Ollama API URL, using environment variable or default."""
#     return os.environ.get("OLLAMA_API_URL", "http://localhost:11434")


# Initialize the manager
manager = EquiLensManager()

# Global state for progress tracking
progress_state = {"current_operation": None, "progress": 0, "status": "Ready"}


def get_system_info() -> str:
    """Get enhanced system information"""
    try:
        status = manager.check_system_status()

        # Enhanced status formatting with emojis and better structure
        output = "🔍 **EquiLens System Status Dashboard**\n"
        output += "=" * 50 + "\n\n"

        # GPU Information
        gpu_info = status["gpu"]
        output += "🎮 **GPU & Acceleration:**\n"
        if gpu_info["gpu_available"]:
            output += "  ✅ **NVIDIA GPU Detected** - Hardware acceleration available\n"
            output += f"  ✅ **NVIDIA Driver:** {'Ready' if gpu_info['nvidia_driver'] else 'Missing'}\n"
            output += f"  ✅ **CUDA Runtime:** {'Available' if gpu_info['cuda_runtime'] else 'Not found'}\n"
            output += "  🚀 **Performance:** 5-10x faster inference expected\n"
        else:
            output += "  ⚠️ **CPU Mode** - No GPU acceleration (slower inference)\n"
            output += "  💡 **Tip:** Install NVIDIA drivers for better performance\n"

        # Docker Services
        docker_info = status["docker"]
        ollama_url = get_ollama_url()
        if docker_info["docker_available"]:
            output += "  ✅ **Docker:** Available and running\n"
            if docker_info["ollama_accessible"]:
                output += f"  ✅ **Ollama API:** Connected ({ollama_url})\n"
                try:
                    import requests

                    response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        models = response.json().get("models", [])
                        output += (
                            f"  📦 **Models Available:** {len(models)} models ready\n"
                        )
                except Exception:
                    pass
            else:
                output += "  ⚠️ **Ollama API:** Not accessible - try starting services\n"

            if docker_info["containers"]:
                output += "  🔄 **Active Containers:**\n"
                for container in docker_info["containers"][:3]:  # Show max 3
                    output += f"    • {container['name']}: {container['status']}\n"
        else:
            output += "  ❌ **Docker:** Not available or not running\n"
            output += "  💡 **Setup:** Please install Docker Desktop\n"

        # System Resources
        output += "\n💻 **System Resources:**\n"
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            output += f"  🔥 **CPU Usage:** {cpu_percent:.1f}%\n"
            output += f"  🧠 **Memory:** {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)\n"
            output += f"  💾 **Disk Space:** {disk.free / (1024**3):.1f}GB free\n"
        except ImportError:
            output += "  📊 **Monitoring:** Install psutil for detailed metrics\n"

        # Recommendations
        output += "\n💡 **Recommendation:**\n"
        output += f"  {status['recommendation']}\n"

        # Quick Actions
        output += "\n🚀 **Quick Actions:**\n"
        if not docker_info["ollama_accessible"]:
            output += "  1️⃣ Click 'Start Services' to begin\n"
            output += "  2️⃣ Pull a model (try 'phi3:mini' for quick start)\n"
        else:
            output += "  1️⃣ Check available models in Models tab\n"
            output += "  2️⃣ Generate test corpus in Corpus tab\n"
            output += "  3️⃣ Run bias audit in Audit tab\n"

        return output

    except Exception as e:
        return f"❌ **Error getting system status:** {str(e)}\n\n💡 Try refreshing or check your installation."


def start_services() -> str:
    """Start EquiLens services"""
    try:
        success = manager.start_services()
        if success:
            return "✅ Services started successfully!\n\n💡 Next steps:\n  • List models to see available options\n  • Pull a model if needed\n  • Generate corpus and run audit"
        else:
            return "❌ Failed to start services. Check Docker installation and permissions."
    except Exception as e:
        return f"❌ Error starting services: {str(e)}"


def stop_services() -> str:
    """Stop EquiLens services"""
    try:
        success = manager.stop_services()
        if success:
            return "✅ Services stopped successfully!"
        else:
            return "❌ Failed to stop services."
    except Exception as e:
        return f"❌ Error stopping services: {str(e)}"


def list_models() -> str:
    """List available Ollama models"""
    try:
        import requests

        ollama_url = get_ollama_url()
        # Try to get models from Ollama API
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            if models:
                output = "📋 Available Models:\n"
                output += "=" * 30 + "\n"
                for model in models:
                    size_gb = model.get("size", 0) / (1024**3)
                    modified = model.get("modified_at", "Unknown")[:10]  # Just date
                    output += (
                        f"• {model['name']} ({size_gb:.1f}GB) - Modified: {modified}\n"
                    )
                return output
            else:
                return "📋 No models installed\n\n💡 Use 'Pull Model' to download a model first"
        else:
            return "❌ Could not connect to Ollama API. Make sure services are started."
    except Exception as e:
        return f"❌ Error listing models: {str(e)}"


def pull_model(model_name: str) -> str:
    """Pull/download a model"""
    if not model_name.strip():
        return "❌ Please enter a model name"

    try:
        success = manager.pull_model(model_name.strip())
        if success:
            return f"✅ Model '{model_name}' downloaded successfully!"
        else:
            return f"❌ Failed to download model '{model_name}'. Check the model name and try again."
    except Exception as e:
        return f"❌ Error pulling model: {str(e)}"


def generate_corpus(config_file: str | None = None) -> str:
    """Generate test corpus"""
    try:
        if config_file and config_file.strip():
            # Use specific config file
            success = manager.generate_corpus(config_file.strip())
            if success:
                return f"✅ Corpus generated with config '{config_file}'"
            else:
                return f"❌ Failed to generate corpus with config '{config_file}'"
        else:
            # Use interactive mode
            result = subprocess.run(
                ["python", "src/Phase1_CorpusGenerator/generate_corpus.py"],
                capture_output=True,
                text=True,
                cwd=str(manager.project_root),
            )

            if result.returncode == 0:
                return "✅ Corpus generated successfully!\n\n📁 Check src/Phase1_CorpusGenerator/corpus/ directory for generated files."
            else:
                return f"❌ Corpus generation failed:\n{result.stderr}"
    except Exception as e:
        return f"❌ Error generating corpus: {str(e)}"


def run_audit(model_name: str, corpus_file: str, output_dir: str = "results") -> str:
    """Run bias audit"""
    if not model_name.strip():
        return "❌ Please enter a model name"
    if not corpus_file.strip():
        return "❌ Please enter a corpus file path"

    try:
        # Check if corpus file exists
        corpus_path = Path(corpus_file.strip())
        if not corpus_path.exists():
            return f"❌ Corpus file not found: {corpus_file}"

        # Run audit using subprocess to get better control
        cmd = [
            "python",
            "src/Phase2_ModelAuditor/audit_model.py",
            "--model",
            model_name.strip(),
            "--corpus",
            str(corpus_path),
            "--output-dir",
            output_dir.strip() if output_dir.strip() else "results",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(manager.project_root)
        )

        if result.returncode == 0:
            return f"✅ Audit completed successfully!\n\n📊 Results saved to: {output_dir}\n\n💡 Use 'Analyze Results' to view the analysis."
        else:
            return f"❌ Audit failed:\n{result.stderr[:1000]}..."  # Limit error output

    except Exception as e:
        return f"❌ Error running audit: {str(e)}"


def analyze_results(results_file: str) -> str:
    """Analyze bias audit results"""
    if not results_file.strip():
        return "❌ Please enter a results file path"

    try:
        results_path = Path(results_file.strip())
        if not results_path.exists():
            return f"❌ Results file not found: {results_file}"

        # Run analysis
        result = subprocess.run(
            ["python", "src/Phase3_Analysis/analyze_results.py", str(results_path)],
            capture_output=True,
            text=True,
            cwd=str(manager.project_root),
        )

        if result.returncode == 0:
            return "✅ Analysis completed successfully!\n\n📊 Check for bias_report.png and console output for detailed results."
        else:
            return f"❌ Analysis failed:\n{result.stderr[:1000]}..."

    except Exception as e:
        return f"❌ Error analyzing results: {str(e)}"


def get_corpus_files() -> list[str]:
    """Get list of available corpus files"""
    corpus_dir = Path("src/Phase1_CorpusGenerator/corpus")
    if corpus_dir.exists():
        return [str(f) for f in corpus_dir.glob("*.csv")]
    return []


def get_results_files() -> list[str]:
    """Get list of available results files"""
    results_dir = Path("results")
    files = []
    if results_dir.exists():
        # Check for files in session directories (new structure)
        for session_dir in results_dir.iterdir():
            if session_dir.is_dir():
                files.extend([str(f) for f in session_dir.glob("results_*.csv")])
        # Check for direct files (old structure)
        files.extend([str(f) for f in results_dir.glob("results_*.csv")])
    return sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)


def update_corpus_dropdown():
    """Update corpus dropdown with available files"""
    return gr.Dropdown(choices=get_corpus_files())


def update_results_dropdown():
    """Update results dropdown with available files"""
    return gr.Dropdown(choices=get_results_files())


# Main Gradio interface
def create_interface():
    """Create the main Gradio interface"""

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', sans-serif;
    }
    .tab-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .tab-nav button {
        border-radius: 8px !important;
        margin: 4px !important;
        font-weight: 600 !important;
    }
    .tab-nav button[aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
    }
    .block {
        border-radius: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }
    .gr-button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.3) !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-secondary {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
        color: #333 !important;
        border: none !important;
    }
    .gr-textbox {
        border-radius: 8px !important;
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
    }
    .gr-textbox:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-good { background-color: #10b981; }
    .status-warning { background-color: #f59e0b; }
    .status-error { background-color: #ef4444; }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    """

    with gr.Blocks(
        theme=Soft(primary_hue="blue", secondary_hue="purple", neutral_hue="slate"),
        css=custom_css,
        title="EquiLens - AI Bias Detection Platform",
    ) as demo:
        # Enhanced Header with Hero Section
        with gr.Row():
            gr.Markdown("""
            <div class="hero-section">
                <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(45deg, #fff, #e0e7ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🔍 EquiLens
                </h1>
                <h2 style="font-size: 1.5rem; margin: 0.5rem 0; opacity: 0.9;">
                    AI Bias Detection Platform
                </h2>
                <p style="font-size: 1.1rem; margin: 1rem 0; max-width: 600px; margin-left: auto; margin-right: auto; opacity: 0.8;">
                    A comprehensive web interface for detecting and analyzing bias in AI language models.
                    Built for researchers, developers, and organizations committed to responsible AI.
                </p>
            </div>
            """)

        with gr.Tabs():
            # Enhanced Dashboard Tab
            with gr.Tab("🏠 Dashboard", id="dashboard"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## 🔍 System Status & Management")
                        gr.Markdown("""
                        Monitor your EquiLens installation and manage core services from this central dashboard.
                        """)

                        with gr.Row():
                            status_btn = gr.Button(
                                "🔍 Check Status", variant="primary", size="lg"
                            )
                            start_btn = gr.Button(
                                "🚀 Start Services", variant="secondary", size="lg"
                            )
                            stop_btn = gr.Button(
                                "🛑 Stop Services", variant="secondary", size="lg"
                            )

                        status_output = gr.Textbox(
                            label="📊 System Status",
                            lines=20,
                            max_lines=25,
                            interactive=False,
                            placeholder="Click 'Check Status' to view system information...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🚀 Quick Start Guide")
                        gr.Markdown("""
                        <div class="feature-card">
                            <strong>Step 1:</strong> Check system status and ensure Docker is running
                        </div>
                        <div class="feature-card">
                            <strong>Step 2:</strong> Start EquiLens services (Ollama + GPU support)
                        </div>
                        <div class="feature-card">
                            <strong>Step 3:</strong> Pull a language model (try phi3:mini for quick start)
                        </div>
                        <div class="feature-card">
                            <strong>Step 4:</strong> Generate test corpus for bias evaluation
                        </div>
                        <div class="feature-card">
                            <strong>Step 5:</strong> Run comprehensive bias audit and analyze results
                        </div>
                        """)

                        gr.Markdown("### 📋 System Requirements")
                        gr.Markdown("""
                        - **Docker Desktop** (with GPU support if available)
                        - **Python 3.11+** (3.13 recommended)
                        - **4GB+ RAM** (8GB+ recommended)
                        - **2GB+ disk space** for models
                        """)

                # Connect dashboard buttons
                status_btn.click(get_system_info, outputs=status_output)
                start_btn.click(start_services, outputs=status_output)
                stop_btn.click(stop_services, outputs=status_output)

            #         # Enhanced Model Management Tab
            #         with gr.Tab("🎯 Models", id="models"):
            #             gr.Markdown("## 🎯 Model Management & Downloads")
            #             gr.Markdown("""
            #             Manage Ollama models for bias testing. Start services first if you haven't already.
            #             Models are the foundation of bias detection - choose wisely based on your research needs.
            #             """)

            #             with gr.Row():
            #                 with gr.Column(scale=2):
            #                     with gr.Row():
            #                         list_models_btn = gr.Button("📋 List Available Models", variant="primary", size="lg")
            #                         refresh_models_btn = gr.Button("🔄 Refresh", variant="secondary")

            #                     models_output = gr.Textbox(
            #                         label="📦 Model Information",
            #                         lines=12,
            #                         max_lines=15,
            #                         interactive=False,
            #                         placeholder="Click 'List Available Models' to see installed models..."
            #                     )

            #                 with gr.Column(scale=1):
            #                     gr.Markdown("### ⬇️ Download New Model")
            #                     pull_model_input = gr.Textbox(
            #                         label="Model Name",
            #                         placeholder="e.g., phi3:mini, llama2:latest, mistral:7b",
            #                         info="Enter the exact model name from Ollama library"
            #                     )
            #                     pull_model_btn = gr.Button("⬇️ Pull Model", variant="primary", size="lg")

            #                     gr.Markdown("### 💡 Recommended Models")
            #                     gr.Markdown("""
            #                     **Quick Start:**
            #                     - `phi3:mini` (3.8GB) - Fast, efficient
            #                     - `qwen2:0.5b` (394MB) - Ultra lightweight

            #                     **Research Grade:**
            #                     - `llama2:7b` (3.8GB) - Well-studied
            #                     - `mistral:7b` (4.1GB) - High quality

            #                     **Advanced:**
            #                     - `llama2:13b` (7.3GB) - Better accuracy
            #                     - `codellama:7b` (3.8GB) - Code-focused
            #                     """)

            #             # Connect model buttons
            #             list_models_btn.click(list_models, outputs=models_output)
            #             refresh_models_btn.click(list_models, outputs=models_output)
            #             pull_model_btn.click(pull_model, inputs=pull_model_input, outputs=models_output)

            #         # Enhanced Corpus Generation Tab
            #         with gr.Tab("📝 Corpus Generation", id="corpus"):
            #             gr.Markdown("## 📝 Test Corpus Generation")
            #             gr.Markdown("""
            #             Generate bias evaluation test sets using configurable templates and word lists.
            #             The corpus is the foundation of bias detection - it defines what biases you'll test for.
            #             """)

            #             with gr.Row():
            #                 with gr.Column(scale=2):
            #                     with gr.Row():
            #                         generate_corpus_btn = gr.Button("📝 Generate New Corpus", variant="primary", size="lg")
            #                         refresh_corpus_btn = gr.Button("🔄 Refresh Files", variant="secondary")

            #                     corpus_config_input = gr.Textbox(
            #                         label="📋 Config File (Optional)",
            #                         placeholder="Leave empty for interactive mode",
            #                         info="Path to configuration file, or leave empty for guided setup"
            #                     )

            #                     corpus_output = gr.Textbox(
            #                         label="📊 Generation Output",
            #                         lines=10,
            #                         interactive=False,
            #                         placeholder="Click 'Generate New Corpus' to start the process..."
            #                     )

            #                 with gr.Column(scale=1):
            #                     gr.Markdown("### 📚 Available Corpus Files")
            #                     available_corpus = gr.Dropdown(
            #                         label="Generated Corpus Files",
            #                         choices=get_corpus_files(),
            #                         info="Previously generated corpus files",
            #                         interactive=True
            #                     )

            #                     gr.Markdown("### 🎯 Bias Categories")
            #                     gr.Markdown("""
            #                     **Built-in Categories:**
            #                     - **Gender Bias** - Names, pronouns, roles
            #                     - **Racial/Ethnic** - Demographics, cultural
            #                     - **Professional** - Career assumptions
            #                     - **Age Bias** - Generational stereotypes
            #                     - **Religious** - Faith-based biases
            #                     - **Socioeconomic** - Class assumptions

            #                     **Custom Categories:**
            #                     Configure your own bias detection templates for domain-specific research.
            #                     """)

            #             # Connect corpus buttons
            #             generate_corpus_btn.click(generate_corpus, inputs=corpus_config_input, outputs=corpus_output)
            #             refresh_corpus_btn.click(lambda: gr.Dropdown(choices=get_corpus_files()), outputs=available_corpus)

            #         # Enhanced Audit Tab
            #         with gr.Tab("🔍 Bias Audit", id="audit"):
            #             gr.Markdown("## 🔍 Bias Audit Execution")
            #             gr.Markdown("""
            #             Run comprehensive bias evaluation against language models using generated test corpus.
            #             This is where the actual bias detection happens - be patient, quality analysis takes time.
            #             """)

            #             with gr.Row():
            #                 with gr.Column(scale=2):
            #                     with gr.Row():
            #                         audit_model_input = gr.Textbox(
            #                             label="🤖 Model Name",
            #                             placeholder="e.g., llama2:latest, phi3:mini",
            #                             info="Name of the model to audit (must be pulled first)"
            #                         )
            #                         audit_output_dir = gr.Textbox(
            #                             label="📁 Output Directory",
            #                             value="results",
            #                             info="Directory to save audit results"
            #                         )

            #                     audit_corpus_input = gr.Dropdown(
            #                         label="📋 Corpus File",
            #                         choices=get_corpus_files(),
            #                         info="Select a generated corpus file"
            #                     )

            #                     with gr.Row():
            #                         run_audit_btn = gr.Button("🔍 Run Bias Audit", variant="primary", size="lg")
            #                         refresh_audit_corpus_btn = gr.Button("🔄 Refresh Corpus", variant="secondary")

            #                     audit_output = gr.Textbox(
            #                         label="⚡ Audit Progress & Results",
            #                         lines=12,
            #                         interactive=False,
            #                         placeholder="Configure your audit parameters and click 'Run Bias Audit'..."
            #                     )

            #                 with gr.Column(scale=1):
            #                     gr.Markdown("### ⚡ Audit Process")
            #                     gr.Markdown("""
            #                     **Phase 1:** Model Loading
            #                     - Initializes the selected model
            #                     - Configures generation parameters

            #                     **Phase 2:** Corpus Processing
            #                     - Reads test sentences
            #                     - Prepares evaluation prompts

            #                     **Phase 3:** Response Generation
            #                     - Sends prompts to model
            #                     - Collects and stores responses

            #                     **Phase 4:** Data Collection
            #                     - Organizes results by category
            #                     - Generates performance metrics
            #                     """)

            #                     gr.Markdown("### 🕐 Time Estimates")
            #                     gr.Markdown("""
            #                     **Small corpus (100 prompts):**
            #                     - CPU: ~15-30 minutes
            #                     - GPU: ~5-10 minutes

            #                     **Medium corpus (500 prompts):**
            #                     - CPU: ~1-2 hours
            #                     - GPU: ~15-30 minutes

            #                     **Large corpus (1000+ prompts):**
            #                     - CPU: 2+ hours
            #                     - GPU: ~30-60 minutes
            #                     """)

            #             # Connect audit buttons
            #             run_audit_btn.click(
            #                 run_audit,
            #                 inputs=[audit_model_input, audit_corpus_input, audit_output_dir],
            #                 outputs=audit_output
            #             )
            #             refresh_audit_corpus_btn.click(
            #                 lambda: gr.Dropdown(choices=get_corpus_files()),
            #                 outputs=audit_corpus_input
            #             )

            #         # Enhanced Analysis Tab
            #         with gr.Tab("📊 Results Analysis", id="analysis"):
            #             gr.Markdown("## 📊 Bias Analysis & Visualization")
            #             gr.Markdown("""
            #             Analyze audit results to identify bias patterns and generate comprehensive reports.
            #             Transform raw model responses into actionable insights and publication-ready visualizations.
            #             """)

            #             with gr.Row():
            #                 with gr.Column(scale=2):
            #                     with gr.Row():
            #                         results_file_input = gr.Dropdown(
            #                             label="📈 Results File",
            #                             choices=get_results_files(),
            #                             info="Select audit results to analyze"
            #                         )
            #                         refresh_results_btn = gr.Button("🔄 Refresh Results", variant="secondary")

            #                     analyze_results_btn = gr.Button("📊 Analyze Results", variant="primary", size="lg")

            #                     results_output = gr.Textbox(
            #                         label="📈 Analysis Results",
            #                         lines=12,
            #                         interactive=False,
            #                         placeholder="Select a results file and click 'Analyze Results' to begin..."
            #                     )

            #                 with gr.Column(scale=1):
            #                     gr.Markdown("### 📈 Analysis Features")
            #                     gr.Markdown("""
            #                     **Statistical Analysis:**
            #                     - Bias magnitude calculations
            #                     - Statistical significance testing
            #                     - Confidence intervals
            #                     - Effect size measurements

            #                     **Visualizations:**
            #                     - Bias heatmaps by category
            #                     - Distribution comparisons
            #                     - Trend analysis charts
            #                     - Word cloud visualizations

            #                     **Reporting:**
            #                     - Comprehensive PDF reports
            #                     - Executive summaries
            #                     - Detailed findings breakdown
            #                     - Recommendations for improvement
            #                     """)

            #                     gr.Markdown("### 🎯 Key Metrics")
            #                     gr.Markdown("""
            #                     - **Bias Score:** Overall bias magnitude
            #                     - **Category Breakdown:** Per-bias analysis
            #                     - **Severity Classification:** Low/Medium/High
            #                     - **Confidence Levels:** Statistical certainty
            #                     - **Comparative Analysis:** Cross-model comparison
            #                     """)

            #             # Connect analysis buttons
            #             analyze_results_btn.click(analyze_results, inputs=results_file_input, outputs=results_output)
            #             refresh_results_btn.click(
            #                 lambda: gr.Dropdown(choices=get_results_files()),
            #                 outputs=results_file_input
            #             )

            #     # Enhanced Footer
            #     gr.Markdown("""
            #     ---
            #     <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 12px; margin-top: 2rem;">
            #         <h3 style="margin: 0; color: #374151;">🔍 EquiLens v1.0 - AI Bias Detection Platform</h3>
            #         <p style="margin: 0.5rem 0; color: #6b7280;">Built with ❤️ for responsible AI development</p>
            #         <div style="margin-top: 1rem;">
            #             <a href="https://github.com/Life-Experimentalist/EquiLens" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">📚 Documentation</a>
            #             <a href="https://github.com/Life-Experimentalist/EquiLens/issues" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">🐛 Report Issues</a>
            #             <a href="https://github.com/Life-Experimentalist/EquiLens" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">⭐ GitHub</a>
            #         </div>
            #     </div>
            #     """)

            # return demo

            # Model Management Tab
            with gr.Tab("🎯 Models", id="models"):
                gr.Markdown("## Model Management")
                gr.Markdown(
                    "Manage Ollama models for bias testing. Start services first if you haven't already."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        list_models_btn = gr.Button(
                            "📋 List Available Models", variant="primary"
                        )
                    with gr.Column(scale=3):
                        pull_model_input = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., llama2:latest, phi3:mini, mistral:latest",
                            info="Enter the exact model name from Ollama library",
                        )
                        pull_model_btn = gr.Button("⬇️ Pull Model", variant="secondary")

                models_output = gr.Textbox(
                    label="Models Output", lines=10, max_lines=15, interactive=False
                )

                # Connect buttons
                list_models_btn.click(list_models, outputs=models_output)
                pull_model_btn.click(
                    pull_model, inputs=pull_model_input, outputs=models_output
                )

                # Add helpful information
                gr.Markdown("""
                ### 💡 Popular Models for Bias Testing:
                - **llama2:latest** - Meta's Llama 2 (good general purpose)
                - **phi3:mini** - Microsoft's Phi-3 (lightweight, fast)
                - **mistral:latest** - Mistral 7B (efficient and capable)
                - **codellama:latest** - Code-focused Llama variant

                **Note:** Larger models provide more nuanced responses but take longer to process.
                """)

            # Corpus Generation Tab
            with gr.Tab("📝 Corpus Generation", id="corpus"):
                gr.Markdown("## Test Corpus Generation")
                gr.Markdown(
                    "Generate bias evaluation test sets using configurable templates and word lists."
                )

                with gr.Row():
                    with gr.Column():
                        corpus_config_input = gr.Textbox(
                            label="Config File (Optional)",
                            placeholder="Leave empty for interactive mode",
                            info="Path to configuration file, or leave empty for guided setup",
                        )
                        generate_corpus_btn = gr.Button(
                            "📝 Generate Corpus", variant="primary"
                        )

                    with gr.Column():
                        refresh_corpus_btn = gr.Button(
                            "🔄 Refresh Available Corpus Files"
                        )
                        available_corpus = gr.Dropdown(
                            label="Available Corpus Files",
                            choices=get_corpus_files(),
                            info="Previously generated corpus files",
                        )

                corpus_output = gr.Textbox(
                    label="Generation Output", lines=8, interactive=False
                )

                # Connect buttons
                generate_corpus_btn.click(
                    generate_corpus, inputs=corpus_config_input, outputs=corpus_output
                )
                refresh_corpus_btn.click(
                    lambda: gr.Dropdown(choices=get_corpus_files()),
                    outputs=available_corpus,
                )

                gr.Markdown("""
                ### 📖 About Corpus Generation:
                The corpus generator creates test sentences designed to evaluate various types of bias:
                - **Gender Bias**: Testing responses to different gendered names and roles
                - **Racial/Ethnic Bias**: Evaluating responses across different demographic groups
                - **Professional Bias**: Testing assumptions about different careers and roles
                - **Age Bias**: Analyzing responses related to different age groups

                Interactive mode will guide you through selecting bias types and configuring parameters.
                """)

            # Audit Tab
            with gr.Tab("🔍 Bias Audit", id="audit"):
                gr.Markdown("## Bias Audit Execution")
                gr.Markdown(
                    "Run comprehensive bias evaluation against language models using generated test corpus."
                )

                with gr.Row():
                    with gr.Column():
                        audit_model_input = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., llama2:latest",
                            info="Name of the model to audit (must be pulled first)",
                        )
                        audit_corpus_input = gr.Dropdown(
                            label="Corpus File",
                            choices=get_corpus_files(),
                            info="Select a generated corpus file",
                        )
                        audit_output_dir = gr.Textbox(
                            label="Output Directory",
                            value="results",
                            info="Directory to save audit results",
                        )

                    with gr.Column():
                        refresh_audit_corpus_btn = gr.Button("🔄 Refresh Corpus List")
                        run_audit_btn = gr.Button(
                            "🔍 Run Bias Audit", variant="primary", size="lg"
                        )

                audit_output = gr.Textbox(
                    label="Audit Output", lines=10, interactive=False
                )

                # Connect buttons
                run_audit_btn.click(
                    run_audit,
                    inputs=[audit_model_input, audit_corpus_input, audit_output_dir],
                    outputs=audit_output,
                )
                refresh_audit_corpus_btn.click(
                    lambda: gr.Dropdown(choices=get_corpus_files()),
                    outputs=audit_corpus_input,
                )

                gr.Markdown("""
                ### ⚡ Audit Process:
                1. **Model Preparation**: Loads the specified model in Ollama
                2. **Corpus Processing**: Reads test sentences from the corpus file
                3. **Response Generation**: Sends prompts to the model and collects responses
                4. **Progress Tracking**: Shows real-time progress with ETA estimates
                5. **Results Storage**: Saves detailed results in CSV format with session management

                **Note**: Audit duration depends on corpus size and model speed. Larger models provide more detailed responses but take longer to process.
                """)

            # Analysis Tab
            with gr.Tab("📊 Results Analysis", id="analysis"):
                gr.Markdown("## Bias Analysis & Visualization")
                gr.Markdown(
                    "Analyze audit results to identify bias patterns and generate comprehensive reports."
                )

                with gr.Row():
                    with gr.Column():
                        results_file_input = gr.Dropdown(
                            label="Results File",
                            choices=get_results_files(),
                            info="Select audit results to analyze",
                        )
                        analyze_results_btn = gr.Button(
                            "📊 Analyze Results", variant="primary"
                        )

                    with gr.Column():
                        refresh_results_btn = gr.Button("🔄 Refresh Results List")

                results_output = gr.Textbox(
                    label="Analysis Output", lines=10, interactive=False
                )

                # Connect buttons
                analyze_results_btn.click(
                    analyze_results, inputs=results_file_input, outputs=results_output
                )
                refresh_results_btn.click(
                    lambda: gr.Dropdown(choices=get_results_files()),
                    outputs=results_file_input,
                )

                gr.Markdown("""
                ### 📈 Analysis Features:
                - **Statistical Summary**: Bias metrics and significance testing
                - **Visualization**: Charts showing bias patterns across different categories
                - **Comparative Analysis**: Compare responses across demographic groups
                - **Report Generation**: Comprehensive PDF reports with findings
                - **Export Options**: Results available in multiple formats

                **Output**: Analysis generates `bias_report.png` with visualizations and detailed console statistics.
                """)

        # Footer
        gr.Markdown("""
        ---
        **EquiLens v1.0** - AI Bias Detection Platform | Built with ❤️ for responsible AI
        📄 [Documentation](https://github.com/Life-Experimentalist/EquiLens) | 🐛 [Report Issues](https://github.com/Life-Experimentalist/EquiLens/issues) | ⭐ [GitHub](https://github.com/Life-Experimentalist/EquiLens)
        """)

    return demo


def main():
    """Main function to launch the Gradio interface"""
    demo = create_interface()

    # Launch with better configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Standard Gradio port
        share=False,  # Set to True if you want a public link
        show_api=False,  # Hide API docs
        show_error=True,  # Show detailed errors
        quiet=False,  # Show startup messages
        inbrowser=True,  # Auto-open browser
    )


if __name__ == "__main__":
    main()
