"""
Enhanced Gradio Web UI for EquiLens - AI Bias Detection Platform

A comprehensive, professional web interface using Gradio.
Provides full functionality for bias detection workflows with real-time progress tracking.

Optimized for fast Docker builds with BuildKit caching.
"""

import subprocess
from pathlib import Path

import gradio as gr
import requests

# (Remove this import from the global scope. Instead, import get_ollama_url inside the functions that use it.)

try:
    import psutil
except ImportError:
    psutil = None

from equilens.core.manager import EquiLensManager

# Initialize the manager
manager = EquiLensManager()

# Global state for progress tracking
progress_state = {"current_operation": None, "progress": 0, "status": "Ready"}


def get_system_info() -> str:
    """Get enhanced system information"""
    try:
        from equilens.core.ollama_config import get_ollama_url

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
        output += "\n🐳 **Docker & Services:**\n"
        if docker_info["docker_available"]:
            output += "  ✅ **Docker:** Available and running\n"
            if docker_info["ollama_accessible"]:
                output += f"  ✅ **Ollama API:** Connected ({ollama_url})\n"

                # Add model count info
                try:
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
        if psutil:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                output += f"  🔥 **CPU Usage:** {cpu_percent:.1f}%\n"
                output += f"  🧠 **Memory:** {memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB ({memory.percent:.1f}%)\n"
                output += f"  💾 **Disk Space:** {disk.free / (1024**3):.1f}GB free\n"
            except Exception:
                output += "  📊 **Monitoring:** Error getting system metrics\n"
        else:
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
        from equilens.core.ollama_config import get_ollama_url

        ollama_url = get_ollama_url()
        # Try to get models from Ollama API
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            if models:
                output = "📋 **Available Models:**\n"
                output += "=" * 40 + "\n"
                for model in models:
                    size_gb = model.get("size", 0) / (1024**3)
                    modified = model.get("modified_at", "Unknown")[:10]  # Just date
                    output += f"• **{model['name']}** ({size_gb:.1f}GB) - Modified: {modified}\n"

                output += f"\n💡 **Total:** {len(models)} models installed\n"
                output += "🚀 **Performance:** Models with GPU acceleration will run 5-10x faster"
                return output
            else:
                return "📋 **No models installed**\n\n💡 Use 'Pull Model' to download a model first\n\n🔥 **Recommended starter models:**\n• phi3:mini (3.8GB) - Fast and efficient\n• llama2:7b (3.8GB) - Well-researched"
        else:
            return "❌ Could not connect to Ollama API. Make sure services are started."
    except Exception as e:
        return f"❌ Error listing models: {str(e)}"


def pull_model(model_name: str) -> str:
    """Pull/download a model"""
    if not model_name.strip():
        return "❌ Please enter a model name"

    try:
        model_name = model_name.strip()

        # Show progress message first
        progress_msg = f"📥 **Downloading model:** {model_name}\n\n"
        progress_msg += "⏳ This may take several minutes depending on model size...\n"
        progress_msg += "💡 Larger models provide better quality but take longer to download and run.\n\n"

        success = manager.pull_model(model_name)
        if success:
            return f"✅ **Model '{model_name}' downloaded successfully!**\n\n🎯 **Next steps:**\n  • Generate a test corpus\n  • Run bias audit with this model\n  • Analyze the results"
        else:
            return f"❌ Failed to download model '{model_name}'.\n\n💡 **Tips:**\n  • Check the model name spelling\n  • Ensure you have enough disk space\n  • Try a smaller model first (e.g., phi3:mini)"
    except Exception as e:
        return f"❌ Error pulling model: {str(e)}"


def generate_corpus(config_file: str = "") -> str:
    """Generate test corpus"""
    try:
        if config_file and config_file.strip():
            # Use specific config file
            success = manager.generate_corpus(config_file.strip())
            if success:
                return f"✅ **Corpus generated successfully** with config '{config_file}'\n\n📁 Check corpus directory for generated files."
            else:
                return f"❌ Failed to generate corpus with config '{config_file}'"
        else:
            # Use interactive mode via subprocess
            result = subprocess.run(
                ["python", "src/Phase1_CorpusGenerator/generate_corpus.py"],
                capture_output=True,
                text=True,
                cwd=str(manager.project_root),
            )

            if result.returncode == 0:
                return "✅ **Corpus generated successfully!**\n\n📁 Files saved to: src/Phase1_CorpusGenerator/corpus/\n\n🎯 **Next step:** Run bias audit with your generated corpus"
            else:
                return f"❌ Corpus generation failed:\n{result.stderr[:500]}..."
    except Exception as e:
        return f"❌ Error generating corpus: {str(e)}"


def run_audit(model_name: str, corpus_file: str, output_dir: str = "results") -> str:
    """Run bias audit"""
    if not model_name.strip():
        return "❌ Please enter a model name"
    if not corpus_file.strip():
        return "❌ Please select a corpus file"

    try:
        # Check if corpus file exists
        corpus_path = Path(corpus_file.strip())
        if not corpus_path.exists():
            return f"❌ Corpus file not found: {corpus_file}\n\n💡 Generate a corpus first in the Corpus Generation tab"

        model_name = model_name.strip()
        output_dir = output_dir.strip() if output_dir.strip() else "results"

        # Show initial progress
        progress_msg = "🔍 **Starting bias audit...**\n\n"
        progress_msg += f"🤖 **Model:** {model_name}\n"
        progress_msg += f"📋 **Corpus:** {corpus_path.name}\n"
        progress_msg += f"📁 **Output:** {output_dir}\n\n"
        progress_msg += "⏳ **This process may take 15-60 minutes depending on:**\n"
        progress_msg += "  • Corpus size (number of test prompts)\n"
        progress_msg += "  • Model size and complexity\n"
        progress_msg += "  • Available hardware (GPU vs CPU)\n\n"
        progress_msg += "🔄 **Processing...** Please wait for completion."

        # Run audit using subprocess
        cmd = [
            "python",
            "src/Phase2_ModelAuditor/audit_model.py",
            "--model",
            model_name,
            "--corpus",
            str(corpus_path),
            "--output-dir",
            output_dir,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(manager.project_root)
        )

        if result.returncode == 0:
            return f"✅ **Audit completed successfully!**\n\n📊 **Results saved to:** {output_dir}\n\n🎯 **Next steps:**\n  • Go to Results Analysis tab\n  • Select your results file\n  • Generate bias analysis report"
        else:
            error_msg = result.stderr[:800] if result.stderr else "Unknown error"
            return f"❌ **Audit failed:**\n\n```\n{error_msg}\n```\n\n💡 **Troubleshooting:**\n  • Ensure the model is running in Ollama\n  • Check corpus file format\n  • Verify sufficient disk space"

    except Exception as e:
        return f"❌ Error running audit: {str(e)}"


def analyze_results(results_file: str) -> str:
    """Analyze bias audit results"""
    if not results_file.strip():
        return "❌ Please select a results file"

    try:
        results_path = Path(results_file.strip())
        if not results_path.exists():
            return f"❌ Results file not found: {results_file}\n\n💡 Run a bias audit first to generate results"

        # Show analysis progress
        progress_msg = f"📊 **Analyzing results:** {results_path.name}\n\n"
        progress_msg += (
            "🔄 **Processing bias patterns and generating visualizations...**\n"
        )
        progress_msg += "⏳ This may take a few minutes for large datasets.\n\n"

        # Run analysis
        result = subprocess.run(
            ["python", "src/Phase3_Analysis/analyze_results.py", str(results_path)],
            capture_output=True,
            text=True,
            cwd=str(manager.project_root),
        )

        if result.returncode == 0:
            return "✅ **Analysis completed successfully!**\n\n📈 **Generated outputs:**\n  • bias_report.png - Visual bias analysis\n  • Statistical summary in console\n  • Detailed bias metrics\n\n💡 **Check the results directory for all generated files**"
        else:
            error_msg = result.stderr[:800] if result.stderr else "Unknown error"
            return f"❌ **Analysis failed:**\n\n```\n{error_msg}\n```\n\n💡 **Troubleshooting:**\n  • Ensure results file is valid CSV format\n  • Check for required analysis dependencies\n  • Verify file permissions"

    except Exception as e:
        return f"❌ Error analyzing results: {str(e)}"


def get_corpus_files() -> list[str]:
    """Get list of available corpus files"""
    corpus_dir = Path("src/Phase1_CorpusGenerator/corpus")
    if corpus_dir.exists():
        files = [str(f) for f in corpus_dir.glob("*.csv")]
        return sorted(files, key=lambda x: Path(x).stat().st_mtime, reverse=True)
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


def create_interface():
    """Create the main Gradio interface"""

    # Custom CSS for professional styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .tab-nav {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        padding: 4px;
    }
    .tab-nav button {
        border-radius: 8px !important;
        margin: 2px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .tab-nav button[aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
    }
    .gr-button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .gr-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    .gr-button-secondary {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        border: none !important;
        color: white !important;
    }
    .gr-textbox, .gr-dropdown {
        border-radius: 8px !important;
        border: 2px solid rgba(0, 0, 0, 0.1) !important;
    }
    .gr-textbox:focus, .gr-dropdown:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    """

    with gr.Blocks(
        css=custom_css,
        title="EquiLens - AI Bias Detection Platform",
        theme="soft",
    ) as demo:
        # Header
        gr.Markdown("""
        <div class="hero-section">
            <h1 style="font-size: 3rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
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
            # Dashboard Tab
            with gr.Tab("🏠 Dashboard"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("## 🔍 System Status & Management")

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
                        **Step 1:** Check system status and ensure Docker is running

                        **Step 2:** Start EquiLens services (Ollama + GPU support)

                        **Step 3:** Pull a language model (try phi3:mini for quick start)

                        **Step 4:** Generate test corpus for bias evaluation

                        **Step 5:** Run comprehensive bias audit and analyze results
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

            # Models Tab
            with gr.Tab("🎯 Models"):
                gr.Markdown("## 🎯 Model Management & Downloads")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            list_models_btn = gr.Button(
                                "📋 List Available Models", variant="primary", size="lg"
                            )
                            refresh_models_btn = gr.Button(
                                "🔄 Refresh", variant="secondary"
                            )

                        models_output = gr.Textbox(
                            label="📦 Model Information",
                            lines=12,
                            max_lines=15,
                            interactive=False,
                            placeholder="Click 'List Available Models' to see installed models...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⬇️ Download New Model")
                        pull_model_input = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., phi3:mini, llama2:latest, mistral:7b",
                            info="Enter the exact model name from Ollama library",
                        )
                        pull_model_btn = gr.Button(
                            "⬇️ Pull Model", variant="primary", size="lg"
                        )

                        gr.Markdown("### 💡 Recommended Models")
                        gr.Markdown("""
                        **Quick Start:**
                        - `phi3:mini` (3.8GB) - Fast, efficient
                        - `qwen2:0.5b` (394MB) - Ultra lightweight

                        **Research Grade:**
                        - `llama2:7b` (3.8GB) - Well-studied
                        - `mistral:7b` (4.1GB) - High quality

                        **Advanced:**
                        - `llama2:13b` (7.3GB) - Better accuracy
                        - `codellama:7b` (3.8GB) - Code-focused
                        """)

                # Connect model buttons
                list_models_btn.click(list_models, outputs=models_output)
                refresh_models_btn.click(list_models, outputs=models_output)
                pull_model_btn.click(
                    pull_model, inputs=pull_model_input, outputs=models_output
                )

            # Corpus Generation Tab
            with gr.Tab("📝 Corpus Generation"):
                gr.Markdown("## 📝 Test Corpus Generation")

                with gr.Row():
                    with gr.Column(scale=2):
                        corpus_config_input = gr.Textbox(
                            label="📋 Config File (Optional)",
                            placeholder="Leave empty for interactive mode",
                            info="Path to configuration file, or leave empty for guided setup",
                        )

                        with gr.Row():
                            generate_corpus_btn = gr.Button(
                                "📝 Generate New Corpus", variant="primary", size="lg"
                            )
                            refresh_corpus_btn = gr.Button(
                                "🔄 Refresh Files", variant="secondary"
                            )

                        corpus_output = gr.Textbox(
                            label="📊 Generation Output",
                            lines=10,
                            interactive=False,
                            placeholder="Click 'Generate New Corpus' to start the process...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📚 Available Corpus Files")
                        available_corpus = gr.Dropdown(
                            label="Generated Corpus Files",
                            choices=get_corpus_files(),
                            info="Previously generated corpus files",
                        )

                        gr.Markdown("### 🎯 Bias Categories")
                        gr.Markdown("""
                        **Built-in Categories:**
                        - **Gender Bias** - Names, pronouns, roles
                        - **Racial/Ethnic** - Demographics, cultural
                        - **Professional** - Career assumptions
                        - **Age Bias** - Generational stereotypes
                        - **Religious** - Faith-based biases
                        - **Socioeconomic** - Class assumptions
                        """)

                # Connect corpus buttons
                generate_corpus_btn.click(
                    generate_corpus, inputs=corpus_config_input, outputs=corpus_output
                )
                refresh_corpus_btn.click(
                    lambda: gr.Dropdown(choices=get_corpus_files()),
                    outputs=available_corpus,
                )

            # Audit Tab
            with gr.Tab("🔍 Bias Audit"):
                gr.Markdown("## 🔍 Bias Audit Execution")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            audit_model_input = gr.Textbox(
                                label="🤖 Model Name",
                                placeholder="e.g., llama2:latest, phi3:mini",
                                info="Name of the model to audit (must be pulled first)",
                            )
                            audit_output_dir = gr.Textbox(
                                label="📁 Output Directory",
                                value="results",
                                info="Directory to save audit results",
                            )

                        audit_corpus_input = gr.Dropdown(
                            label="📋 Corpus File",
                            choices=get_corpus_files(),
                            info="Select a generated corpus file",
                        )

                        with gr.Row():
                            run_audit_btn = gr.Button(
                                "🔍 Run Bias Audit", variant="primary", size="lg"
                            )
                            refresh_audit_corpus_btn = gr.Button(
                                "🔄 Refresh Corpus", variant="secondary"
                            )

                        audit_output = gr.Textbox(
                            label="⚡ Audit Progress & Results",
                            lines=12,
                            interactive=False,
                            placeholder="Configure your audit parameters and click 'Run Bias Audit'...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚡ Audit Process")
                        gr.Markdown("""
                        **Phase 1:** Model Loading
                        - Initializes the selected model
                        - Configures generation parameters

                        **Phase 2:** Corpus Processing
                        - Reads test sentences
                        - Prepares evaluation prompts

                        **Phase 3:** Response Generation
                        - Sends prompts to model
                        - Collects and stores responses
                        """)

                        gr.Markdown("### 🕐 Time Estimates")
                        gr.Markdown("""
                        **Small corpus (100 prompts):**
                        - CPU: ~15-30 minutes
                        - GPU: ~5-10 minutes

                        **Medium corpus (500 prompts):**
                        - CPU: ~1-2 hours
                        - GPU: ~15-30 minutes
                        """)

                # Connect audit buttons
                run_audit_btn.click(
                    run_audit,
                    inputs=[audit_model_input, audit_corpus_input, audit_output_dir],
                    outputs=audit_output,
                )
                refresh_audit_corpus_btn.click(
                    lambda: gr.Dropdown(choices=get_corpus_files()),
                    outputs=audit_corpus_input,
                )

            # Analysis Tab
            with gr.Tab("📊 Results Analysis"):
                gr.Markdown("## 📊 Bias Analysis & Visualization")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            results_file_input = gr.Dropdown(
                                label="📈 Results File",
                                choices=get_results_files(),
                                info="Select audit results to analyze",
                            )
                            refresh_results_btn = gr.Button(
                                "🔄 Refresh Results", variant="secondary"
                            )

                        analyze_results_btn = gr.Button(
                            "📊 Analyze Results", variant="primary", size="lg"
                        )

                        results_output = gr.Textbox(
                            label="📈 Analysis Results",
                            lines=12,
                            interactive=False,
                            placeholder="Select a results file and click 'Analyze Results' to begin...",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📈 Analysis Features")
                        gr.Markdown("""
                        **Statistical Analysis:**
                        - Bias magnitude calculations
                        - Statistical significance testing
                        - Confidence intervals

                        **Visualizations:**
                        - Bias heatmaps by category
                        - Distribution comparisons
                        - Trend analysis charts

                        **Reporting:**
                        - Comprehensive PDF reports
                        - Detailed findings breakdown
                        """)

                # Connect analysis buttons
                analyze_results_btn.click(
                    analyze_results, inputs=results_file_input, outputs=results_output
                )
                refresh_results_btn.click(
                    lambda: gr.Dropdown(choices=get_results_files()),
                    outputs=results_file_input,
                )

        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; margin-top: 2rem;">
            <h3 style="margin: 0; color: #374151;">🔍 EquiLens v1.0 - AI Bias Detection Platform</h3>
            <p style="margin: 0.5rem 0; color: #6b7280;">Built with ❤️ for responsible AI development</p>
            <div style="margin-top: 1rem;">
                <a href="https://github.com/Life-Experimentalist/EquiLens" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">📚 Documentation</a>
                <a href="https://github.com/Life-Experimentalist/EquiLens/issues" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">🐛 Report Issues</a>
                <a href="https://github.com/Life-Experimentalist/EquiLens" target="_blank" style="margin: 0 1rem; color: #4f46e5; text-decoration: none;">⭐ GitHub</a>
            </div>
        </div>
        """)

    return demo


def main():
    """Main function to launch the Gradio interface"""
    print("🚀 Starting EquiLens Web Interface...")
    print("🌐 This will open in your browser automatically...")

    demo = create_interface()

    # Launch with optimal configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Standard Gradio port
        share=False,  # Set to True for public links
        show_api=False,  # Hide API documentation
        show_error=True,  # Show detailed errors
        quiet=False,  # Show startup messages
        inbrowser=True,  # Auto-open browser
        favicon_path=None,  # Could add custom favicon
        app_kwargs={"docs_url": None, "redoc_url": None},  # Disable API docs
    )


if __name__ == "__main__":
    main()
