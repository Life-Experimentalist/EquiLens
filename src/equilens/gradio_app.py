"""
EquiLens Gradio Frontend

Pure frontend interface that communicates with the FastAPI backend.
Auto-detects Docker vs local environment.
"""

import gradio as gr
import requests

from equilens.telemetry import stats_html


class EquiLensClient:
    """Client for communicating with EquiLens backend API."""

    def __init__(self):
        self.backend_url = self._detect_backend_url()
        self.check_backend_health()

    def _detect_backend_url(self) -> str:
        """Auto-detect backend URL based on environment."""
        from equilens.core.ports import get_backend_url

        backend_url = get_backend_url()
        print(f"🔗 Backend URL: {backend_url}")
        return backend_url

    def check_backend_health(self) -> bool:
        """Check if backend is healthy."""
        try:
            response = requests.get(f"{self.backend_url}/api/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_system_status(self) -> dict:
        """Get system status from backend."""
        try:
            response = requests.get(f"{self.backend_url}/api/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

    def create_job(self, job_type: str, config: dict) -> dict:
        """Create a new job."""
        try:
            response = requests.post(
                f"{self.backend_url}/api/jobs",
                json={"job_type": job_type, "config": config},
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to create job: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def get_job(self, job_id: str) -> dict:
        """Get job status."""
        try:
            response = requests.get(f"{self.backend_url}/api/jobs/{job_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error in get_job: {e}")
        return {}

    def list_jobs(self, status: str | None = None) -> list[dict]:
        """List all jobs."""
        try:
            params = {"status": status} if status else {}
            response = requests.get(
                f"{self.backend_url}/api/jobs", params=params, timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error in list_jobs: {e}")
        return []

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        try:
            response = requests.post(
                f"{self.backend_url}/api/jobs/{job_id}/cancel", timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error in cancel_job: {e}")
            return False
            return False

    def get_job_logs(self, job_id: str) -> list[dict]:
        try:
            response = requests.get(
                f"{self.backend_url}/api/jobs/{job_id}/logs", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("logs", [])
        except Exception as e:
            print(f"Error in get_job_logs: {e}")
        return []

    def list_results(self) -> list[dict]:
        try:
            response = requests.get(f"{self.backend_url}/api/results", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
        except Exception as e:
            print(f"Error in list_results: {e}")
        return []

    def list_models(self) -> list[dict]:
        try:
            response = requests.get(f"{self.backend_url}/api/models", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            print(f"Error in list_models: {e}")
        return []

    def pull_model(self, model_name: str) -> dict:
        """Pull an Ollama model."""
        try:
            response = requests.post(
                f"{self.backend_url}/api/models/pull",
                params={"model_name": model_name},
                timeout=10,
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}


# Initialize client
client = EquiLensClient()


# ===== UI Functions =====


def show_system_status():
    """Display system status."""
    status = client.get_system_status()

    if "error" in status:
        return f"❌ **Backend Error**: {status['error']}"

    output = "# 🔍 EquiLens System Status\n\n"
    output += f"**Backend**: {'✅ Running' if status.get('backend_status') == 'running' else '❌ Offline'}\n\n"
    output += f"**Environment**: {'🐳 Docker' if status.get('docker_detected') else '💻 Local'}\n\n"
    output += f"**Ollama**: {'✅ Available' if status.get('ollama_available') else '❌ Not Available'}\n\n"
    output += f"**Ollama URL**: `{status.get('ollama_url', 'Unknown')}`\n\n"

    return output


def show_models_list():
    """Display available Ollama models."""
    models = client.list_models()

    if not models:
        return "📋 No models installed\n\n💡 Use the 'Pull Model' section to download models"

    output = "# 📦 Available Ollama Models\n\n"
    for model in models:
        size_gb = model.get("size", 0) / (1024**3)
        modified = model.get("modified", "")[:10]
        output += f"- **{model['name']}** ({size_gb:.1f} GB) - Modified: {modified}\n"

    return output


def pull_model_action(model_name: str):
    """Pull/download a model."""
    if not model_name:
        return "❌ Please enter a model name"

    result = client.pull_model(model_name)

    if "error" in result:
        return f"❌ **Error**: {result['error']}"

    return f"✅ **Pulling model**: {model_name}\n\n📋 **Job ID**: {result.get('job_id')}\n\n{result.get('message', '')}"


def submit_corpus_generation(config_file: str):
    """Submit corpus generation job."""
    config = {}
    if config_file:
        config["config_file"] = config_file

    result = client.create_job("corpus_generation", config)

    if "error" in result:
        return f"❌ **Error**: {result['error']}"

    return f"✅ **Job Created Successfully!**\n\n📋 **Job ID**: `{result['job_id']}`\n**Status**: {result['status']}\n**Created**: {result['created_at']}"


def submit_audit_job(model_name: str, corpus_file: str, output_dir: str, silent: bool):
    """Submit audit job."""
    if not model_name:
        return "❌ Please select a model"

    config = {
        "model": model_name,
        "output_dir": output_dir or "results",
        "silent": silent,
    }

    if corpus_file:
        config["corpus"] = corpus_file

    result = client.create_job("audit", config)

    if "error" in result:
        return f"❌ **Error**: {result['error']}"

    return f"✅ **Audit Job Created!**\n\n📋 **Job ID**: `{result['job_id']}`\n**Model**: {model_name}\n**Status**: {result['status']}\n**Created**: {result['created_at']}"


def submit_analysis_job(results_file: str, no_ai: bool, advanced: bool):
    """Submit analysis job."""
    if not results_file:
        return "❌ Please enter results file path"

    config = {
        "results_file": results_file,
        "no_ai": no_ai,
        "advanced": advanced,
    }

    result = client.create_job("analysis", config)

    if "error" in result:
        return f"❌ **Error**: {result['error']}"

    return f"✅ **Analysis Job Created!**\n\n📋 **Job ID**: `{result['job_id']}`\n**Status**: {result['status']}\n**Created**: {result['created_at']}"


def get_job_status(job_id: str):
    """Get status of a specific job."""
    if not job_id:
        return "❌ Please enter a job ID"

    job = client.get_job(job_id)

    if not job:
        return f"❌ Job not found: {job_id}"

    output = f"# 📋 Job Status: {job_id}\n\n"
    output += f"**Type**: {job['job_type']}\n"
    output += f"**Status**: {job['status'].upper()}\n"
    output += f"**Progress**: {job['progress']}/{job['total']} ({(job['progress'] / job['total'] * 100):.1f}%)\n"
    output += f"**Created**: {job['created_at']}\n"

    if job.get("started_at"):
        output += f"**Started**: {job['started_at']}\n"
    if job.get("completed_at"):
        output += f"**Completed**: {job['completed_at']}\n"
    if job.get("result_path"):
        output += f"**Result**: `{job['result_path']}`\n"
    if job.get("error_message"):
        output += f"\n❌ **Error**: {job['error_message']}\n"

    return output


def show_job_logs(job_id: str):
    """Display job logs."""
    if not job_id:
        return "❌ Please enter a job ID"

    logs = client.get_job_logs(job_id)

    if not logs:
        return f"📝 No logs available for job: {job_id}"

    output = f"# 📝 Logs for Job: {job_id}\n\n"
    output += "```\n"

    # Show logs in reverse chronological order
    for log in reversed(logs[-100:]):  # Last 100 logs
        timestamp = log["timestamp"][:19]
        level = log["level"].upper()
        message = log["message"]
        output += f"[{timestamp}] {level}: {message}\n"

    output += "```\n"
    return output


def list_all_jobs(status_filter: str):
    """List all jobs with optional status filter."""
    filter_value = status_filter if status_filter != "All" else None
    jobs = client.list_jobs(status=filter_value)

    if not jobs:
        return "📋 No jobs found"

    output = f"# 📋 Jobs ({len(jobs)} total)\n\n"

    for job in jobs:
        status_emoji = {
            "queued": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "cancelled": "🚫",
        }.get(job["status"], "❓")

        progress = (
            f"{job['progress']}/{job['total']}" if job.get("total", 0) > 0 else "N/A"
        )

        output += f"{status_emoji} **{job['job_id']}** - {job['job_type']}\n"
        output += f"   Status: {job['status']} | Progress: {progress} | Created: {job['created_at'][:19]}\n\n"

    return output


def cancel_job_action(job_id: str):
    """Cancel a job."""
    if not job_id:
        return "❌ Please enter a job ID"

    success = client.cancel_job(job_id)

    if success:
        return f"✅ Job {job_id} has been cancelled"
    else:
        return f"❌ Failed to cancel job {job_id}"


def list_results_action():
    """List available results."""
    results = client.list_results()

    if not results:
        return "📁 No results found"

    output = "# 📊 Available Results\n\n"

    for result in results:
        output += f"📂 **{result['name']}**\n"
        output += (
            f"   CSV: `{result['csv_file']}` | Created: {result['created'][:19]}\n"
        )
        output += f"   Path: `{result['path']}`\n\n"

    return output


# ===== Create Gradio Interface =====


def create_interface():
    """Create the main Gradio interface."""

    with gr.Blocks(
        title="EquiLens - AI Bias Detection Platform", theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown(
            """
            # 🔍 EquiLens — AI Bias Detection Platform

            **Black-box prompt-engineering framework for detecting, measuring, and reporting bias in SLMs & LLMs — running fully locally via Ollama.**
            """
        )
        gr.HTML(stats_html())

        with gr.Tabs():
            # ===== System Status Tab =====
            with gr.Tab("🏠 System Status"):
                gr.Markdown("### System Status & Health Check")

                status_output = gr.Markdown()
                status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                status_btn.click(show_system_status, outputs=status_output)

                # Auto-load on startup
                interface.load(show_system_status, outputs=status_output)

            # ===== Models Tab =====
            with gr.Tab("📦 Models"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Available Models")
                        models_output = gr.Markdown()
                        refresh_models_btn = gr.Button("🔄 Refresh Models")
                        refresh_models_btn.click(
                            show_models_list, outputs=models_output
                        )

                    with gr.Column():
                        gr.Markdown("### Pull New Model")
                        model_name_input = gr.Textbox(
                            label="Model Name",
                            placeholder="e.g., llama3.2, phi3:mini, mistral",
                        )
                        pull_btn = gr.Button("⬇️ Pull Model", variant="primary")
                        pull_output = gr.Markdown()
                        pull_btn.click(
                            pull_model_action,
                            inputs=model_name_input,
                            outputs=pull_output,
                        )

            # ===== Job Submission Tab =====
            with gr.Tab("🚀 Submit Jobs"):
                gr.Markdown("### Submit New Jobs")

                with gr.Accordion("📝 Generate Corpus", open=False):
                    corpus_config_file = gr.Textbox(
                        label="Config File (optional)",
                        placeholder="Path to corpus generation config JSON",
                    )
                    submit_corpus_btn = gr.Button("Generate Corpus", variant="primary")
                    corpus_output = gr.Markdown()
                    submit_corpus_btn.click(
                        submit_corpus_generation,
                        inputs=corpus_config_file,
                        outputs=corpus_output,
                    )

                with gr.Accordion("🔍 Run Audit", open=True):
                    audit_model = gr.Textbox(
                        label="Model Name", placeholder="e.g., llama3.2:latest"
                    )
                    audit_corpus = gr.Textbox(
                        label="Corpus File (optional)", placeholder="Path to corpus CSV"
                    )
                    audit_output_dir = gr.Textbox(
                        label="Output Directory (optional)", placeholder="results"
                    )
                    audit_silent = gr.Checkbox(label="Silent Mode", value=False)
                    submit_audit_btn = gr.Button("Run Audit", variant="primary")
                    audit_output = gr.Markdown()
                    submit_audit_btn.click(
                        submit_audit_job,
                        inputs=[
                            audit_model,
                            audit_corpus,
                            audit_output_dir,
                            audit_silent,
                        ],
                        outputs=audit_output,
                    )

                with gr.Accordion("📊 Run Analysis", open=False):
                    analysis_results = gr.Textbox(
                        label="Results File", placeholder="Path to audit results CSV"
                    )
                    analysis_no_ai = gr.Checkbox(label="Skip AI Insights", value=False)
                    analysis_advanced = gr.Checkbox(
                        label="Advanced Analytics", value=True
                    )
                    submit_analysis_btn = gr.Button("Run Analysis", variant="primary")
                    analysis_output = gr.Markdown()
                    submit_analysis_btn.click(
                        submit_analysis_job,
                        inputs=[analysis_results, analysis_no_ai, analysis_advanced],
                        outputs=analysis_output,
                    )

            # ===== Job Monitoring Tab =====
            with gr.Tab("📊 Monitor Jobs"):
                gr.Markdown("### Job Status & Monitoring")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Job Status")
                        job_id_input = gr.Textbox(
                            label="Job ID", placeholder="Enter job ID"
                        )
                        check_status_btn = gr.Button("🔍 Check Status")
                        status_display = gr.Markdown()
                        check_status_btn.click(
                            get_job_status, inputs=job_id_input, outputs=status_display
                        )

                    with gr.Column():
                        gr.Markdown("#### Job Logs")
                        logs_btn = gr.Button("📝 View Logs")
                        logs_display = gr.Markdown()
                        logs_btn.click(
                            show_job_logs, inputs=job_id_input, outputs=logs_display
                        )

                gr.Markdown("---")
                gr.Markdown("#### All Jobs")

                status_filter = gr.Radio(
                    choices=["All", "queued", "running", "completed", "failed"],
                    value="All",
                    label="Filter by Status",
                )
                list_jobs_btn = gr.Button("📋 List Jobs")
                jobs_display = gr.Markdown()
                list_jobs_btn.click(
                    list_all_jobs, inputs=status_filter, outputs=jobs_display
                )

                gr.Markdown("---")
                gr.Markdown("#### Cancel Job")

                cancel_job_id = gr.Textbox(label="Job ID to Cancel")
                cancel_btn = gr.Button("🚫 Cancel Job", variant="stop")
                cancel_output = gr.Markdown()
                cancel_btn.click(
                    cancel_job_action, inputs=cancel_job_id, outputs=cancel_output
                )

            # ===== Results Tab =====
            with gr.Tab("📈 Results"):
                gr.Markdown("### View & Export Results")

                list_results_btn = gr.Button("📂 List Available Results")
                results_display = gr.Markdown()
                list_results_btn.click(list_results_action, outputs=results_display)

                gr.Markdown(
                    """
                    #### Export Instructions

                    To download results as a .zip file:

                    1. Note the result directory name from the list above
                    2. Visit: `http://localhost:8000/api/results/{result_name}/export`
                    3. Replace `{result_name}` with your actual result directory name

                    The export includes:
                    - All visualization PNG files
                    - HTML report with embedded images
                    - Markdown report
                    - Raw CSV data
                    - README file
                    """
                )

        gr.Markdown(
            """
            ---

            **EquiLens v2.0** · AI Bias Detection Platform · B.Tech Final Year Project, Amrita Vishwa Vidyapeetham
            [GitHub](https://github.com/Life-Experimentalist/EquiLens) · [Docs](https://github.com/Life-Experimentalist/EquiLens/tree/main/docs) · [DOI](https://doi.org/10.5281/zenodo.17014103) · [Website](https://equilens.vkrishna04.me)
            """
        )

    return interface


def main():
    """Launch the Gradio interface."""
    from equilens.core.ports import get_frontend_port

    # Get available port
    port = get_frontend_port()

    print("\n" + "=" * 70)
    print("🔍 EquiLens Gradio Frontend")
    print("=" * 70)
    print()

    # Check backend connectivity
    if not client.check_backend_health():
        print("⚠️  WARNING: Cannot connect to backend!")
        print(f"   Backend URL: {client.backend_url}")
        print("   Make sure the backend service is running:")
        print("   uv run equilens backend")
        print()
    else:
        print(f"✅ Connected to backend: {client.backend_url}")
        print()

    print(f"🌐 Starting web interface on port {port}...")
    print(f"   URL: http://localhost:{port}")
    print()
    print("💡 Tip: Set custom port with environment variable:")
    print(f"   $env:FRONTEND_PORT = {port + 1}")
    print("=" * 70)
    print()

    interface = create_interface()

    # Launch
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
