"""
Interactive Terminal User Interface for EquiLens using Textual

A beautiful, interactive terminal UI for managing EquiLens operations.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Log,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
)

from equilens.core.manager import EquiLensManager


class EquiLensTUI(App):
    """EquiLens Terminal User Interface"""

    TITLE = "EquiLens - AI Bias Detection Platform"
    SUB_TITLE = "Interactive Terminal Interface"

    CSS = """
    #welcome {
        height: auto;
        margin: 1;
        padding: 1;
        border: solid $primary;
        border-title-align: center;
    }

    .status-box {
        height: auto;
        margin: 1;
        padding: 1;
        border: solid $secondary;
    }

    .action-buttons {
        height: auto;
        margin: 1;
        padding: 1;
    }

    #log {
        height: 50%;
        border: solid $warning;
        border-title-align: center;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("s", "status", "Status"),
        Binding("r", "refresh", "Refresh"),
        Binding("h", "help", "Help"),
    ]

    def __init__(self):
        super().__init__()
        self.manager = EquiLensManager()
        self.status_data = {}

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield from self.compose_dashboard()

            with TabPane("Models", id="models"):
                yield from self.compose_models()

            with TabPane("Audit", id="audit"):
                yield from self.compose_audit()

            with TabPane("Results", id="results"):
                yield from self.compose_results()

        yield Footer()

    def compose_dashboard(self) -> ComposeResult:
        """Compose the dashboard tab"""
        welcome_md = """
# 🔍 EquiLens Dashboard

Welcome to the **EquiLens AI Bias Detection Platform**!

## Quick Actions
- Check system status and GPU support
- Start/stop services
- View service logs

## Getting Started
1. Ensure Docker is running
2. Start EquiLens services
3. Pull required models
4. Run bias audits
        """

        yield Vertical(
            Markdown(welcome_md, id="welcome"),
            Horizontal(
                Static(
                    "🎮 	GPU Status: Loading...",
                    classes="status-box",
                    id="gpu-status",
                ),
                Static(
                    "🐳 	Docker Status: Loading...",
                    classes="status-box",
                    id="docker-status",
                ),
                Static(
                    "🤖 	Ollama Status: Loading...",
                    classes="status-box",
                    id="ollama-status",
                ),
            ),
            Horizontal(
                Button("🚀  Start Services", id="start-btn", variant="success"),
                Button("🛑  Stop Services", id="stop-btn", variant="error"),
                Button("📊  Refresh Status", id="refresh-btn", variant="primary"),
                Button("🎮  GPU Check", id="gpu-btn", variant="default"),
                classes="action-buttons",
            ),
            Log(id="log", auto_scroll=True),
        )

    def compose_models(self) -> ComposeResult:
        """Compose the models tab"""
        yield Vertical(
            Static("🎯 	**Model Management**", classes="status-box"),
            Horizontal(
                Button("📋  List Models", id="list-models-btn", variant="primary"),
                Button("📥  Pull llama2", id="pull-llama2-btn", variant="success"),
                Button("📥  Pull phi3", id="pull-phi3-btn", variant="success"),
                classes="action-buttons",
            ),
            DataTable(id="models-table"),
            Log(id="models-log", auto_scroll=True),
        )

    def compose_audit(self) -> ComposeResult:
        """Compose the audit tab"""
        yield Vertical(
            Static("🔍  **Bias Audit Operations**", classes="status-box"),
            Horizontal(
                Button("📝  Generate Corpus", id="generate-btn", variant="primary"),
                Button("🔍  Run Audit", id="audit-btn", variant="success"),
                Button("📁  Open Results", id="results-btn", variant="default"),
                classes="action-buttons",
            ),
            Log(id="audit-log", auto_scroll=True),
        )

    def compose_results(self) -> ComposeResult:
        """Compose the results tab"""
        yield Vertical(
            Static("📊  **Analysis Results**", classes="status-box"),
            Horizontal(
                Button("📊  Analyze Latest", id="analyze-btn", variant="primary"),
                Button("📁  Browse Results", id="browse-btn", variant="default"),
                classes="action-buttons",
            ),
            DataTable(id="results-table"),
            Log(id="results-log", auto_scroll=True),
        )

    def on_mount(self) -> None:
        """Called when app starts."""
        self.log_message("🚀 EquiLens TUI started")
        self.action_refresh()

    def log_message(self, message: str, log_id: str = "log") -> None:
        """Add a message to the specified log widget"""
        try:
            log_widget = self.query_one(f"#{log_id}", Log)
            log_widget.write_line(message)
        except Exception:
            # Fallback to main log if specific log not found
            try:
                main_log = self.query_one("#log", Log)
                main_log.write_line(message)
            except Exception as e:
                self.log_message(f"❌ Error logging message: {e}")

                pass

    def action_status(self) -> None:
        """Show current status"""
        self.action_refresh()

    def action_refresh(self) -> None:
        """Refresh system status"""
        self.log_message("🔄 Refreshing status...")

        try:
            # Get status data
            self.status_data = self.manager.check_system_status()

            # Update GPU status
            gpu_info = self.status_data["gpu"]
            gpu_status = "🟢 Ready" if gpu_info["gpu_available"] else "🟡 CPU Mode"
            try:
                gpu_widget = self.query_one("#gpu-status", Static)
                gpu_widget.update(f"🎮 GPU Status: {gpu_status}")
            except Exception:
                pass

            # Update Docker status
            docker_info = self.status_data["docker"]
            docker_status = (
                "🟢 Ready" if docker_info["docker_available"] else "🔴 Not Available"
            )
            try:
                docker_widget = self.query_one("#docker-status", Static)
                docker_widget.update(f"🐳 Docker Status: {docker_status}")
            except Exception:
                pass

            # Update Ollama status
            ollama_status = (
                "🟢 Ready" if docker_info["ollama_accessible"] else "🔴 Not Accessible"
            )
            try:
                ollama_widget = self.query_one("#ollama-status", Static)
                ollama_widget.update(f"🤖 Ollama Status: {ollama_status}")
            except Exception:
                pass

            self.log_message("✅ Status refreshed")

        except Exception as e:
            self.log_message(f"❌ Error refreshing status: {e}")

    def action_help(self) -> None:
        """Show help information"""
        help_text = """
🔍 **EquiLens TUI Help**

**Keyboard Shortcuts:**
- `q` - Quit application
- `s` - Refresh status
- `r` - Refresh current view
- `h` - Show this help

**Navigation:**
- Use Tab/Shift+Tab to navigate between tabs
- Click buttons or use Enter to activate
- Scroll in logs with arrow keys

**Tabs:**
- **Dashboard** - System status and service control
- **Models** - Model management and downloads
- **Audit** - Run bias detection audits
- **Results** - View and analyze results
        """
        self.log_message(help_text)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id

        if button_id == "start-btn":
            self.log_message("🚀 Starting services...")
            try:
                success = self.manager.start_services()
                if success:
                    self.log_message("✅ Services started successfully!")
                else:
                    self.log_message("❌ Failed to start services")
            except Exception as e:
                self.log_message(f"❌ Error starting services: {e}")

        elif button_id == "stop-btn":
            self.log_message("🛑 Stopping services...")
            try:
                success = self.manager.stop_services()
                if success:
                    self.log_message("✅ Services stopped successfully!")
                else:
                    self.log_message("❌ Failed to stop services")
            except Exception as e:
                self.log_message(f"❌ Error stopping services: {e}")

        elif button_id == "refresh-btn":
            self.action_refresh()

        elif button_id == "gpu-btn":
            self.log_message("🎮 Checking GPU status...")
            self.manager.gpu_manager.check_gpu_support()
            self.log_message("✅ GPU check completed")

        elif button_id == "list-models-btn":
            self.log_message("📋 Listing models...", "models-log")
            try:
                self.manager.list_models()
                self.log_message("✅ Models listed", "models-log")
            except Exception as e:
                self.log_message(f"❌ Error listing models: {e}", "models-log")

        elif button_id == "pull-llama2-btn":
            self.log_message("📥 Pulling llama2 model...", "models-log")
            try:
                success = self.manager.pull_model("llama2")
                if success:
                    self.log_message(
                        "✅ llama2 model pulled successfully!", "models-log"
                    )
                else:
                    self.log_message("❌ Failed to pull llama2 model", "models-log")
            except Exception as e:
                self.log_message(f"❌ Error pulling model: {e}", "models-log")

        elif button_id == "pull-phi3-btn":
            self.log_message("📥 Pulling phi3 model...", "models-log")
            try:
                success = self.manager.pull_model("phi3")
                if success:
                    self.log_message("✅ phi3 model pulled successfully!", "models-log")
                else:
                    self.log_message("❌ Failed to pull phi3 model", "models-log")
            except Exception as e:
                self.log_message(f"❌ Error pulling model: {e}", "models-log")

        elif button_id == "generate-btn":
            self.log_message("📝 Generating corpus...", "audit-log")
            try:
                success = self.manager.generate_corpus(
                    "src/Phase1_CorpusGenerator/word_lists.json"
                )
                if success:
                    self.log_message("✅ Corpus generated successfully!", "audit-log")
                else:
                    self.log_message("❌ Failed to generate corpus", "audit-log")
            except Exception as e:
                self.log_message(f"❌ Error generating corpus: {e}", "audit-log")

        elif button_id == "audit-btn":
            self.log_message("🔍 Running audit...", "audit-log")
            try:
                success = self.manager.run_audit(
                    "Phase1_CorpusGenerator/word_lists.json"
                )
                if success:
                    self.log_message("✅ Audit completed successfully!", "audit-log")
                else:
                    self.log_message("❌ Audit failed", "audit-log")
            except Exception as e:
                self.log_message(f"❌ Error running audit: {e}", "audit-log")


def main() -> None:
    """Run the TUI application"""
    app = EquiLensTUI()
    app.run()


if __name__ == "__main__":
    main()
