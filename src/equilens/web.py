"""
Web interface for EquiLens using FastAPI

A modern web interface for the EquiLens AI bias detection platform.
This will provide a browser-based UI for all EquiLens operations.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from equilens.core.manager import EquiLensManager

app = FastAPI(
    title="EquiLens Web Interface",
    description="AI Bias Detection Platform - Web UI",
    version="0.1.0",
)

# Global manager instance
manager = EquiLensManager()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>EquiLens - AI Bias Detection Platform</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                line-height: 1.6;
            }
            .header { text-align: center; margin-bottom: 40px; }
            .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
            .status-card {
                border: 2px solid #e1e5e9;
                padding: 20px;
                border-radius: 8px;
                background: #f8f9fa;
            }
            .status-card.ready { border-color: #28a745; background: #d4edda; }
            .status-card.warning { border-color: #ffc107; background: #fff3cd; }
            .status-card.error { border-color: #dc3545; background: #f8d7da; }
            .actions { margin: 20px 0; }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                margin: 5px;
                background: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                border: none;
                cursor: pointer;
            }
            .btn:hover { background: #0056b3; }
            .btn.success { background: #28a745; }
            .btn.danger { background: #dc3545; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🔍 EquiLens</h1>
            <p>AI Bias Detection Platform</p>
        </div>

        <div class="status-grid">
            <div class="status-card warning">
                <h3>🚧 Coming Soon</h3>
                <p>Web interface is under development</p>
            </div>
            <div class="status-card">
                <h3>📱 Current Options</h3>
                <p>Use CLI or TUI for now</p>
            </div>
        </div>

        <div class="actions">
            <h3>Available Interfaces:</h3>
            <p>
                <strong>Command Line:</strong><br>
                <code>uv run equilens --help</code>
            </p>
            <p>
                <strong>Interactive Terminal UI:</strong><br>
                <code>uv run equilens tui</code>
            </p>
        </div>

        <div>
            <h3>🚀 Quick Start</h3>
            <ol>
                <li>Check GPU: <code>uv run equilens gpu-check</code></li>
                <li>Start services: <code>uv run equilens start</code></li>
                <li>Run audit: <code>uv run equilens audit</code></li>
                <li>Analyze results: <code>uv run equilens analyze results.csv</code></li>
            </ol>
        </div>
    </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    """Get system status"""
    try:
        status = manager.check_system_status()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/models")
async def get_models():
    """Get available models"""
    try:
        # This would need to be implemented to return model data
        return {"success": True, "models": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/start")
async def start_services():
    """Start EquiLens services"""
    try:
        success = manager.start_services()
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/stop")
async def stop_services():
    """Stop EquiLens services"""
    try:
        success = manager.stop_services()
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    """Run the web server"""
    print("🌐 Starting EquiLens Web Interface...")
    print("📍 URL: http://localhost:8000")
    print("🚧 Note: This is a preview - full web UI coming soon!")

    uvicorn.run(
        "equilens.web:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    main()
