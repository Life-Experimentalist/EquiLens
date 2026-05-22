# MCP servers - EquiLens (project-specific)

This file lists MCP servers and lightweight guidance relevant to the EquiLens codebase.

- **upstash/context7** — Use to fetch up-to-date docs, API snippets, or external library examples when researching integrations and writing doc-driven code search.
- **microsoft/playwright-mcp** — Useful for end-to-end UI testing for Gradio or any web UI components; run headful/headless browser tests.
- **chromedevtools/chrome-devtools-mcp** — Attach to a running Chrome for debugging complex UI behavior or inspecting network/console.
- **microsoft/markitdown** — Render Markdown + Mermaid diagrams when generating or previewing docs programmatically.
- **sqlite** — Local lightweight storage server for small experiment state or reproducible local runs.

Usage notes:
- Add servers with your MCP client (example): `uvx mcp-add upstash/context7` or use your local MCP gateway.
- Prefer ephemeral configuration per developer — do not hardcode credentials in the repo.

If you want, I can generate a matching `.mcp/servers.json` or add example `mcp-add` commands for each server.
