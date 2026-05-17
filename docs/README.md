# EquiLens Documentation Index

This directory contains the user and developer documentation for EquiLens. The docs have been organized into focused subfolders to make it easier to find installation instructions, Docker guidance, architecture diagrams, analytics and auditing reference material.

## Quick links
- **Get Started**: `setup/GRADIO_QUICKSTART.md` - Web interface quick start
- **CLI Quick Start**: `setup/QUICK_START.md` - UV and command reference
- **Setup scripts**: `../scripts/setup/` - Ollama, Docker, local setup scripts
- **Configuration**: `setup/CONFIGURATION_GUIDE.md`
- **Execution**: `setup/EXECUTION_GUIDE.md`
- **UV Scripts**: `setup/UV_SCRIPTS_GUIDE.md` - UV workflow reference

## Table of contents

### Setup & Getting Started
- `setup/GRADIO_QUICKSTART.md` ⭐ **NEW** - Quick start with Gradio web interface and backend API
- `setup/NEW_FEATURES.md` ⭐ **NEW** - Overview of new Gradio/Backend architecture features
- `setup/PORT_MANAGEMENT.md` ⭐ **NEW** - Flexible port configuration for multiple instances
- `setup/QUICKSTART.md` - Quick start and installation steps (UV, venv, Docker) with verification
- `setup/CONFIGURATION_GUIDE.md` - Configuration options and settings
- `setup/EXECUTION_GUIDE.md` - Execution workflows and command reference
- `setup/SETUP_SCRIPTS_GUIDE.md` - Guide to installer scripts and one-command options
- `setup/SETUP_IMPROVEMENTS.md` - Notes about recent setup improvements
- `setup/SMART_CONFIG_IMPLEMENTATION.md` - Smart configuration system details
- `setup/SMART_CONFIG_QUICKREF.md` - Smart config quick reference
- `setup/TROUBLESHOOTING_SETUP.md` - Troubleshooting tips for common setup problems

### Docker & Deployment
- `docker/DEPLOYMENT.md` - Deployment guide for production environments
- `docker/DOCKER_README.md` - Docker quick reference: access points (Gradio/API/Ollama), common commands
- `docker/DOCKER_SETUP.md` - Full Docker setup and compose examples
- `docker/DOCKER_CONFIG_GUIDE.md` - Configurable ports, volumes, container names, environment variables
- `docker/DOCKER_CONFIG_QUICK.md` - Quick Docker configuration reference
- `docker/DOCKER_ATTESTATION_SETUP.md` - Docker image attestation setup
- `docker/DOCKER_NETWORKING_FIX.md` - Networking configuration and fixes
- `docker/DOCKER_CHECKLIST.md` - Checklist for building and deploying container images

### Architecture & Design
- `architecture/BACKEND_ARCHITECTURE.md` ⭐ **NEW** - Backend API architecture, endpoints, database schema
- `architecture/INTERFACE_ARCHITECTURE.md` ⭐ **NEW** - Dual interface design (CLI + Web), state sharing
- `architecture/CLI_COMPATIBILITY.md` ⭐ **NEW** - CLI commands work independently (no backend required!)
- `architecture/IMPLEMENTATION_SUMMARY.md` ⭐ **NEW** - Complete implementation summary of new features
- `architecture/ARCHITECTURE.md` - System architecture diagrams (Mermaid) and component interactions
- `architecture/ARCHITECTURE_SIMPLE.md` - Simplified architecture overview
- `architecture/PIPELINE.md` - Dataflow and phase-to-phase pipeline overview

### Analytics & Reporting
- `analytics/LLM_DATA_FLOW.md` ⭐ **NEW** - How data flows to LLMs for AI-powered insights
- `analytics/QUICK_REFERENCE.md` - Quick reference for analytics workflows
- `analytics/ADVANCED_ANALYTICS_GUIDE.md` - Advanced analytics with professional visualizations and reports
- `analytics/INTERACTIVE_ANALYTICS_GUIDE.md` - Interactive analytics features
- `analytics/FLEXIBLE_ANALYTICS_REFERENCE.md` - Flexible analytics configuration
- `analytics/MULTI_CATEGORY_QUICK_REFERENCE.md` - Multi-category analysis reference

### Auditing & Testing
- `auditing/AUDITING_MECHANISM.md` - Deep dive into auditors, workflows, retry/resume features
- `auditing/ENHANCED_AUDITOR_DEFAULT.md` - Enhanced auditor as default configuration
- `auditing/OLLAMA_SETUP.md` - Ollama-specific setup notes and options
- `auditing/EXISTING_OLLAMA_GUIDE.md` - Guide for using existing Ollama installations
- `auditing/OLLAMA_FLEXIBLE_SETUP.md` - Flexible Ollama configuration options
- `auditing/SMART_IMAGE_MANAGEMENT.md` - Smart Docker image management
- `auditing/VOLUME_MANAGEMENT.md` - Volume and data persistence management

### Miscellaneous
- `misc/CLI_FIXES.md` - CLI fixes and improvements
- `misc/READY_TO_PUBLISH.md` - Publication readiness checklist
- `misc/CODE_ANALYSIS_REPORT.md` - Code analysis and recommended refactors
- `misc/REPORT.md` - Miscellaneous project reports
- `misc/SCRIPT_REORGANIZATION_PLAN.md` - Documentation reorganization plan

### Archived
- `archived/ONE_CLICK_SETUP.md` - Archived historical one-click installer instructions
- `archived/RECOVERY_NOTES.md` - Archived recovery and troubleshooting notes

## How to use this index
- ⭐ marks new documentation added for the Gradio/Backend architecture
- Click any file path above to open the detailed document
- Start with `setup/GRADIO_QUICKSTART.md` for the new web interface
- Use `setup/QUICKSTART.md` for traditional CLI-only workflows
- If a link points to an old location, please open an issue via the repository's issue tracker

## Key Documentation Paths

### For New Users
1. Start here: `setup/GRADIO_QUICKSTART.md` - Web interface quick start
2. Then read: `setup/NEW_FEATURES.md` - What's new in EquiLens
3. Reference: `setup/CONFIGURATION_GUIDE.md` - Configure your setup
4. Troubleshoot: `setup/TROUBLESHOOTING_SETUP.md` - Common issues

### For Developers
1. Architecture: `architecture/BACKEND_ARCHITECTURE.md` - Backend API design
2. Implementation: `architecture/IMPLEMENTATION_SUMMARY.md` - Technical details
3. System Design: `architecture/ARCHITECTURE.md` - Overall system architecture
4. Data Flow: `analytics/LLM_DATA_FLOW.md` - AI integration details

### For DevOps/Deployment
1. Docker Setup: `docker/DOCKER_README.md` - Quick Docker reference
2. Deployment: `docker/DEPLOYMENT.md` - Production deployment guide
3. Configuration: `docker/DOCKER_CONFIG_GUIDE.md` - Docker configuration options
4. Checklist: `docker/DOCKER_CHECKLIST.md` - Pre-deployment checklist

## Contributing
- To add or modify docs, create a branch and submit a PR against `main`
- Keep individual docs focused (one concept per file)
- Add a one-line summary in this index when adding new files
- Use relative links when linking between docs so they work both on GitHub and locally
- Mark new documentation with ⭐ for visibility

---

**Last Updated**: 2025-10-20
**Documentation Version**: 2.0 (Gradio/Backend Architecture)
