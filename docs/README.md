# 📚 EquiLens Documentation

**Comprehensive documentation for the EquiLens AI Bias Detection Platform**

This directory contains user-facing documentation for EquiLens — a state-of-the-art bias detection framework for Small and Large Language Models, built as part of a research project at Amrita Vishwa Vidyapeetham.

## 🚀 Getting Started

| Document                               | Description                                  | When to Use                          |
| -------------------------------------- | -------------------------------------------- | ------------------------------------ |
| **[QUICKSTART.md](QUICKSTART.md)**     | Complete setup guide with installation steps | **Start here** - First-time users    |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture and three-phase design   | Understanding the platform structure |

## 🏗️ Core Documentation

| Document                                     | Description                                      | Audience                 |
| -------------------------------------------- | ------------------------------------------------ | ------------------------ |
| **[PIPELINE.md](PIPELINE.md)**               | End-to-end workflow and execution patterns       | All users                |
| **[CONFIGURATION.md](CONFIGURATION.md)**     | Configuration files, corpus creation, validation | Power users, researchers |
| **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** | Local and containerized execution                | Production deployments   |

## 📊 Research & Analysis

| Document                                         | Description                                     | Target                       |
| ------------------------------------------------ | ----------------------------------------------- | ---------------------------- |
| **[BIAS_ANALYSIS.md](BIAS_ANALYSIS.md)**         | Bias measurement methodology and interpretation | Researchers, academics       |
| **[CORPUS_GENERATION.md](CORPUS_GENERATION.md)** | Creating custom bias detection datasets         | Content creators             |
| **[ENHANCED_AUDITOR.md](ENHANCED_AUDITOR.md)**   | Advanced auditor features and customization     | Research and experimentation |

## 🛠️ Technical References

| Document                                     | Description                                   | Audience     |
| -------------------------------------------- | --------------------------------------------- | ------------ |
| **[API_REFERENCE.md](API_REFERENCE.md)**     | CLI commands and parameters                   | Developers   |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues and solutions                   | Support      |
| **[CONTRIBUTING.md](CONTRIBUTING.md)**       | Development setup and contribution guidelines | Contributors |

## 📱 Platform Features

EquiLens provides a comprehensive bias detection platform with:

- **🎯 Interactive CLI Interface** - Rich terminal UI with guided workflows
- **⏱️ Real-Time ETA Estimation** - Accurate timing with dynamic updates
- **⚡ GPU Acceleration** - NVIDIA CUDA support for 5-10x faster inference
- **🔄 Interruption & Resume** - Graceful session management and recovery
- **🎨 Enhanced Progress Display** - Rich progress bars with performance metrics
- **📊 Comprehensive Analytics** - Detailed bias analysis and reporting
- **🛡️ Custom Bias Categories** - Create domain-specific bias detection datasets
- **🐳 Docker Integration** - Containerized deployment with GPU support

## 🌟 Quick Navigation

### New Users
1. **[QUICKSTART.md](QUICKSTART.md)** - Complete setup guide from installation to first audit
2. **[PIPELINE.md](PIPELINE.md)** - Understand the complete workflow
3. **[CORPUS_GENERATION.md](CORPUS_GENERATION.md)** - Create your first custom dataset

### Researchers
1. **[BIAS_ANALYSIS.md](BIAS_ANALYSIS.md)** - Research methodology and bias measurement
2. **[ENHANCED_AUDITOR.md](ENHANCED_AUDITOR.md)** - Advanced features and statistical analysis
3. **[RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)** - Interpreting audit findings

### Developers
1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and project structure
2. **[API_REFERENCE.md](API_REFERENCE.md)** - CLI commands and parameters
3. **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup and guidelines

### Advanced Users
1. **[PERFORMANCE.md](PERFORMANCE.md)** - Optimization strategies and metrics
2. **[ETA_USAGE_GUIDE.md](ETA_USAGE_GUIDE.md)** - Time estimation and planning
3. **[RESUME.md](RESUME.md)** - Session management and recovery
4. **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)** - Ollama configuration and troubleshooting

## 🎯 Key Concepts

- **Phase 1: Corpus Generation** - Create balanced, reproducible bias detection datasets ✅ **RELEASED**
- **Phase 2: Model Auditing** - Execute bias tests against language models
- **Phase 3: Analysis & Reporting** - Analyze results and generate comprehensive reports

## 📖 External Resources

- **[GitHub Repository](https://github.com/Life-Experimentalists/EquiLens)** - Source code and issues
- **[Zenodo DOI](https://doi.org/10.5281/zenodo.17014103)** - Archived releases
- **[Project Website](https://life-experimentalists.github.io/EquiLens/)** - Interactive documentation
- **[Phase 1 CorpusGen](https://github.com/Life-Experimentalists/EquiLens/tree/main/src/Phase1_CorpusGenerator)** - Released corpus generator

## 🆘 Need Help?

- **Quick Issues**: Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Setup Problems**: See [QUICKSTART.md](QUICKSTART.md)
- **Research Questions**: Review [BIAS_ANALYSIS.md](BIAS_ANALYSIS.md)
- **Performance Issues**: Check [PERFORMANCE.md](PERFORMANCE.md)
- **Session Problems**: See [RESUME.md](RESUME.md)
- **Report Bugs**: [GitHub Issues](https://github.com/Life-Experimentalists/EquiLens/issues)

---

**Note**: This documentation structure reflects the current state of EquiLens as described in the main README.md and matches the website at https://life-experimentalists.github.io/EquiLens/

## 📁 Consolidated Documentation Structure

The docs directory has been streamlined to eliminate redundancy and improve navigation:

### ✅ Retained Core Documents
- **QUICKSTART.md** - Comprehensive setup guide (merged with INSTALLATION.md)
- **ARCHITECTURE.md** - Complete system architecture (merged with PROJECT_ORGANIZATION.md and SCHEMA_SUMMARY.md)
- **ENHANCED_AUDITOR.md** - Advanced auditor features (merged with ENHANCED_AUDITOR_SUMMARY.md)
- **PERFORMANCE.md** - Performance optimization (merged with PERFORMANCE_METRICS.md)
- **CONFIGURATION.md** - Configuration guide (merged with CONFIGURATION_GUIDE.md)
- **ETA_USAGE_GUIDE.md** - ETA estimation guide (merged with ETA_INTEGRATION.md)
- **RESUME.md** - Session management (merged with INTERRUPTION_RESUMPTION.md)
- **OLLAMA_SETUP.md** - Ollama setup and troubleshooting (merged with OLLAMA_CONNECTION_ISSUES.md)

### ❌ Removed Redundant Files
- INSTALLATION.md → merged into QUICKSTART.md
- PROJECT_ORGANIZATION.md → merged into ARCHITECTURE.md
- SCHEMA_SUMMARY.md → merged into ARCHITECTURE.md
- ORGANIZATION_SUMMARY.md → merged into ARCHITECTURE.md
- ENHANCED_AUDITOR_SUMMARY.md → merged into ENHANCED_AUDITOR.md
- PERFORMANCE_METRICS.md → merged into PERFORMANCE.md
- CONFIGURATION_GUIDE.md → merged into CONFIGURATION.md
- ETA_INTEGRATION.md → merged into ETA_USAGE_GUIDE.md
- INTERRUPTION_RESUMPTION.md → merged into RESUME.md
- OLLAMA_CONNECTION_ISSUES.md → merged into OLLAMA_SETUP.md
- DYNAMIC_CONCURRENCY_AUDITING.md → removed (covered in ENHANCED_AUDITOR.md)
- AUDIT_SYSTEM.md → removed (covered in ARCHITECTURE.md)
- SYSTEM_INSTRUCTIONS_BIAS_IMPACT.md → removed (specialized content)

---

**Note**: This documentation structure reflects the current state of EquiLens as described in the main README.md and matches the website at https://life-experimentalists.github.io/EquiLens/

*Last Updated: September 2025*
