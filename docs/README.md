# 📚 EquiLens Documentation

**Comprehensive documentation for the EquiLens AI Bias Detection Platform**

Welcome to the EquiLens documentation! This collection provides detailed guides, tutorials, and reference materials for all aspects of the platform.

## 🚀 Quick Start Documentation

### Essential Guides
- **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in minutes
- **[SETUP.md](../SETUP.md)** - Detailed installation and configuration
- **[EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)** - Step-by-step audit execution

### New User Path
1. 📖 Read [QUICKSTART.md](QUICKSTART.md) for immediate setup
2. 🔧 Follow [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) for customization
3. 🎯 Execute your first audit with [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)

## 🎯 Feature Documentation

### Core Features
- **[CLI_FEATURES.md](CLI_FEATURES.md)** - Interactive CLI with Rich UI components
- **[INTERRUPTION_RESUMPTION.md](INTERRUPTION_RESUMPTION.md)** - Session management and recovery
- **[PERFORMANCE_METRICS.md](PERFORMANCE_METRICS.md)** - Comprehensive analytics and tracking
- **[AUDITOR_COMPARISON.md](AUDITOR_COMPARISON.md)** - Legacy vs Enhanced auditor comparison

### Advanced Features
- **[PIPELINE.md](PIPELINE.md)** - End-to-end workflow automation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System design and components
- **[SCHEMA_SUMMARY.md](SCHEMA_SUMMARY.md)** - Data structures and formats

## 🛠️ Technical Documentation

### Setup & Configuration
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Comprehensive configuration options
- **[OLLAMA_SETUP.md](OLLAMA_SETUP.md)** - AI model service configuration
- **[example_configurations.json](example_configurations.json)** - Ready-to-use configuration templates

### Development & Architecture
- **[PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md)** - Codebase structure and organization
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design patterns
- **[ppt.md](ppt.md)** - Presentation materials and slides

## 📊 Feature Highlights

### 🎨 Enhanced CLI Experience
```bash
# Interactive audit with auto-discovery
uv run equilens audit

# Beautiful progress tracking
🔍 Processing 3/6 (50.0%): John, the nurse, is known for being very caring...
✅ Success! Score: 396915805.67 | Time: 32.8s
```

### 🔄 Interruption & Resumption
```bash
# Graceful interruption handling
🛑 Received shutdown signal. Saving progress...

# Auto-resume detection
🔄 Found interrupted audit sessions:
  1. llama2:latest - 3/6 tests (50.0% complete)
     Started: 2025-08-08 21:14:38
```

### 📈 Performance Analytics
```bash
# Comprehensive completion metrics
📊 Performance Summary:
   • Total Duration: 7m 42.3s
   • Success Rate: 100% (6/6)
   • Throughput: 0.013 tests/second
   • Peak Memory: 245.1 MB
```

## 📋 Documentation Categories

### 📚 User Guides
| Document                                         | Description                     | Audience       |
| ------------------------------------------------ | ------------------------------- | -------------- |
| [QUICKSTART.md](QUICKSTART.md)                   | Immediate setup and first audit | New users      |
| [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md)         | Step-by-step audit process      | All users      |
| [CLI_FEATURES.md](CLI_FEATURES.md)               | Interactive CLI capabilities    | All users      |
| [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md) | Customization options           | Advanced users |

### 🔧 Technical References
| Document                                           | Description            | Audience     |
| -------------------------------------------------- | ---------------------- | ------------ |
| [ARCHITECTURE.md](ARCHITECTURE.md)                 | System design overview | Developers   |
| [PROJECT_ORGANIZATION.md](PROJECT_ORGANIZATION.md) | Codebase structure     | Contributors |
| [SCHEMA_SUMMARY.md](SCHEMA_SUMMARY.md)             | Data formats           | Integrators  |
| [PIPELINE.md](PIPELINE.md)                         | Workflow automation    | DevOps       |

### ⚡ Feature Deep Dives
| Document                                                 | Description               | Use Case                   |
| -------------------------------------------------------- | ------------------------- | -------------------------- |
| [INTERRUPTION_RESUMPTION.md](INTERRUPTION_RESUMPTION.md) | Session management        | Long-running audits        |
| [PERFORMANCE_METRICS.md](PERFORMANCE_METRICS.md)         | Analytics & tracking      | Performance optimization   |
| [AUDITOR_COMPARISON.md](AUDITOR_COMPARISON.md)           | Implementation comparison | Choosing the right auditor |
| [OLLAMA_SETUP.md](OLLAMA_SETUP.md)                       | AI model configuration    | Model management           |

## 🎯 Common Use Cases

### Quick Bias Audit
```bash
# For immediate bias detection
📖 Read: QUICKSTART.md → EXECUTION_GUIDE.md
🎯 Run: uv run equilens audit
```

### Production Deployment
```bash
# For enterprise environments
📖 Read: CONFIGURATION_GUIDE.md → ARCHITECTURE.md → PIPELINE.md
🎯 Setup: Docker + CI/CD + Monitoring
```

### Development & Research
```bash
# For academic or development use
📖 Read: PROJECT_ORGANIZATION.md → AUDITOR_COMPARISON.md
🎯 Customize: Enhanced auditor + Custom metrics
```

### Large-Scale Auditing
```bash
# For comprehensive bias analysis
📖 Read: INTERRUPTION_RESUMPTION.md → PERFORMANCE_METRICS.md
🎯 Execute: Long-running sessions with resumption
```

## 🔍 Finding Information

### Search by Topic
- **Setup**: QUICKSTART.md, SETUP.md, CONFIGURATION_GUIDE.md
- **Usage**: EXECUTION_GUIDE.md, CLI_FEATURES.md
- **Performance**: PERFORMANCE_METRICS.md, AUDITOR_COMPARISON.md
- **Reliability**: INTERRUPTION_RESUMPTION.md, ARCHITECTURE.md
- **Integration**: PIPELINE.md, example_configurations.json

### Search by Audience
- **New Users**: QUICKSTART.md, EXECUTION_GUIDE.md
- **Power Users**: CLI_FEATURES.md, CONFIGURATION_GUIDE.md
- **Developers**: ARCHITECTURE.md, PROJECT_ORGANIZATION.md
- **DevOps**: PIPELINE.md, OLLAMA_SETUP.md

## 📈 Recent Documentation Updates

### New Features Documented
- ✅ **Interruption & Resumption System** - Complete session management
- ✅ **Enhanced CLI Interface** - Rich UI with auto-discovery
- ✅ **Performance Metrics** - Comprehensive analytics tracking
- ✅ **Auditor Comparison** - Legacy vs Enhanced analysis
- ✅ **Cross-Platform Support** - Windows, macOS, Linux compatibility

### Enhanced Existing Docs
- 🔄 **QUICKSTART.md** - Updated with new CLI features
- 🔄 **EXECUTION_GUIDE.md** - Added resumption workflows
- 🔄 **CONFIGURATION_GUIDE.md** - New configuration options
- 🔄 **ARCHITECTURE.md** - Updated component diagrams

## 🎉 Contributing to Documentation

### Documentation Standards
- 📝 Use clear, actionable language
- 🎨 Include visual examples (code blocks, outputs)
- 🔗 Cross-reference related documents
- 📊 Provide practical examples
- ⚡ Keep information current

### Feedback & Improvements
- 🐛 Report documentation issues via GitHub Issues
- 💡 Suggest improvements via Pull Requests
- 📧 Contact maintainers for major changes
- 🔄 Regular reviews and updates

Ready to explore EquiLens? Start with [QUICKSTART.md](QUICKSTART.md) for immediate results or dive into specific features with the guides above! 🚀
