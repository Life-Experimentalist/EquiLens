# Interactive Analytics Selection Guide

## Overview

EquiLens now features **interactive prompts** that guide you through choosing the right analytics mode for your needs. No more command-line flags to remember!

---

## 🎯 Two Ways to Choose Analytics

### 1. **During Audit** (Recommended)
When starting a NEW audit, EquiLens asks if you want automatic analytics after completion.

### 2. **During Analysis**
When running `analyze` command without `--advanced` flag, you'll be asked to choose.

---

## 📋 Audit-Time Selection

### When It Appears
- **Only on NEW audits** (not when resuming)
- After Step 3 (Configuration Review)
- Before audit execution begins

### What You'll See

```
╭────────────────────────────────────────────────────────╮
│ Step 3.5: Post-Audit Analytics Preference              │
│                                                        │
│ Would you like to automatically run analytics after   │
│ the audit completes?                                   │
│                                                        │
│ Options:                                               │
│   1. None - Skip automatic analysis                    │
│   2. Standard - Quick analysis with basic charts       │
│   3. Advanced - Comprehensive with 8+ charts           │
╰────────────────────────────────────────────────────────╯

Select analytics preference (1/2/3): _
```

### Options Explained

| Choice | Name | What Happens | Time |
|--------|------|--------------|------|
| **1** | None | No auto-analysis (run manually later) | - |
| **2** | Standard | Auto-runs basic analysis after audit | +5 sec |
| **3** | Advanced | Auto-runs comprehensive analysis after audit | +15 sec |

### Example Flow

```bash
# Start audit
uv run equilens audit

# ... (model selection, corpus selection, config review)

# Step 3.5 appears:
Select analytics preference (1/2/3): 3  # Choose Advanced

✓ Advanced analytics will run after audit

# ... (audit runs)

✅ Audit Completed Successfully!

# Automatically runs advanced analytics:
🚀 Auto-Running Advanced Analytics
...
✅ Advanced Analysis Complete!
```

### Benefits
✅ Set preference once at the start
✅ No need to remember to run analysis
✅ Seamless workflow from audit → analysis
✅ Always get consistent results

---

## 📊 Analyze-Time Selection

### When It Appears
- When running `uv run equilens analyze` **without** `--advanced` flag
- After Step 1 (Results File Selection)
- Before analysis execution begins

### What You'll See

```
╭──────────────────────────────────────────────────────────────╮
│ Step 1.5: Analytics Mode Selection                           │
│                                                              │
│ Choose the type of analysis you want to perform:             │
│                                                              │
│ Options:                                                     │
│   1. Standard - Quick analysis (~5 sec)                      │
│      • Single bias_report.png visualization                  │
│      • Console statistics summary                            │
│      • Perfect for quick checks                              │
│                                                              │
│   2. Advanced - Comprehensive analysis (~15 sec)             │
│      • comprehensive_dashboard.png                           │
│      • 7 additional professional charts                      │
│      • statistical_report.md                                 │
│      • Effect sizes, t-tests, CIs                            │
│      • Perfect for presentations                             │
│                                                              │
│ Tip: Use --advanced flag to skip this prompt next time       │
╰──────────────────────────────────────────────────────────────╯

Select analysis mode (1 or 2): _
```

### Options Explained

| Choice | Mode | Output | Best For |
|--------|------|--------|----------|
| **1** | Standard | 1 PNG chart | Quick checks, iterations |
| **2** | Advanced | 8 files (7 PNG + 1 MD) | Presentations, research |

### Example Flows

#### Quick Analysis
```bash
uv run equilens analyze

# Step 1.5:
Select analysis mode (1 or 2): 1

✓ Standard analytics selected
# ... generates bias_report.png
```

#### Comprehensive Analysis
```bash
uv run equilens analyze

# Step 1.5:
Select analysis mode (1 or 2): 2

✓ Advanced analytics selected
# ... generates 8 files
```

#### Skip Prompt with Flag
```bash
# Directly run advanced (no prompt)
uv run equilens analyze --advanced

# Directly run standard (no prompt)
uv run equilens analyze
# (then choose 1)
```

---

## 🎓 Decision Guide

### Choose **NONE** (Audit-time) if:
- ❌ Don't want any analysis right now
- ❌ Will analyze results later manually
- ❌ Testing/debugging audit setup

### Choose **STANDARD** if:
- ✅ Quick visual check needed
- ✅ Iterating on model/corpus
- ✅ Just want basic bar chart
- ✅ Time-constrained (~5 sec)

### Choose **ADVANCED** if:
- ✅ Presenting to faculty/stakeholders
- ✅ Writing research paper
- ✅ Need statistical rigor
- ✅ Want comprehensive documentation
- ✅ Publication-quality figures (~15 sec)

---

## 📖 Complete Workflows

### Workflow 1: Full Automation (Recommended)

```bash
# 1. Start audit
uv run equilens audit

# 2. At Step 3.5, choose advanced
Select analytics preference (1/2/3): 3

# 3. Wait for completion
# ✅ Audit done
# 🚀 Auto-running advanced analytics
# ✅ Analysis done

# 4. Review all outputs
# - results CSV
# - 8 visualization files
# - statistical report
```

**Time**: Audit time + ~15 seconds
**Output**: Complete analysis package

---

### Workflow 2: Manual Analysis Later

```bash
# 1. Start audit
uv run equilens audit

# 2. At Step 3.5, skip analytics
Select analytics preference (1/2/3): 1

# ✅ Audit done (no analysis)

# 3. Later, run analysis
uv run equilens analyze

# 4. At Step 1.5, choose mode
Select analysis mode (1 or 2): 2

# ✅ Analysis done
```

**Time**: Audit time, then later +15 seconds
**Benefit**: Separate concerns, review results first

---

### Workflow 3: Quick Iteration

```bash
# Testing different models quickly

# Audit 1 (Llama2)
uv run equilens audit
Select analytics preference: 2  # Standard

# Audit 2 (GPT-4)
uv run equilens audit
Select analytics preference: 2  # Standard

# Audit 3 (Mistral)
uv run equilens audit
Select analytics preference: 2  # Standard

# Later, do comprehensive analysis on best model
uv run equilens analyze --advanced --results results/best_model.csv
```

**Time**: Fast iterations with quick checks
**Benefit**: Defer comprehensive analysis to final choice

---

## 💡 Pro Tips

### Tip 1: Resume Never Asks
```bash
# First run: asks for preference
uv run equilens audit
Select analytics preference: 3

# Interrupted! Resume:
uv run equilens audit --resume path/to/progress.json

# ✅ No prompt - continues with original preference
```

### Tip 2: Use Flags for Scripts
```bash
# In automation scripts, use explicit flags:
uv run equilens audit --model llama2:latest --corpus test.csv
# (Will ask interactively)

# Better for scripts:
uv run equilens analyze --advanced --results results.csv
# (No prompt, direct execution)
```

### Tip 3: Default Choices
- **Audit-time**: Default is "1" (None)
- **Analyze-time**: Default is "1" (Standard)
- Just press Enter to accept default

### Tip 4: Silent Mode Still Works
```bash
# Interactive + silent output
uv run equilens analyze --silent
Select analysis mode: 2
# Runs advanced, suppresses subprocess output
```

---

## 🚫 When Prompts DON'T Appear

### Audit Command
- ❌ **Resume mode** (`--resume` flag used)
  - Original preference is preserved
  - No re-selection needed

### Analyze Command
- ❌ **Explicit `--advanced` flag** provided
  - Goes directly to advanced mode
  - Skips prompt entirely

---

## 📋 Quick Reference

### Command Matrix

| Command | Flag | Prompt? | Mode |
|---------|------|---------|------|
| `uv run equilens audit` | None | ✅ Yes (Step 3.5) | Interactive |
| `uv run equilens audit --resume ...` | `--resume` | ❌ No | Uses saved pref |
| `uv run equilens analyze` | None | ✅ Yes (Step 1.5) | Interactive |
| `uv run equilens analyze --advanced` | `--advanced` | ❌ No | Advanced |
| `uv run equilens analyze --results ...` | `--results` | ✅ Yes (Step 1.5) | Interactive |

---

## 🎯 Recommended Patterns

### For Learning / First Time
```bash
uv run equilens audit        # Interactive, choose None
uv run equilens analyze      # Interactive, try Standard first
uv run equilens analyze      # Interactive, then try Advanced
```

### For Regular Use
```bash
uv run equilens audit        # Choose Advanced at Step 3.5
# Done! Everything auto-generated
```

### For Scripting / Automation
```bash
# Non-interactive mode:
uv run equilens audit --model $MODEL --corpus $CORPUS --silent
# Manual analysis later with explicit flag:
uv run equilens analyze --advanced --results $RESULTS --silent
```

---

## ✨ Summary

**New Interactive Features:**

1. ✅ **Audit-time prompt** (Step 3.5)
   - Choose analytics preference before audit
   - Auto-runs after completion
   - Only appears for NEW audits

2. ✅ **Analyze-time prompt** (Step 1.5)
   - Choose Standard or Advanced
   - Appears when no `--advanced` flag
   - Always appears unless flag used

**Benefits:**

- 🎯 No need to remember flags
- 🔄 Seamless workflow
- 📚 Educational (explains options)
- ⚡ Fast defaults (press Enter)
- 🚀 Automation-friendly (flags still work)

**Get Started:**

```bash
uv run equilens audit
# Follow the prompts!
```

---

**Questions? Check the main documentation or run with `--help`!** 🚀
