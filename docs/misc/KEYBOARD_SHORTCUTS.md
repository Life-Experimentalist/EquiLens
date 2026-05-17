# ⌨️ Keyboard Shortcuts for EquiLens Development

## Main Shortcuts

| Shortcut               | Action          | Description                                |
| ---------------------- | --------------- | ------------------------------------------ |
| **`Ctrl+Shift+B`**     | Open Task Menu  | Launch any EquiLens task (25+ options)     |
| **`Ctrl+`** (backtick) | Toggle Terminal | Show/hide integrated terminal              |
| **`Ctrl+J`**           | Toggle Panel    | Show/hide bottom panel (terminal + output) |
| **`Ctrl+K, Ctrl+O`**   | Open Folder     | Open new workspace                         |

---

## Task Groups in `Ctrl+Shift+B` Menu

### 🧪 "Build" Group (`Ctrl+Shift+B` → Filter: "build")
- Dev: Run Tests
- Dev: Lint Code (Ruff)
- Dev: Format Code (Ruff)
- Dev: Type Check (MyPy)
- Dev: Full Check (Lint + Type)

### 🧪 "Test" Group (`Ctrl+Shift+B` → Filter: "test")
- Audit: Run with Logprobs (Default)
- Audit: Run with Timing Fallback
- Audit: Resume Session
- Analyze: Generate Reports
- Analyze: Advanced with AI Insights

### 📋 Other Tasks
All other tasks under "Run Task" → select by name

---

## Task Execution Shortcuts

Once a task is open, use:

| Shortcut             | Action             |
| -------------------- | ------------------ |
| **`Ctrl+Shift+B`**   | Reopen task menu   |
| **`Ctrl+C`**         | Stop running task  |
| **`↑/↓ Arrow Keys`** | Navigate task list |
| **`Enter`**          | Run selected task  |
| **`Escape`**         | Close task menu    |

---

## Terminal Shortcuts

| Shortcut                  | Action                     |
| ------------------------- | -------------------------- |
| **`Ctrl+`** (backtick)    | Toggle terminal            |
| **`Ctrl+Shift+``**        | Create new terminal        |
| **`Ctrl+Alt+Right/Left`** | Navigate between terminals |
| **`Ctrl+L`**              | Clear terminal             |
| **`Ctrl+A`**              | Select all in terminal     |
| **`Ctrl+C`**              | Cancel running command     |

---

## Quick Commands in Terminal

```powershell
# Show EquiLens help
uv run equilens --help

# Check status
uv run equilens status

# Run audit (default: logprobs)
uv run equilens audit run --model=phi3:mini --corpus=data/corpus/gender_bias.csv

# Analyze results
uv run equilens analyze --advanced --use-ai

# Format code
uv run ruff format src/

# Check code quality
uv run ruff check src/ && uv run mypy src/
```

---

## VS Code Command Palette

Press **`Ctrl+Shift+P`** for Command Palette:

| Command          | Type in Palette | What It Does                  |
| ---------------- | --------------- | ----------------------------- |
| Run Task         | `task:`         | Show all tasks                |
| Format Document  | `format`        | Auto-format current file      |
| Go to File       | `Ctrl+P`        | Quick file search             |
| Go to Line       | `Ctrl+G`        | Jump to line number           |
| Search in Files  | `Ctrl+Shift+F`  | Find across project           |
| Replace in Files | `Ctrl+H`        | Find & replace across project |
| Open Settings    | `settings`      | Open VS Code settings         |
| Open Keybindings | `keybindings`   | Edit keyboard shortcuts       |

---

## Recommended Workflow

### Quick Audit Session
```
1. Press Ctrl+Shift+B
2. Search for "Audit: Run with Logprobs"
3. Press Enter to start
4. Wait for completion
5. View results in integrated terminal
```

### Full Development Cycle
```
1. Make code changes
2. Press Ctrl+Shift+B → "Dev: Format Code"
3. Press Ctrl+Shift+B → "Dev: Full Check"
4. If all good, press Ctrl+Shift+B → "Audit"
5. Press Ctrl+Shift+B → "Analyze: Advanced"
```

### Quick Test & Lint
```
1. Save file (Ctrl+S)
2. Press Ctrl+Shift+B → "Dev: Lint Code"
3. Press Ctrl+Shift+B → "Dev: Type Check"
4. Review output in terminal
```

---

## Custom Keyboard Shortcuts

You can customize shortcuts in VS Code:

1. Press **`Ctrl+K, Ctrl+S`** to open Keybindings
2. Search for "Run Build Task"
3. Click the pencil icon to assign custom key

### Suggested Custom Bindings
```json
// .vscode/keybindings.json examples:

[
    {
        "key": "ctrl+alt+a",
        "command": "workbench.action.tasks.runTask",
        "args": "Audit: Run with Logprobs (Default)"
    },
    {
        "key": "ctrl+alt+z",
        "command": "workbench.action.tasks.runTask",
        "args": "Analyze: Advanced with AI Insights"
    },
    {
        "key": "ctrl+alt+l",
        "command": "workbench.action.tasks.runTask",
        "args": "Dev: Lint Code (Ruff)"
    }
]
```

---

## Navigation Tips

| Action                           | Shortcut                      |
| -------------------------------- | ----------------------------- |
| **Open file browser sidebar**    | **`Ctrl+B`**                  |
| **Focus on file explorer**       | **`Ctrl+Shift+E`**            |
| **Focus on search**              | **`Ctrl+Shift+F`**            |
| **Focus on source control**      | **`Ctrl+Shift+G`**            |
| **Focus on debug**               | **`Ctrl+Shift+D`**            |
| **Focus on extensions**          | **`Ctrl+Shift+X`**            |
| **Switch between editor groups** | **`Ctrl+K, Ctrl+Left/Right`** |
| **Split editor**                 | **`Ctrl+\`**                  |

---

## Terminal Output Navigation

When a task completes:

| Action                | How                                |
| --------------------- | ---------------------------------- |
| **Scroll terminal**   | Scroll wheel or **`Page Up/Down`** |
| **Search in output**  | **`Ctrl+F`** in terminal           |
| **Select all output** | **`Ctrl+A`**                       |
| **Copy output**       | Select + **`Ctrl+C`**              |
| **Clear terminal**    | **`Ctrl+L`**                       |

---

## Often-Used Task Shortcuts (Suggested)

Add these to `.vscode/keybindings.json` for lightning-fast access:

```json
{
    "key": "alt+a",
    "command": "workbench.action.tasks.runTask",
    "args": "Audit: Run with Logprobs (Default)"
},
{
    "key": "alt+n",
    "command": "workbench.action.tasks.runTask",
    "args": "Analyze: Advanced with AI Insights"
},
{
    "key": "alt+l",
    "command": "workbench.action.tasks.runTask",
    "args": "Dev: Lint Code (Ruff)"
},
{
    "key": "alt+f",
    "command": "workbench.action.tasks.runTask",
    "args": "Dev: Format Code (Ruff)"
}
```

Then use:
- **`Alt+A`** = Audit
- **`Alt+N`** = aNalyze
- **`Alt+L`** = Lint
- **`Alt+F`** = Format

---

**Pro Tip:** Learn **`Ctrl+Shift+B`** first — it's your gateway to all tasks! 🚀

Last Updated: March 5, 2026
