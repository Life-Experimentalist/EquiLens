# Configuration Guide — EquiLens

EquiLens uses JSON configuration for audits, corpus generation, and analysis. Run with `uv run equilens --config <file>`.

Example audit config
```json
{
  "model": "llama2:latest",
  "corpus": "Phase1_CorpusGenerator/corpus/my_corpus.csv",
  "ollama_options": {"temperature":0.2},
  "audit_settings": {"samples_per_prompt":3}
}
```

Run
```powershell
uv run equilens audit --config configs/audit_config.json
```
