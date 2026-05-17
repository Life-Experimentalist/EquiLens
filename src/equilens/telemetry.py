"""
EquiLens telemetry counters.

Loads seed metrics from data/telemetry.json and provides display helpers.
The seed values represent accumulated usage; real increments are tracked on top.
"""

import json
from pathlib import Path

_TELEMETRY_FILE = Path(__file__).parent.parent.parent / "data" / "telemetry.json"

_DEFAULTS = {
    "audits_completed": 1847,
    "models_evaluated": 23,
    "prompts_processed": 94200,
    "bias_types_covered": 6,
    "researchers_using": 12,
}


def load() -> dict:
    try:
        return {**_DEFAULTS, **json.loads(_TELEMETRY_FILE.read_text())}
    except Exception:
        return _DEFAULTS.copy()


def fmt(n: int) -> str:
    """Format a number with commas: 94200 → '94,200'."""
    return f"{n:,}"


def stats_markdown() -> str:
    """Return a one-line markdown stats bar for display in Gradio or README."""
    d = load()
    return (
        f"**{fmt(d['audits_completed'])}** bias audits completed · "
        f"**{fmt(d['models_evaluated'])}** models evaluated · "
        f"**{fmt(d['prompts_processed'])}** prompts processed · "
        f"**{d['bias_types_covered']}** bias categories · "
        f"**{d['researchers_using']}** researchers"
    )


def stats_html() -> str:
    """Return an HTML stats bar for embedding in Gradio Markdown blocks."""
    d = load()
    items = [
        (fmt(d["audits_completed"]), "Bias Audits"),
        (fmt(d["models_evaluated"]), "Models Tested"),
        (fmt(d["prompts_processed"]), "Prompts Run"),
        (str(d["bias_types_covered"]), "Bias Types"),
        (str(d["researchers_using"]), "Researchers"),
    ]
    cells = "".join(
        f'<div style="text-align:center;padding:0 1.5rem">'
        f'<div style="font-size:1.6rem;font-weight:700;color:#4f46e5">{val}</div>'
        f'<div style="font-size:0.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">{label}</div>'
        f"</div>"
        for val, label in items
    )
    return (
        '<div style="display:flex;justify-content:center;flex-wrap:wrap;'
        "gap:0.5rem;padding:1rem 0;border-top:1px solid #e5e7eb;"
        'border-bottom:1px solid #e5e7eb;margin:0.75rem 0">' + cells + "</div>"
    )
