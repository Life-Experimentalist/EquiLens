"""Phase2_ModelAuditor
======================

Module summary
--------------
This package implements the Phase 2 auditor used by EquiLens. It contains
the core auditor implementations that drive model evaluation runs against a
corpus of test prompts. The package provides both a stable, script-friendly
auditor (`audit_model.py`) and an experimental enhanced auditor
(`enhanced_audit_model.py`) that integrates a richer interactive UI.

See `README.md` in this directory for detailed usage, configuration options,
retry semantics, and troubleshooting tips.

Public API
----------
This package is primarily used by the EquiLens CLI. Typical integration is to
call the `main()` entrypoint in `audit_model.py` or instantiate the
`ModelAuditor` class exported by that module.

"""

from . import audit_model, enhanced_audit_model

__all__ = ["audit_model", "enhanced_audit_model"]
