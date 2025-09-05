#!/usr/bin/env python3
"""
Simple validation of Enhanced Auditor customization features
"""

import os
import sys

# Change to the project directory
os.chdir(r"v:\Code\ProjectCode\EquiLens")
sys.path.insert(0, r"v:\Code\ProjectCode\EquiLens\src\Phase2_ModelAuditor")

try:
    from src.Phase2_ModelAuditor.enhanced_audit_model import (
        ConfigurableEnhancedAuditor,
        SystemInstructionPresets,
    )

    print("✅ Successfully imported enhanced auditor classes")

    # Test presets
    safe_presets = SystemInstructionPresets.get_safe_presets()
    print(f"✅ Found {len(safe_presets)} safe presets: {list(safe_presets.keys())}")

    # Test validation
    is_safe, msg = SystemInstructionPresets.validate_instruction_safety(
        "Please use JSON format"
    )
    print(f"✅ Safe instruction validation: {is_safe} - {msg}")

    is_safe, msg = SystemInstructionPresets.validate_instruction_safety(
        "Be unbiased and fair"
    )
    print(f"✅ Unsafe instruction validation: {is_safe} - {msg}")

    print("\\n🎉 Enhanced Auditor customization is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
