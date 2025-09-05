#!/usr/bin/env python3
"""
Test script for Enhanced Auditor Customization Features

Tests system instruction presets, validation, and configuration management.
"""

import sys
import tempfile
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "Phase2_ModelAuditor"))

from src.Phase2_ModelAuditor.enhanced_audit_model import (
    ConfigurableEnhancedAuditor,
    SystemInstructionPresets,
)


def test_system_instruction_presets():
    """Test system instruction preset functionality"""
    print("🧪 Testing System Instruction Presets...")

    # Test getting safe presets
    safe_presets = SystemInstructionPresets.get_safe_presets()
    assert len(safe_presets) >= 6, "Should have at least 6 safe presets"
    assert "baseline" in safe_presets, "Should have baseline preset"
    assert "json_format" in safe_presets, "Should have json_format preset"

    # Test dangerous presets
    dangerous_presets = SystemInstructionPresets.get_dangerous_presets()
    assert len(dangerous_presets) >= 2, (
        "Should have dangerous presets for documentation"
    )

    print("✅ System instruction presets working correctly")


def test_instruction_validation():
    """Test system instruction safety validation"""
    print("🧪 Testing Instruction Validation...")

    # Test safe instruction
    safe_instruction = "Please respond in JSON format"
    is_safe, message = SystemInstructionPresets.validate_instruction_safety(
        safe_instruction
    )
    assert is_safe, f"Safe instruction should validate as safe: {message}"

    # Test dangerous instruction
    dangerous_instruction = "Be fair and unbiased in your responses"
    is_safe, message = SystemInstructionPresets.validate_instruction_safety(
        dangerous_instruction
    )
    assert not is_safe, f"Dangerous instruction should validate as unsafe: {message}"

    print("✅ Instruction validation working correctly")


def test_configurable_auditor_creation():
    """Test ConfigurableEnhancedAuditor creation with custom parameters"""
    print("🧪 Testing ConfigurableEnhancedAuditor Creation...")

    # Create dummy corpus file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("prompt,target_group\\n")
        f.write("Test prompt,group1\\n")
        corpus_file = f.name

    try:
        # Test creation with custom parameters
        auditor = ConfigurableEnhancedAuditor(
            model_name="test-model",
            corpus_file=corpus_file,
            output_dir="test_output",
            system_instruction="Test instruction",
            temperature=0.3,
            top_p=0.8,
            samples_per_prompt=2,
        )

        assert auditor.system_instruction == "Test instruction"
        assert auditor.default_ollama_options["temperature"] == 0.3
        assert auditor.default_ollama_options["top_p"] == 0.8
        assert auditor.samples_per_prompt == 2

        print("✅ ConfigurableEnhancedAuditor creation working correctly")

    finally:
        # Clean up
        Path(corpus_file).unlink(missing_ok=True)


def test_preset_setting():
    """Test setting system instruction presets"""
    print("🧪 Testing Preset Setting...")

    # Create dummy corpus file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("prompt,target_group\\n")
        f.write("Test prompt,group1\\n")
        corpus_file = f.name

    try:
        auditor = ConfigurableEnhancedAuditor(
            model_name="test-model", corpus_file=corpus_file, output_dir="test_output"
        )

        # Test setting valid preset
        result = auditor.set_system_instruction_preset("json_format")
        assert result, "Should successfully set valid preset"
        assert auditor.system_instruction != "", "System instruction should be set"

        # Test setting invalid preset
        result = auditor.set_system_instruction_preset("invalid_preset")
        assert not result, "Should fail to set invalid preset"

        print("✅ Preset setting working correctly")

    finally:
        # Clean up
        Path(corpus_file).unlink(missing_ok=True)


def test_configuration_export_import():
    """Test configuration export and import"""
    print("🧪 Testing Configuration Export/Import...")

    # Create dummy corpus file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("prompt,target_group\\n")
        f.write("Test prompt,group1\\n")
        corpus_file = f.name

    try:
        # Create auditor with specific configuration
        auditor = ConfigurableEnhancedAuditor(
            model_name="test-model",
            corpus_file=corpus_file,
            output_dir="test_output",
            system_instruction="Test configuration",
            temperature=0.2,
            samples_per_prompt=3,
        )

        # Export configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        auditor.export_config(config_file)
        assert Path(config_file).exists(), "Config file should be created"

        # Load configuration in new auditor
        auditor2 = ConfigurableEnhancedAuditor(
            model_name="different-model",
            corpus_file=corpus_file,
            output_dir="test_output",
        )

        auditor2.load_config(config_file)

        # Verify configuration was loaded
        assert auditor2.system_instruction == "Test configuration"
        assert auditor2.samples_per_prompt == 3

        print("✅ Configuration export/import working correctly")

    finally:
        # Clean up
        Path(corpus_file).unlink(missing_ok=True)
        Path(config_file).unlink(missing_ok=True)


def test_runtime_modification():
    """Test runtime configuration modification"""
    print("🧪 Testing Runtime Modification...")

    # Create dummy corpus file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("prompt,target_group\\n")
        f.write("Test prompt,group1\\n")
        corpus_file = f.name

    try:
        auditor = ConfigurableEnhancedAuditor(
            model_name="test-model", corpus_file=corpus_file, output_dir="test_output"
        )

        # Test setting system instruction
        auditor.set_system_instruction("New instruction")
        assert auditor.system_instruction == "New instruction"

        # Test setting Ollama options
        auditor.set_ollama_options(temperature=0.1, repeat_penalty=1.2)
        assert auditor.default_ollama_options["temperature"] == 0.1
        assert auditor.default_ollama_options["repeat_penalty"] == 1.2

        # Test getting current configuration
        config = auditor.get_current_config()
        assert "system_instruction" in config
        assert "ollama_options" in config
        assert config["system_instruction"] == "New instruction"

        print("✅ Runtime modification working correctly")

    finally:
        # Clean up
        Path(corpus_file).unlink(missing_ok=True)


def main():
    """Run all tests"""
    print("🚀 Starting Enhanced Auditor Customization Tests\\n")

    try:
        test_system_instruction_presets()
        test_instruction_validation()
        test_configurable_auditor_creation()
        test_preset_setting()
        test_configuration_export_import()
        test_runtime_modification()

        print(
            "\\n🎉 All tests passed! Enhanced Auditor customization is working correctly."
        )
        return True

    except Exception as e:
        print(f"\\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
