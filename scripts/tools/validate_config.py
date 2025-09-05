#!/usr/bin/env python3
"""
EquiLens Configuration Validator

This script validates word_lists.json against the schema and provides
detailed feedback about the configuration.
"""

import json
import os
import sys
from pathlib import Path


def validate_config():
    """Validate configuration against JSON schema if jsonschema is available."""
    try:
        import jsonschema
    except ImportError:
        print("⚠️  jsonschema package not installed. Install with: pip install jsonschema")
        return True  # Skip validation if not available

    # Load schema from src/Phase1_CorpusGenerator directory
    schema_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "Phase1_CorpusGenerator",
        "word_lists_schema.json",
    )
    if not os.path.exists(schema_path):
        print(f"❌ Error: Schema file not found at {schema_path}")
        return False

    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Load config
        config_path = (
            Path(__file__).parent.parent
            / "src"
            / "Phase1_CorpusGenerator"
            / "word_lists.json"
        )
        if not config_path.exists():
            print(
                "❌ Configuration file 'src/Phase1_CorpusGenerator/word_lists.json' not found."
            )
            return False

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate against schema
        jsonschema.validate(config, schema)
        print("✅ Configuration is valid according to the schema!")
        return True

    except jsonschema.ValidationError as e:
        print(f"❌ Schema validation failed:")
        print(f"   Error: {e.message}")
        if e.absolute_path:
            print(f"   Path: {' -> '.join(str(p) for p in e.absolute_path)}")
        return False
    except Exception as e:
        print(f"❌ Error during schema validation: {e}")
        return False

def validate_configuration():
    """Perform detailed validation of the configuration."""
    config_path = (
        Path(__file__).parent.parent
        / "src"
        / "Phase1_CorpusGenerator"
        / "word_lists.json"
    )

    if not config_path.exists():
        print(
            "❌ Configuration file not found at src/Phase1_CorpusGenerator/word_lists.json"
        )
        return False

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False

    print("🔍 Validating configuration structure...")

    # Check root structure
    if 'active_comparison' not in config:
        print("❌ Missing 'active_comparison' field")
        return False

    if 'comparisons' not in config:
        print("❌ Missing 'comparisons' field")
        return False

    active_comp = config['active_comparison']
    comparisons = config['comparisons']

    # Check active comparison exists
    if active_comp not in comparisons:
        print(f"❌ Active comparison '{active_comp}' not found in comparisons")
        print(f"   Available: {list(comparisons.keys())}")
        return False

    print(f"✅ Active comparison: {active_comp}")

    # Validate each comparison
    all_valid = True
    for comp_name, comp_config in comparisons.items():
        print(f"\n📋 Validating comparison: {comp_name}")

        # Check required fields
        required_fields = ['description', 'name_categories', 'professions', 'trait_categories', 'templates']
        for field in required_fields:
            if field not in comp_config:
                print(f"❌ Missing required field: {field}")
                all_valid = False
                continue

        # Validate name_categories
        if 'name_categories' in comp_config:
            name_cats = comp_config['name_categories']
            if len(name_cats) != 2:
                print(f"❌ name_categories must have exactly 2 categories, found {len(name_cats)}")
                all_valid = False
            else:
                for i, cat in enumerate(name_cats):
                    if 'category' not in cat or 'items' not in cat:
                        print(f"❌ name_categories[{i}] missing 'category' or 'items'")
                        all_valid = False
                    elif len(cat['items']) < 3:
                        print(f"⚠️  name_categories[{i}] '{cat['category']}' has only {len(cat['items'])} names (recommend 5+)")
                    else:
                        print(f"✅ name_categories[{i}] '{cat['category']}': {len(cat['items'])} names")

        # Validate trait_categories
        if 'trait_categories' in comp_config:
            trait_cats = comp_config['trait_categories']
            if len(trait_cats) != 2:
                print(f"❌ trait_categories must have exactly 2 categories, found {len(trait_cats)}")
                all_valid = False
            else:
                for i, cat in enumerate(trait_cats):
                    if 'category' not in cat or 'items' not in cat:
                        print(f"❌ trait_categories[{i}] missing 'category' or 'items'")
                        all_valid = False
                    elif len(cat['items']) < 3:
                        print(f"⚠️  trait_categories[{i}] '{cat['category']}' has only {len(cat['items'])} traits (recommend 5+)")
                    else:
                        print(f"✅ trait_categories[{i}] '{cat['category']}': {len(cat['items'])} traits")

        # Validate professions
        if 'professions' in comp_config:
            profs = comp_config['professions']
            if len(profs) < 3:
                print(f"⚠️  Only {len(profs)} professions (recommend 10+)")
            else:
                print(f"✅ professions: {len(profs)} items")

        # Validate templates
        if 'templates' in comp_config:
            templates = comp_config['templates']
            if len(templates) < 3:
                print(f"⚠️  Only {len(templates)} templates (recommend 5+)")
            else:
                print(f"✅ templates: {len(templates)} items")

            # Check template placeholders
            invalid_templates = []
            for i, template in enumerate(templates):
                if not all(placeholder in template for placeholder in ['{name}', '{profession}', '{trait}']):
                    invalid_templates.append(i)

            if invalid_templates:
                print(f"❌ Templates missing required placeholders: {invalid_templates}")
                print("   Required: {{name}}, {{profession}}, {{trait}}")
                all_valid = False

        # Calculate combinations
        if all(field in comp_config for field in required_fields):
            try:
                total_names = sum(len(cat['items']) for cat in comp_config['name_categories'])
                total_traits = sum(len(cat['items']) for cat in comp_config['trait_categories'])
                total_profs = len(comp_config['professions'])
                total_templates = len(comp_config['templates'])

                combinations = total_names * total_traits * total_profs * total_templates
                print(f"📊 Total combinations: {combinations:,}")

                if combinations > 10_000_000:
                    print(f"⚠️  Very large dataset ({combinations:,} combinations). Consider reducing scope for initial testing.")
                elif combinations < 1000:
                    print(f"⚠️  Small dataset ({combinations:,} combinations). Consider expanding for robust results.")

            except Exception as e:
                print(f"⚠️  Could not calculate combinations: {e}")

    return all_valid

def main():
    """Main validation function."""
    print("🔧 EquiLens Configuration Validator")
    print("=" * 40)

    # Schema validation
    schema_valid = validate_config()

    # Configuration validation
    config_valid = validate_configuration()

    print("\n" + "=" * 40)
    if schema_valid and config_valid:
        print("🎉 Configuration validation passed!")
        print("   Your word_lists.json is ready for use with EquiLens.")
        return 0
    else:
        print("💥 Configuration validation failed!")
        print("   Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
