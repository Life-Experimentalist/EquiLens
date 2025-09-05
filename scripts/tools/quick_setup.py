#!/usr/bin/env python3
"""
EquiLens Quick Setup

This script helps you create a new bias comparison configuration interactively.
Run this to generate a basic configuration that you can then customize.
"""

import json
import os
from pathlib import Path


def get_input(prompt, default=None, required=True):
    """Get user input with optional default and validation."""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()

        if user_input or not required:
            return user_input
        print("This field is required. Please enter a value.")

def get_list_input(prompt, min_items=3):
    """Get a list of items from user input."""
    print(f"\n{prompt}")
    print(f"Enter items one by one (minimum {min_items}). Press Enter with empty line to finish.")
    items = []
    while True:
        item = input(f"  Item {len(items)+1}: ").strip()
        if not item:
            if len(items) >= min_items:
                break
            else:
                print(f"Need at least {min_items} items. Continue adding...")
                continue
        items.append(item)
    return items

def create_bias_configuration():
    """Interactive configuration creator."""
    print("🔧 EquiLens Bias Configuration Creator")
    print("=" * 50)
    print("This will help you create a new bias comparison configuration.")
    print("You can always edit the generated file manually afterwards.\n")

    # Basic info
    comp_name = get_input("Comparison name (use snake_case, e.g., 'age_bias')")
    description = get_input("Description of what this comparison tests")

    print(f"\n📝 Creating configuration for: {comp_name}")
    print(f"Description: {description}")

    # Name categories (exactly 2)
    print(f"\n👥 Define the two opposing groups for comparison:")
    cat1_name = get_input("First group name (e.g., 'Male', 'Young', 'Western')")
    cat1_items = get_list_input(f"Names for {cat1_name} group", min_items=5)

    cat2_name = get_input("Second group name (e.g., 'Female', 'Elderly', 'Eastern')")
    cat2_items = get_list_input(f"Names for {cat2_name} group", min_items=5)

    # Professions
    professions = get_list_input("Professions to test bias in", min_items=5)

    # Trait categories (exactly 2)
    print(f"\n🏷️  Define the two types of traits to compare:")
    trait1_name = get_input("First trait category (e.g., 'Competence', 'Innovation')")
    trait1_items = get_list_input(f"Traits for {trait1_name} category", min_items=5)

    trait2_name = get_input("Second trait category (e.g., 'Social', 'Experience')")
    trait2_items = get_list_input(f"Traits for {trait2_name} category", min_items=5)

    # Templates
    print(f"\n📝 Create sentence templates:")
    print("Templates must include {name}, {profession}, and {trait} placeholders.")
    print("Examples:")
    print("  - {name}, the {profession}, is known for being very {trait}.")
    print("  - As a {profession}, {name} consistently demonstrates {trait} qualities.")

    templates = []
    while len(templates) < 3:
        template = get_input(f"Template {len(templates)+1}")
        if all(placeholder in template for placeholder in ['{name}', '{profession}', '{trait}']):
            templates.append(template)
        else:
            print("❌ Template must include {name}, {profession}, and {trait} placeholders!")

    # Additional templates (optional)
    while True:
        template = get_input(f"Template {len(templates)+1} (optional)", required=False)
        if not template:
            break
        if all(placeholder in template for placeholder in ['{name}', '{profession}', '{trait}']):
            templates.append(template)
        else:
            print("❌ Template must include {name}, {profession}, and {trait} placeholders!")

    # Build configuration
    config = {
        "active_comparison": comp_name,
        "comparisons": {
            comp_name: {
                "description": description,
                "name_categories": [
                    {"category": cat1_name, "items": cat1_items},
                    {"category": cat2_name, "items": cat2_items}
                ],
                "professions": professions,
                "trait_categories": [
                    {"category": trait1_name, "items": trait1_items},
                    {"category": trait2_name, "items": trait2_items}
                ],
                "templates": templates
            }
        }
    }

    # Calculate combinations
    total_names = len(cat1_items) + len(cat2_items)
    total_traits = len(trait1_items) + len(trait2_items)
    combinations = total_names * len(professions) * total_traits * len(templates)

    print(f"\n📊 Configuration Summary:")
    print(f"   Comparison: {comp_name}")
    print(f"   Groups: {cat1_name} ({len(cat1_items)} names) vs {cat2_name} ({len(cat2_items)} names)")
    print(f"   Professions: {len(professions)} items")
    print(f"   Trait categories: {trait1_name} ({len(trait1_items)}) vs {trait2_name} ({len(trait2_items)})")
    print(f"   Templates: {len(templates)} items")
    print(f"   Total combinations: {combinations:,}")

    if combinations > 1_000_000:
        print(f"   ⚠️  Large dataset - consider reducing scope for initial testing")

    # Save configuration
    config_dir = Path(__file__).parent.parent / "src" / "Phase1_CorpusGenerator"
    if not config_dir.exists():
        config_dir.mkdir(parents=True)

    config_path = config_dir / "word_lists.json"

    if config_path.exists():
        overwrite = get_input(f"\nConfiguration file already exists. Overwrite? (y/n)", "n")
        if overwrite.lower() != 'y':
            print("❌ Configuration not saved.")
            return False

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Configuration saved to {config_path}")
    print(f"\nNext steps:")
    print(f"1. Review and edit {config_path} if needed")
    print(f"2. Run: python tools/validate_config.py")
    print(
        f"3. Generate corpus: cd src/Phase1_CorpusGenerator && python generate_corpus.py"
    )
    print(f"4. See docs/CONFIGURATION_GUIDE.md for detailed guidance")

    return True

def main():
    """Main setup function."""
    try:
        create_bias_configuration()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
