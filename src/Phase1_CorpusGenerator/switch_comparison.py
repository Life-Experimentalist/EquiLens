#!/usr/bin/env python3
"""
Helper script to switch between different bias comparison types in EquiLens.
Usage: python switch_comparison.py [comparison_type]
"""

import json
import sys
import argparse


def list_available_comparisons():
    """List all available comparison types."""
    with open("word_lists.json", "r") as f:
        config = json.load(f)

    print("Available comparison types:")
    print("=" * 50)
    for comp_name, comp_data in config["comparisons"].items():
        print(f"🔍 {comp_name}")
        print(f"   Description: {comp_data['description']}")
        name_cats = [cat['category'] for cat in comp_data['name_categories']]
        trait_cats = [cat['category'] for cat in comp_data['trait_categories']]
        print(f"   Name categories: {' vs '.join(name_cats)}")
        print(f"   Trait categories: {' vs '.join(trait_cats)}")

        # Calculate total combinations
        names = sum(len(cat['items']) for cat in comp_data['name_categories'])
        profs = len(comp_data['professions'])
        traits = sum(len(cat['items']) for cat in comp_data['trait_categories'])
        templates = len(comp_data['templates'])
        total = names * profs * traits * templates
        print(f"   Total combinations: {total:,}")
        print()


def switch_comparison(comparison_type):
    """Switch to a specific comparison type."""
    with open("word_lists.json", "r") as f:
        config = json.load(f)

    if comparison_type not in config["comparisons"]:
        print(f"❌ Error: Comparison type '{comparison_type}' not found.")
        print(f"Available types: {list(config['comparisons'].keys())}")
        return False

    config["active_comparison"] = comparison_type

    with open("word_lists.json", "w") as f:
        json.dump(config, f, indent=2)

    comp_data = config["comparisons"][comparison_type]
    print(f"✅ Successfully switched to: {comparison_type}")
    print(f"   Description: {comp_data['description']}")
    name_cats = [cat['category'] for cat in comp_data['name_categories']]
    print(f"   Comparing: {' vs '.join(name_cats)}")
    return True


def get_current_comparison():
    """Get the currently active comparison type."""
    with open("word_lists.json", "r") as f:
        config = json.load(f)

    current = config["active_comparison"]
    comp_data = config["comparisons"][current]
    print(f"📊 Current active comparison: {current}")
    print(f"   Description: {comp_data['description']}")
    name_cats = [cat['category'] for cat in comp_data['name_categories']]
    trait_cats = [cat['category'] for cat in comp_data['trait_categories']]
    print(f"   Name categories: {' vs '.join(name_cats)}")
    print(f"   Trait categories: {' vs '.join(trait_cats)}")


def main():
    parser = argparse.ArgumentParser(
        description="Switch between different bias comparison types in EquiLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python switch_comparison.py                    # Show current and available types
  python switch_comparison.py gender_bias       # Switch to gender bias
  python switch_comparison.py nationality_bias  # Switch to nationality bias
  python switch_comparison.py cross_cultural_gender # Switch to cross-cultural gender
        """
    )
    parser.add_argument(
        "comparison_type",
        nargs="?",
        help="The comparison type to switch to (optional)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available comparison types"
    )

    args = parser.parse_args()

    if args.list:
        list_available_comparisons()
    elif args.comparison_type:
        if switch_comparison(args.comparison_type):
            print("\n🚀 You can now run: python generate_corpus.py")
    else:
        get_current_comparison()
        print("\n" + "=" * 50)
        list_available_comparisons()


if __name__ == "__main__":
    main()
