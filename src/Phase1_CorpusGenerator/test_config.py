#!/usr/bin/env python3
"""Simple validation test for the word_lists.json configuration."""

import json
import sys
import os

def test_config():
    """Test the configuration file for basic validity."""
    try:
        # Load configuration
        config_path = "word_lists.json"
        if not os.path.exists(config_path):
            print("❌ Configuration file not found")
            return False

        with open(config_path, 'r') as f:
            config = json.load(f)

        print("✅ Configuration loaded successfully")

        # Basic structure checks
        active = config.get('active_comparison')
        comparisons = config.get('comparisons', {})

        print(f"✅ Active comparison: {active}")
        print(f"✅ Available comparisons: {list(comparisons.keys())}")

        # Check for duplicates in gender_bias
        if 'gender_bias' in comparisons:
            gender_config = comparisons['gender_bias']
            male_names = gender_config['name_categories'][0]['items']
            female_names = gender_config['name_categories'][1]['items']

            male_unique = len(set(male_names)) == len(male_names)
            female_unique = len(set(female_names)) == len(female_names)

            print(f"✅ Male names unique: {male_unique} ({len(male_names)} total)")
            print(f"✅ Female names unique: {female_unique} ({len(female_names)} total)")

            if male_unique and female_unique:
                print("✅ No duplicate names found")
            else:
                print("❌ Duplicate names found")
                return False

        print("✅ All tests passed!")
        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON syntax error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
