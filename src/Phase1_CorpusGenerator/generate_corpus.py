# Copyright 2025 Krishna GSVV
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def _preflight_check(base_dir: Path) -> None:
    """Run strict validator before generating corpus. Abort if validation fails."""
    validator = base_dir / "test_config.py"
    if not validator.exists():
        raise SystemExit(f"Validator not found: {validator}")

    print("Running configuration validator...")
    try:
        res = subprocess.run(["python", str(validator)], check=False)
        if res.returncode != 0:
            raise SystemExit(f"Validation failed (exit code {res.returncode})")
    except FileNotFoundError as exc:
        raise SystemExit("Python not found in PATH for validator execution") from exc


def interactive_select_comparison(config, base_dir):
    """Prompt user to confirm or select active comparison. Updates config and file if changed."""
    comparisons = config.get("comparisons", {})
    current = config.get("active_comparison", None)
    if not comparisons:
        print("No comparisons found in config. Aborting.")
        sys.exit(1)

    def print_comparison_info(comp_name):
        comp = comparisons[comp_name]
        print(f"\n🔍 {comp_name}")
        print(f"   Description: {comp.get('description', '')}")
        name_cats = [cat["category"] for cat in comp.get("name_categories", [])]
        trait_cats = [cat["category"] for cat in comp.get("trait_categories", [])]
        print(f"   Name categories: {' vs '.join(name_cats)}")
        print(f"   Trait categories: {' vs '.join(trait_cats)}")
        names = sum(len(cat["items"]) for cat in comp.get("name_categories", []))
        profs = len(comp.get("professions", []))
        traits = sum(len(cat["items"]) for cat in comp.get("trait_categories", []))
        templates = len(comp.get("templates", []))
        total = names * profs * traits * templates
        print(f"   Total combinations: {total:,}")

    # If current is valid, show and ask
    if current and current in comparisons:
        print(f"Current active comparison: {current}")
        print_comparison_info(current)
        ans = input("Continue with this comparison? [Y/n]: ").strip().lower()
        if ans in ("", "y", "yes"):
            return current
    # Otherwise, or if user said no, prompt for selection
    print("\nAvailable comparisons:")
    for i, comp_name in enumerate(comparisons.keys(), 1):
        print(f"  {i}. {comp_name}")
    while True:
        sel = input("Enter the number or name of the comparison to use: ").strip()
        if sel.isdigit():
            idx = int(sel) - 1
            if 0 <= idx < len(comparisons):
                chosen = list(comparisons.keys())[idx]
            else:
                print("Invalid selection. Try again.")
                continue
        else:
            chosen = sel
        if chosen in comparisons:
            print_comparison_info(chosen)
            confirm = (
                input(f"Use '{chosen}' as active comparison? [Y/n]: ").strip().lower()
            )
            if confirm in ("", "y", "yes"):
                # Update config and file
                config["active_comparison"] = chosen
                with open(base_dir / "word_lists.json", "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Switched to: {chosen}")
                return chosen
            else:
                continue
        else:
            print("Not a valid comparison name. Try again.")


def generate_corpus() -> None:
    """
    Generates a CSV file of templated sentences based on word lists in a JSON file.
    This script creates all possible combinations of names, professions, traits,
    and sentence templates defined in 'word_lists.json' for the active comparison.
    """
    # Resolve paths relative to this script for cross-platform reproducibility
    base_dir = Path(__file__).resolve().parent

    # Run preflight: strict validation
    _preflight_check(base_dir)

    word_lists_path = base_dir / "word_lists.json"
    # Load configuration
    print(f"Loading word lists from {word_lists_path}...")
    with open(word_lists_path, encoding="utf-8") as f:
        config = json.load(f)

    # Interactive selection of comparison
    interactive_select_comparison(config, base_dir)
    # Reload config in case it was updated
    with open(word_lists_path, encoding="utf-8") as f:
        config = json.load(f)
    active_comparison = config["active_comparison"]
    print(f"Active comparison: {active_comparison}")

    if active_comparison not in config["comparisons"]:
        print(f"Error: Active comparison '{active_comparison}' not found in configuration.")
        print(f"Available comparisons: {list(config['comparisons'].keys())}")
        return

    comparison_config = config["comparisons"][active_comparison]
    print(f"Description: {comparison_config['description']}")

    # Unpack the configuration for the active comparison
    name_categories = comparison_config["name_categories"]
    professions = comparison_config["professions"]
    trait_categories = comparison_config["trait_categories"]
    templates = comparison_config["templates"]

    # Create flat lists with category information for efficient iteration
    all_names = [
        (name, cat["category"]) for cat in name_categories for name in cat["items"]
    ]
    all_traits = [
        (trait, cat["category"]) for cat in trait_categories for trait in cat["items"]
    ]

    corpus_data = []

    # Use itertools.product for a memory-efficient way to create all combinations
    all_combinations = itertools.product(all_names, professions, all_traits, templates)

    # Calculate total number of combinations for the progress bar
    total_combinations = (
        len(all_names) * len(professions) * len(all_traits) * len(templates)
    )
    print(f"Generating {total_combinations} unique prompts...")

    pbar = tqdm(total=total_combinations, desc="Generating Corpus", unit="prompts")

    for (name, name_cat), profession, (trait, trait_cat), template in all_combinations:
        # If template is a tuple, use the first element as the template string
        template_str = template[0] if isinstance(template, tuple) else template
        prompt_text = template_str.format(name=name, profession=profession, trait=trait)
        corpus_data.append(
            {
                "comparison_type": active_comparison,
                "name": name,
                "name_category": name_cat,
                "profession": profession,
                "trait": trait,
                "trait_category": trait_cat,
                "template_id": templates.index(template),
                "full_prompt_text": prompt_text,
            }
        )
        pbar.update(1)

    pbar.close()

    df = pd.DataFrame(corpus_data)

    # Create corpus directory inside the package folder
    corpus_dir = base_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Create filename based on active comparison
    output_filename = corpus_dir / f"audit_corpus_{active_comparison}.csv"
    print(f"\nSaving corpus to {output_filename}...")
    # Use UTF-8 and newline='' via open() for universal CSV compatibility across platforms
    with open(output_filename, "w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)

    print(f"\nSuccessfully generated {output_filename} with {len(df)} rows.")
    print(f"Comparison type: {active_comparison}")
    print(f"Categories: {[cat['category'] for cat in name_categories]}")
    print(f"Trait categories: {[cat['category'] for cat in trait_categories]}")


if __name__ == "__main__":
    generate_corpus()
