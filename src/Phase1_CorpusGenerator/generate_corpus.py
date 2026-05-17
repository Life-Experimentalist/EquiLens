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
    """Prompt user to select comparison(s) to generate. Returns list of comparison names."""
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

    print("\n" + "=" * 60)
    print("📝 CORPUS GENERATION - SELECT COMPARISON(S)")
    print("=" * 60)

    # Show current active comparison
    if current and current in comparisons:
        print(f"\n✓ Current active comparison: {current}")

    # Show all available comparisons
    print("\nAvailable comparisons:")
    print("  0. [ALL] Generate all comparisons")
    for i, comp_name in enumerate(comparisons.keys(), 1):
        comp = comparisons[comp_name]
        names = sum(len(cat["items"]) for cat in comp.get("name_categories", []))
        profs = len(comp.get("professions", []))
        traits = sum(len(cat["items"]) for cat in comp.get("trait_categories", []))
        templates = len(comp.get("templates", []))
        total = names * profs * traits * templates
        print(f"  {i}. {comp_name} ({total:,} prompts)")

    while True:
        try:
            sel = input(
                "\n👉 Enter number(s) [0 for all, 'q' to quit, or 1-N]: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  Generation cancelled by user.")
            sys.exit(0)

        # Handle quit/exit
        if sel.lower() in ("q", "quit", "exit"):
            print("\n⚠️  Generation cancelled by user.")
            sys.exit(0)

        # Handle "all" option
        if sel == "0":
            print("\n✓ Selected: ALL comparisons")
            total_prompts = sum(
                sum(len(cat["items"]) for cat in comp.get("name_categories", []))
                * len(comp.get("professions", []))
                * sum(len(cat["items"]) for cat in comp.get("trait_categories", []))
                * len(comp.get("templates", []))
                for comp in comparisons.values()
            )
            print(f"   Total prompts across all comparisons: {total_prompts:,}")
            try:
                confirm = input("   Generate ALL? [Y/n/q]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Generation cancelled by user.")
                sys.exit(0)

            if confirm in ("q", "quit", "exit"):
                print("\n⚠️  Generation cancelled by user.")
                sys.exit(0)
            if confirm in ("", "y", "yes"):
                return list(comparisons.keys())
            continue

        # Handle single selection
        if sel.isdigit():
            idx = int(sel) - 1
            if 0 <= idx < len(comparisons):
                chosen = list(comparisons.keys())[idx]
                print_comparison_info(chosen)
                try:
                    confirm = input(f"Generate '{chosen}'? [Y/n/q]: ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n\n⚠️  Generation cancelled by user.")
                    sys.exit(0)

                if confirm in ("q", "quit", "exit"):
                    print("\n⚠️  Generation cancelled by user.")
                    sys.exit(0)
                if confirm in ("", "y", "yes"):
                    # Update active_comparison in config file
                    config["active_comparison"] = chosen
                    with (base_dir / "word_lists.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(config, f, indent=2)
                    print(f"✅ Updated active comparison to: {chosen}")
                    return [chosen]
                continue
            else:
                print("❌ Invalid number. Try again.")
                continue

        # Handle by name
        if sel in comparisons:
            print_comparison_info(sel)
            try:
                confirm = input(f"Generate '{sel}'? [Y/n/q]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Generation cancelled by user.")
                sys.exit(0)

            if confirm in ("q", "quit", "exit"):
                print("\n⚠️  Generation cancelled by user.")
                sys.exit(0)
            if confirm in ("", "y", "yes"):
                # Update active_comparison in config file
                config["active_comparison"] = sel
                with (base_dir / "word_lists.json").open("w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
                print(f"✅ Updated active comparison to: {sel}")
                return [sel]
            continue

        print("❌ Invalid selection. Try again.")


def generate_single_corpus(config, comparison_name, base_dir):
    """Generate corpus for a single comparison."""
    if comparison_name not in config["comparisons"]:
        print(f"Error: Comparison '{comparison_name}' not found in configuration.")
        return False

    comparison_config = config["comparisons"][comparison_name]

    print(f"\n{'=' * 60}")
    print(f"Generating: {comparison_name}")
    print(f"Description: {comparison_config['description']}")
    print(f"{'=' * 60}")

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

    pbar = tqdm(total=total_combinations, desc=f"  {comparison_name}", unit="prompts")

    try:
        for (name, name_cat), profession, (
            trait,
            trait_cat,
        ), template in all_combinations:
            # If template is a tuple, use the first element as the template string
            template_str = template[0] if isinstance(template, tuple) else template
            prompt_text = template_str.format(
                name=name, profession=profession, trait=trait
            )
            corpus_data.append(
                {
                    "comparison_type": comparison_name,
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
    except KeyboardInterrupt:
        pbar.close()
        print("\n\n⚠️  Generation interrupted by user. Partial data discarded.")
        return False

    pbar.close()

    df = pd.DataFrame(corpus_data)

    # Create corpus directory inside the package folder
    corpus_dir = base_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Create filename based on comparison name
    output_filename = corpus_dir / f"audit_corpus_{comparison_name}.csv"
    # Use UTF-8 and newline='' via open() for universal CSV compatibility across platforms
    with output_filename.open("w", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)

    print(f"✅ Generated: {output_filename.name} ({len(df):,} rows)")
    return True


def generate_corpus() -> None:
    """
    Generates CSV file(s) of templated sentences based on word lists in a JSON file.
    This script creates all possible combinations of names, professions, traits,
    and sentence templates defined in 'word_lists.json' for selected comparison(s).
    """
    try:
        # Resolve paths relative to this script for cross-platform reproducibility
        base_dir = Path(__file__).resolve().parent

        # Run preflight: strict validation
        _preflight_check(base_dir)

        word_lists_path = base_dir / "word_lists.json"
        # Load configuration
        print(f"Loading word lists from {word_lists_path}...")
        with word_lists_path.open(encoding="utf-8") as f:
            config = json.load(f)

        # Interactive selection of comparison(s) - returns list
        selected_comparisons = interactive_select_comparison(config, base_dir)

        # Reload config in case it was updated
        with word_lists_path.open(encoding="utf-8") as f:
            config = json.load(f)

        # Generate corpus for each selected comparison
        print(f"\n{'=' * 60}")
        print(f"GENERATING {len(selected_comparisons)} CORPUS FILE(S)")
        print(f"{'=' * 60}")

        success_count = 0
        for comparison_name in selected_comparisons:
            if generate_single_corpus(config, comparison_name, base_dir):
                success_count += 1

        print(f"\n{'=' * 60}")
        print("✅ GENERATION COMPLETE")
        print(f"{'=' * 60}")
        print(
            f"Successfully generated {success_count}/{len(selected_comparisons)} corpus file(s)"
        )
        print(f"Location: {base_dir / 'corpus'}")

        # Show summary of generated files
        corpus_dir = base_dir / "corpus"
        if corpus_dir.exists():
            csv_files = sorted(corpus_dir.glob("audit_corpus_*.csv"))
            if csv_files:
                print("\nAvailable corpus files:")
                for csv_file in csv_files:
                    size = csv_file.stat().st_size
                    size_mb = size / (1024 * 1024)
                    print(f"  • {csv_file.name} ({size_mb:.2f} MB)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error during generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        generate_corpus()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation cancelled by user.")
        sys.exit(0)
