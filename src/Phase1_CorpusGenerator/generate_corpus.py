import itertools
import json
import os

import pandas as pd
from tqdm import tqdm


def generate_corpus():
    """
    Generates a CSV file of templated sentences based on word lists in a JSON file.
    This script creates all possible combinations of names, professions, traits,
    and sentence templates defined in 'word_lists.json' for the active comparison.
    """
    print("Loading word lists from word_lists.json...")
    with open("word_lists.json") as f:
        config = json.load(f)

    # Get the active comparison configuration
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

    # Create corpus directory if it doesn't exist
    corpus_dir = "corpus"
    os.makedirs(corpus_dir, exist_ok=True)

    # Create filename based on active comparison
    output_filename = f"corpus/audit_corpus_{active_comparison}.csv"
    print(f"\nSaving corpus to {output_filename}...")
    df.to_csv(output_filename, index=False)

    print(f"\nSuccessfully generated {output_filename} with {len(df)} rows.")
    print(f"Comparison type: {active_comparison}")
    print(f"Categories: {[cat['category'] for cat in name_categories]}")
    print(f"Trait categories: {[cat['category'] for cat in trait_categories]}")


if __name__ == "__main__":
    generate_corpus()
