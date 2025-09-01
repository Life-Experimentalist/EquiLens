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
"""

Strict validator for `word_lists.json` used by CorpusGen.

This script enforces the expected structure and exits with non-zero status
on any validation failure so it can be used in CI or pre-flight checks.

Checks performed:
 - `active_comparison` exists and is a key within `comparisons`
 - `comparisons` is a dict mapping to config objects
 - each comparison has: description (str), name_categories (non-empty list of {category, items}), professions (non-empty list), trait_categories (non-empty list of {category, items}), templates (non-empty list)
 - names and trait items are non-empty strings and there are no empty categories
 - no duplicate names inside each name category

Usage: run from `src/Phase1_CorpusGenerator` and it will exit 1 on any error.
"""

import json
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def validate():
    base = Path(__file__).parent
    cfg_path = base / "word_lists.json"
    if not cfg_path.exists():
        fail(f"Configuration file not found: {cfg_path}")

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Failed to parse JSON: {e}")

    # Top-level keys
    if "active_comparison" not in data:
        fail("Missing top-level key: active_comparison")

    if "comparisons" not in data or not isinstance(data["comparisons"], dict):
        fail("Missing or invalid 'comparisons' mapping")

    active = data["active_comparison"]
    comparisons = data["comparisons"]

    if active not in comparisons:
        fail(f"active_comparison '{active}' not found in comparisons")

    if not comparisons:
        fail("'comparisons' is empty")

    for comp_name, comp in comparisons.items():
        if not isinstance(comp, dict):
            fail(f"Comparison {comp_name} is not an object")

        # required fields
        required = [
            "description",
            "name_categories",
            "professions",
            "trait_categories",
            "templates",
        ]
        for k in required:
            if k not in comp:
                fail(f"Comparison '{comp_name}' missing required key: {k}")

        # description
        if not isinstance(comp["description"], str) or not comp["description"].strip():
            fail(f"Comparison '{comp_name}' has invalid description")

        # name_categories
        nc = comp["name_categories"]
        if not isinstance(nc, list) or len(nc) == 0:
            fail(f"Comparison '{comp_name}' has empty or invalid 'name_categories'")

        for cat in nc:
            if not isinstance(cat, dict) or "category" not in cat or "items" not in cat:
                fail(f"Invalid name category entry in '{comp_name}': {cat}")
            if not isinstance(cat["items"], list) or len(cat["items"]) == 0:
                fail(
                    f"Empty 'items' in name category '{cat.get('category')}' of '{comp_name}'"
                )
            # check duplicates
            names = cat["items"]
            if len(set(names)) != len(names):
                fail(
                    f"Duplicate names detected in category '{cat.get('category')}' of '{comp_name}'"
                )

        # professions
        profs = comp["professions"]
        if not isinstance(profs, list) or len(profs) == 0:
            fail(f"Comparison '{comp_name}' has empty 'professions'")
        for p in profs:
            if not isinstance(p, str) or not p.strip():
                fail(f"Invalid profession entry in '{comp_name}': {p}")

        # trait_categories
        tc = comp["trait_categories"]
        if not isinstance(tc, list) or len(tc) == 0:
            fail(f"Comparison '{comp_name}' has empty or invalid 'trait_categories'")
        for cat in tc:
            if not isinstance(cat, dict) or "category" not in cat or "items" not in cat:
                fail(f"Invalid trait category entry in '{comp_name}': {cat}")
            if not isinstance(cat["items"], list) or len(cat["items"]) == 0:
                fail(
                    f"Empty 'items' in trait category '{cat.get('category')}' of '{comp_name}'"
                )

        # templates
        templates = comp["templates"]
        if not isinstance(templates, list) or len(templates) == 0:
            fail(f"Comparison '{comp_name}' has empty 'templates'")

    # If we reach here, everything looks ok
    print("✅ word_lists.json validation passed")
    return 0


if __name__ == "__main__":
    exit(validate())
