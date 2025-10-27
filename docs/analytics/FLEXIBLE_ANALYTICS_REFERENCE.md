# 📊 Flexible Analytics - Quick Reference

## Auto-Detected Fields

When you load a corpus, analytics automatically detects:

```python
self.comparison_type       # e.g., "gender_bias", "nationality_bias"
self.name_categories       # e.g., ["Female", "Male"] or ["Western", "Eastern"]
self.trait_categories      # e.g., ["Competence", "Social"]
self.grouping_fields       # e.g., ["profession", "trait", "trait_category"]
self.category_label_1      # First category name
self.category_label_2      # Second category name
```

## Example Detection Output

### Gender Bias Corpus
```
📋 Detected corpus structure:
   • Comparison: gender_bias
   • Categories: Female, Male
   • Trait types: Competence, Social
   • Grouping fields: profession, trait, trait_category, template_id
```

### Nationality Bias Corpus (Hypothetical)
```
📋 Detected corpus structure:
   • Comparison: nationality_bias
   • Categories: Eastern, Western
   • Trait types: Communication, Technical
   • Grouping fields: profession, trait, trait_category, template_id
```

## Flexible Visualizations

All charts now use dynamic labels:

| Chart | Before | After |
|-------|--------|-------|
| Violin Plot | "Male Names" / "Female Names" | f"{category_label_1} Names" / f"{category_label_2} Names" |
| Dashboard Title | "Overall Gender Distribution" | f"Overall {comparison_type} Distribution" |
| Effect Sizes | "Gender Bias" | f"{comparison_type.replace('_', ' ').title()}" |
| Statistical Tests | "overall_gender" | "overall_comparison" |

## Statistical Results Keys

Results now use dynamic keys based on detected categories:

```python
# Gender Bias Example
{
    "female_mean": 42.5,
    "male_mean": 38.2,
    "difference": 4.3
}

# Nationality Bias Example (Hypothetical)
{
    "western_mean": 40.1,
    "eastern_mean": 41.8,
    "difference": 1.7
}
```

## Adding New Comparison Types

### Step 1: Configure
Edit `word_lists.json`:
```json
{
  "active_comparison": "age_bias",
  "comparisons": {
    "age_bias": {
      "name_categories": [
        {"category": "Young", "items": ["Alex", "Jordan", "Sam"]},
        {"category": "Senior", "items": ["Robert", "Barbara", "William"]}
      ],
      "professions": ["engineer", "manager", "consultant"],
      "trait_categories": [
        {"category": "Energy", "items": ["energetic", "dynamic", "active"]},
        {"category": "Experience", "items": ["experienced", "wise", "knowledgeable"]}
      ],
      "templates": ["..."]
    }
  }
}
```

### Step 2: Generate Corpus
```bash
python src/Phase1_CorpusGenerator/generate_corpus.py
```

### Step 3: Audit
```bash
python src/equilens/cli.py audit --corpus-file corpus/audit_corpus_age_bias.csv --model llama2:latest
```

### Step 4: Analyze
```bash
python src/equilens/cli.py analyze --results results/<file>.csv --advanced
```

**Expected Output:**
```
📋 Detected corpus structure:
   • Comparison: age_bias
   • Categories: Senior, Young
   • Trait types: Energy, Experience
   • Grouping fields: profession, trait, trait_category, template_id
```

## CSV Structure

**Required Columns:**
- `surprisal_score` - The bias metric
- `name_category` - Category label (e.g., "Male", "Female")
- `trait_category` - Trait grouping (e.g., "Competence", "Social")

**Optional Columns:**
- `comparison_type` - Type of bias (e.g., "gender_bias")
- `profession` - Job category
- `trait` - Specific trait
- `template_id` - Template variant
- `name` - Actual name used
- `full_prompt_text` - Complete prompt

## API Usage

```python
from Phase3_Analysis.analytics import BiasAnalytics

# Initialize
analytics = BiasAnalytics("results/results_file.csv")

# Load and auto-detect structure
if analytics.load_and_validate_data():
    print(f"Comparison Type: {analytics.comparison_type}")
    print(f"Categories: {', '.join(analytics.name_categories)}")

    # Run analysis (automatically adapts)
    analytics.run_complete_analysis(mode="advanced")
```

## Troubleshooting

### Issue: Categories Not Detected
**Symptom:** Empty or single category
**Fix:** Ensure CSV has `name_category` column with at least 2 unique values

### Issue: Wrong Comparison Type
**Symptom:** Shows "bias_detection" instead of actual type
**Fix:** Add `comparison_type` column to CSV or ensure Phase1 generator includes it

### Issue: Visualizations Fail
**Symptom:** Charts missing or error messages
**Check:**
1. Are there at least 2 name categories?
2. Does `profession` column exist (for profession-based charts)?
3. Are there enough data points (>1 per category)?

## Performance Tips

### Standard Mode (Fast)
```bash
# No --advanced flag
python src/equilens/cli.py analyze --results <file>.csv
```
- ⏱️ ~5 seconds
- 📊 1-2 basic charts
- ✅ Good for quick checks

### Advanced Mode (Comprehensive)
```bash
python src/equilens/cli.py analyze --results <file>.csv --advanced
```
- ⏱️ ~15-20 seconds
- 📊 7 charts + dashboard
- 📝 HTML + Markdown reports
- 🤖 AI insights (with fallback)
- ✅ Good for presentations

## Validation

The analytics validates corpus structure at load:
```
✅ Loaded 40 valid results (100.0% clean)
📊 Model: llama2_latest
📁 Output directory: results\llama2_latest_20251016_011024

📋 Detected corpus structure:
   • Comparison: gender_bias
   • Categories: Female, Male
   • Trait types: Competence, Social
   • Grouping fields: profession, trait, trait_category, template_id
```

If validation fails:
- ❌ Missing required columns
- ❌ Invalid data types
- ❌ No valid surprisal scores

## Future Enhancements

1. **N-way comparisons** (3+ categories)
2. **Nested comparisons** (gender × profession)
3. **Custom chart types** per comparison
4. **Interactive filters** in HTML reports
5. **Comparison type templates** for AI prompts
