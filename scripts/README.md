# Experiment scripts

Run from the artifact root, containing `data/`, `scripts/`, and `outputs/`.
Examples use Qwen; replace `qwen` with `llama`, `codegemma`, `qwen_coder`, or `deepseek` for another model.
Requires the [datasets](../data/README.md), model checkpoint, and Python dependencies below; step 08 also needs the [evaluator setup](../evaluator/README.md).
Steps 01, 03, 04b, 07, and 17 use a GPU in these examples; other steps use the CPU.

Activate a Python 3.11 environment, then install the dependencies (the PyTorch build uses CUDA 12.8):

```bash
python -m pip install -r requirements.txt
```

All commands below use `python` from that environment. 

Results go to `outputs/<model>/`; plots go to its `figures/` folder, or `outputs/figures/` for multiple models.

## 01: Extract program representations

Uses `data/programs.jsonl` and the model config to mean-pool code-body token activations at every layer.

```bash
python scripts/01_extract_hidden_states.py --config scripts/config/qwen.json --device cuda
```

## 02: Construct repair vectors

After **01**, subtracts misuse from secure representations and averages the four structural variants of each original.

```bash
python scripts/02_construct_repair_vectors.py --input-dir outputs/qwen
```

## 03: Select a layer

After **01**, fits language/category-held-out linear probes and selects the layer with the highest joint AUROC.

```bash
python scripts/03_probe_layers.py --input-dir outputs/qwen --device cuda
```

## 04: Fit full-source directions

After **02 and 03**, fits Cell DiffMean, Pooled DiffMean, Normalized DiffMean, and CrypSteer unit directions.

```bash
python scripts/04_fit_directions.py --input-dir outputs/qwen
```

## 04a: Fit source-excluded directions

After **02 and 03**, fits CrypSteer with the target language, misuse category, or both excluded.

```bash
python scripts/04a_fit_source_excluded_directions.py --input-dir outputs/qwen
```

## 04b: Extract comparison activations

After **03**, extracts SCS-Code activations from `data/programs.jsonl` and Surgical Repair activations from `data/surgical_prompt_pairs.jsonl`.

```bash
python scripts/04b_extract_comparison_activations.py --config scripts/config/qwen.json --device cuda
```

## 04c: Fit comparison directions

After **04b**, constructs the SCS-Code and Surgical Repair unit directions.

```bash
python scripts/04c_fit_comparison_directions.py --input-dir outputs/qwen
```

## 05: Calibrate intervention strength

After **01 and 03**, computes sigma as the square root of total sample variance at the selected layer.

```bash
python scripts/05_calibrate_sigma.py --input-dir outputs/qwen
```

## 06: Analyze repair geometry

After **02 and 03**, calculates cell cosine similarities, repair-length ratios, and cross-language category retrieval.

```bash
python scripts/06_analyze_geometry.py --input-dir outputs/qwen
```

## 07: Generate programs

After **04, 04a, 04c, and 05**, generates programs from `data/generation_prompts.jsonl`.

```bash
python scripts/07_generate.py --config scripts/config/qwen.json --device cuda \
  --methods baseline cell pooled normalized crypsteer \
    crypsteer_exclude_language crypsteer_exclude_misuse crypsteer_exclude_both scs surgical --alpha 1

python scripts/07_generate.py --config scripts/config/qwen.json --device cuda \
  --methods cell pooled normalized crypsteer --alpha -1

for alpha in 0.5 2; do
  python scripts/07_generate.py --config scripts/config/qwen.json --device cuda \
    --methods crypsteer --alpha "$alpha"
done
```

Add `--resume` to continue interrupted generation.

## 08: Evaluate programs

After **07** and evaluator setup, checks target security and structural validity for every setting.

```bash
python scripts/08_evaluate_generations.py --input-dir outputs/qwen/generation
```

## 09: Summarize generation results

After **08**, reports secure@1, valid@1, and secure-valid@1 overall and by language, misuse category, and cell.

```bash
python scripts/09_summarize_results.py --input-dir outputs/qwen/evaluation
```

## 10: Compute paired statistics

After **08**, computes paired confidence intervals and Holm-adjusted sign-flip tests for the main and literature comparisons at +1 sigma.

```bash
python scripts/10_paired_statistics.py --input-dir outputs/qwen/evaluation
```

## 11: Test held-out repair alignment

After **02 and 03**, fits CrypSteer on 70% of originals and measures alignment on the remaining 30% at the fixed layer.

```bash
python scripts/11_test_heldout_alignment.py --input-dir outputs/qwen
```

## 12: Analyze structural variants

After **02 and 03**, measures within-original repair cosine/norm agreement and cross-structure category retrieval.

```bash
python scripts/12_analyze_structure.py --input-dir outputs/qwen
```

## 13: Measure category and language effects

After **02 and 03**, computes category and language shares of variation among the 40 unit cell directions.

```bash
python scripts/13_analyze_variance.py --input-dir outputs/qwen
```

## 14: Cluster cell directions

After **02 and 03**, clusters the 40 unit cell directions and reports category/language ARI.

```bash
python scripts/14_cluster_directions.py --input-dir outputs/qwen
```

## 15: Compare layers

After **02**, computes cross-layer RSA from the cell cosine matrices at all extracted layers.

```bash
python scripts/15_compare_layers.py --input-dir outputs/qwen
```

## 16: Analyze repair combinations

After **04** and **01 on `data/combinations.jsonl`**, fits additive responses and predicts 1,013 combinations per language from eleven reference programs.

```bash
python scripts/01_extract_hidden_states.py \
  --data data/combinations.jsonl --config scripts/config/qwen.json --device cuda
python scripts/16_analyze_combinations.py --input-dir outputs/qwen
```

## 17: Create code heatmaps

After **04**, extracts token activations for `data/heatmap_programs.jsonl`, aligns paired spans, and saves interactive HTML heatmaps with adjustable delta and color range.

```bash
python scripts/17_generate_code_heatmaps.py --config scripts/config/qwen.json --device cuda
```

## 18: Plot repair geometry

After **06**, plots the cell cosine and norm-ratio matrices as a PDF.

```bash
python scripts/18_plot_geometry.py --input-dirs outputs/qwen
```

## 19: Plot strength responses

After **09** for baseline and CrypSteer at -1, 0.5, 1, and 2 sigma, plots the three generation metrics as a PDF.

```bash
python scripts/19_plot_strength_response.py --input-dirs outputs/qwen
```

## 20: Plot structural agreement

After **12**, plots cosine and repair-norm histograms for one language–category cell as a PDF.

```bash
python scripts/20_plot_structure_agreement.py \
  --input-dir outputs/qwen --language typescript --misuse M02
```

## 21: Plot cross-layer RSA

After **15**, plots the cross-layer similarity matrix as a PDF.

```bash
python scripts/21_plot_layer_rsa.py --input-dirs outputs/qwen
```

## 22: Plot combination fits

After **16**, plots additive-model fitted values against measured projections for one language as a PDF.

```bash
python scripts/22_plot_combination_fit.py --input-dirs outputs/qwen --language java
```

## 23: Plot clusters

After **14**, visualizes saved cell directions and cluster labels with metric MDS as a PDF.

```bash
python scripts/23_plot_clusters.py --input-dirs outputs/qwen
```

For combined plots, scripts 18, 19, 21, 22, and 23 accept several model folders after `--input-dirs`; each model needs that plot's preceding results.

## 24: Export result tables

After **03, 04, 05, 06, 09, 10, 11, 13, 14, and 16**, exports all 13 experiment tables to CSV; the five-model example saves them in `outputs/tables/`.

```bash
python scripts/24_export_tables.py \
  --input-dirs outputs/llama outputs/codegemma outputs/qwen outputs/qwen_coder outputs/deepseek \
  --output-dir outputs/tables
```
