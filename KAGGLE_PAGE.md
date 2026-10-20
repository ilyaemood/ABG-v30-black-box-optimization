# Kaggle Publication Copy — ABG-v30 Benchmark

Use this file to populate the Kaggle Dataset title, subtitle, and description. The full scientific article is in [`PAPER.md`](PAPER.md).

## 1. Dataset title

**ABG-v30: Adaptive Box Subdivision Optimizer Benchmark**

## 2. Subtitle

**8 optimization methods · 33 function families · 87 deterministic tasks · paired raw results and figures**

## 3. Dataset description (copy/paste)

### Overview

This dataset accompanies an exploratory research report on **ABG-v30**, an adaptive box-subdivision approach for budgeted black-box optimization. The method combines priority-guided region subdivision with scheduled exploration, periodic region audits, an optional validated quadratic surrogate, bounded local refinement, and coordinate-balanced polling.

### Benchmark contents

The archive contains 7,536 run-level records from eight optimizer configurations: 6,960 deterministic runs across 87 function–dimension tasks and 576 runs in a selected additive-Gaussian-noise stress test. Each deterministic task uses ten paired trials and a hard cap of 300 counted objective evaluations per run. The function suite includes convex, ill-conditioned, shifted/rotated, narrow-valley, nonsmooth, and multimodal objectives, plus special 2-D and 4-D cases.

### Main reported result

On this suite, ABG-v30 obtained mean rank **2.270**, average success rates of **74.5%** at normalized gap ≤ 10⁻³ and **65.2%** at ≤ 10⁻⁶, and top-three placement on **73 of 87** deterministic tasks. In the selected noisy stress subset, the observed success rate was 41.7% at 1% relative noise and 27.8% at 5% relative noise under the 10⁻³ threshold.

### Important interpretation

These results are **exploratory, not proof of universal superiority**. None of 693 paired Wilcoxon tests remained significant after global Holm correction. The benchmark is not an official COCO/BBOB run, baseline hyperparameters were not comprehensively tuned, and the CMA-ES baseline is a compact implementation. Wall-clock timing is environment-dependent.

### Files

- `raw_results.csv`: one row per method/trial/task, including normalized gap, success flags, evaluation counts, timing, and algorithm metadata.
- `summary_by_function_dimension.csv`: per-task medians, IQRs, success rates, runtime, and failures.
- `overall_ranking.csv`: eight-method deterministic aggregate ranking.
- `paired_tests_vs_ABG.csv`: paired wins/losses/ties and globally Holm-adjusted p-values.
- `convergence_traces.csv`: best-so-far data for five representative tasks.
- `landscape_catalog.csv`: labels and paths for all 33 landscape plots.
- `plots/`: 15 comparison plots, a contact sheet, and 33 function landscape visualizations.
- `index.html`: standalone offline dashboard.
- `benchmark_top8_20plus.py` and `ABG_research_v30.py`: benchmark and optimizer source.
- `PAPER.md`: full article-style report and references.

## 4. Suggested Kaggle tags

`optimization`, `benchmark`, `algorithms`, `numerical-optimization`, `black-box-optimization`, `scientific-computing`, `reproducibility`

## 5. Suggested notebook outline

1. **Introduction and research question** — link `PAPER.md` and state that this is an exploratory benchmark.
2. **Load the data** — read `raw_results.csv` with pandas.
3. **Check data integrity** — verify row count, duplicates, failures, and budget violations.
4. **Aggregate deterministic results** — show `overall_ranking.csv` and explain the rank calculation.
5. **Visualize the result** — display the ranking chart, gap heatmap, performance profile, and contact sheet.
6. **Explore noise stress results** — compare success rates at 1% and 5% relative Gaussian noise.
7. **Statistical caution** — load `paired_tests_vs_ABG.csv`; show that no result remained significant after global Holm correction.
8. **Limitations and next steps** — baseline tuning, more trials, and official COCO/BBOB evaluation.

## 6. Example first notebook cell

```python
from pathlib import Path
import pandas as pd

DATA = Path('/kaggle/input/abg-v30-benchmark')  # Update to your Kaggle dataset slug
raw = pd.read_csv(DATA / 'ABG_top8_benchmark' / 'raw_results.csv')
ranking = pd.read_csv(DATA / 'ABG_top8_benchmark' / 'overall_ranking.csv')
print('Rows:', len(raw))
print('Methods:', raw['method'].nunique())
print('Deterministic rows:', (raw['noise_level'] == 0).sum())
print('Noisy rows:', (raw['noise_level'] > 0).sum())
print('Budget violations:', int((raw['evals'] > raw['budget']).sum()))
print('Duplicate result keys:', int(raw.duplicated(
    ['function', 'dimension', 'budget', 'noise_level', 'trial', 'method']
).sum()))
display(ranking)
```

**Note:** Kaggle mounts an uploaded dataset under a slug chosen on Kaggle. Update the `DATA` path to match the actual uploaded dataset directory. Do not claim that the Kaggle notebook reran the optimization unless it actually does so; loading the archived CSVs is data analysis, not a fresh benchmark run.
