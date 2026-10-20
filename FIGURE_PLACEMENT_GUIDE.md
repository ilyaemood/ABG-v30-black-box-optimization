# Figure Placement Guide

`PAPER.md` is already arranged in this order. Keep the images and their relative paths intact when uploading the repository.

## Main-paper figures

### Figure 1 — Intuition for adaptive box subdivision
**Section:** 2.1, Design overview  
**File:** `ABG_top8_benchmark/plots/adaptive_box_subdivision_schematic.png`  
**Caption:** The selected region is split along its widest coordinate; other active regions remain eligible. Schematic only, not an actual optimizer trace.

### Figure 2 — ABG-v30 workflow
**Section:** 2.5, after high-level pseudocode  
**File:** `ABG_top8_benchmark/plots/abg_v30_workflow.png`  
**Caption:** Initial design, priority-guided region search, anti-starvation exploration, periodic audit, optional validated surrogate, local refinement, balanced polling, and incumbent output.

### Figure 3 — Landscape atlas
**Section:** 3.1, after describing the function suite  
**File:** `ABG_top8_benchmark/plots/landscape_contact_sheet.png`  
**Caption:** Two-dimensional slices for 33 function families. The slices are illustrative; scores come from full-dimensional runs.

### Figure 4 — Evaluation protocol
**Section:** 3.6, after the statistical-analysis paragraph  
**File:** `ABG_top8_benchmark/plots/benchmark_protocol.png`  
**Caption:** Benchmark design, run records, normalized scoring, descriptive analysis, and paired inference.

### Figure 5 — Aggregate mean rank
**Section:** 4.1, after the opening results paragraph  
**File:** `ABG_top8_benchmark/plots/01_overall_ranking.png`  
**Caption:** Mean task rank across 87 deterministic tasks; lower is better.

### Figure 6 — Task-level median gap heatmap
**Section:** 4.2  
**File:** `ABG_top8_benchmark/plots/03_gap_heatmap.png`

### Figure 7 — Rank by landscape category
**Section:** 4.2, after Figure 6  
**File:** `ABG_top8_benchmark/plots/04_category_rank_heatmap.png`

### Figure 8 — Performance profile
**Section:** 4.3  
**File:** `ABG_top8_benchmark/plots/07_performance_profile.png`

### Figures 9a–9c — Convergence examples
**Section:** 4.3, after performance-profile discussion  
**Files:**
- `ABG_top8_benchmark/plots/08_convergence_1.png` — shifted Sphere, 2-D
- `ABG_top8_benchmark/plots/10_convergence_3.png` — Rastrigin, 2-D
- `ABG_top8_benchmark/plots/12_convergence_5.png` — rotated Ellipsoid, 30-D

### Figure 10 — Noise stress test
**Section:** 4.4  
**File:** `ABG_top8_benchmark/plots/13_noise_robustness.png`

### Figure 11 — Paired win/loss/tie counts
**Section:** 4.5  
**File:** `ABG_top8_benchmark/plots/06_paired_wins_losses.png`  
**Caption:** Descriptive paired outcomes only; these counts do not supersede the globally corrected tests.

### Figure 12 — Runtime versus success
**Section:** 4.6  
**File:** `ABG_top8_benchmark/plots/05_runtime_success.png`

### Figure 13 — Budget utilization
**Section:** 4.6, after Figure 12  
**File:** `ABG_top8_benchmark/plots/14_budget_utilization.png`

## Supplementary images

The supplementary chart `ABG_top8_benchmark/plots/15_quality_tradeoff.png` shows rank versus threshold success. All 33 individual landscape images live in `ABG_top8_benchmark/plots/landscapes/`; the mapping is recorded in `landscape_catalog.csv`. The SVG versions of the three newly added schematics are available alongside their PNG files for scalable use outside GitHub Markdown.

Do not alter axes, scales, result labels, or captions to make the optimizer look better. The strongest defensible presentation is to put the method and its broad descriptive result near the top, then show the protocol and statistical caveats where readers can easily find them.
