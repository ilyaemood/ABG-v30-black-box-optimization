# Publishing Guide — GitHub and Kaggle

The sequence below is optimized to introduce the research idea and its strongest defensible evidence first, while keeping the evaluation protocol and caveats easy to find.

## A. Before posting

1. The author is listed as **Iliya Emam** in `PAPER.md` and `CITATION.cff`; no institution or affiliation is invented.
2. Review the paper's factual claims against `ABG_top8_benchmark/overall_ranking.csv`, `raw_results.csv`, and `paired_tests_vs_ABG.csv`.
3. Choose a license only after confirming that you have the rights to license the code, data, and included artifacts. No license is selected in this package; see `LICENSE_DECISION.md`.
4. After creating the GitHub repository, add its real URL as `repository-code` in `CITATION.cff`. Do not publish a placeholder URL.
5. Keep the raw data and all referenced figures in the repository. The article relies on relative paths, so do not rename images after publishing unless you update the Markdown links.

## B. Recommended GitHub repository order

Use the following repository root layout:

```text
README.md                         # First page visitors see
PAPER.md                          # Full English scientific-style article
KAGGLE_PAGE.md                    # Copy/paste Dataset description
PUBLISHING_GUIDE.md               # Publishing workflow
FIGURE_PLACEMENT_GUIDE.md         # Figure order and captions
CITATION.cff                      # Author metadata is set; repository URL can be added after repo creation
LICENSE                           # Add after choosing a license
requirements.txt                  # Runtime dependencies
benchmark_top8_20plus.py          # Benchmark runner
ABG_research_v30.py               # ABG-v30 implementation
ABG_top8_benchmark/               # Raw data, plots, report, dashboard
```

### Recommended first-view sequence

1. Put the **overall ranking chart** and headline numbers near the top of `README.md`.
2. Put the research abstract and detailed method in `PAPER.md`.
3. Link to the **33-function landscape contact sheet** after the benchmark scope paragraph.
4. Follow with the gap heatmap, category heatmap, and performance profile.
5. Present convergence plots and the noise stress test as evidence of behavior, not as decoration.
6. Place the paired statistical tests and limitations after the headline results, not in a hidden file.
7. Link the raw CSV and complete figure directory so the reader can audit the claims.

This ordering foregrounds what ABG contributes before evaluating how strong the evidence is. It does not hide a limitation or label a non-significant result as significant.

## C. GitHub steps

1. Create a public repository with a descriptive name, for example `abg-v30-black-box-optimization`.
2. Upload the contents of this package so `README.md`, `PAPER.md`, both Python files, and `ABG_top8_benchmark/` sit at the repository root (not nested inside an extra wrapper directory).
3. Confirm that GitHub renders the images in the README. Relative links such as `ABG_top8_benchmark/plots/01_overall_ranking.png` should work from the repository root.
4. Open the HTML dashboard from a local checkout, or use GitHub Pages only if you intentionally configure static hosting. The dashboard is a self-contained HTML artifact and does not need to be embedded as an image.
5. Commit the selected `LICENSE` and completed citation metadata only after reviewing them.
6. Make a versioned release after checking that the code and data match the article version. If a durable archival DOI is desired, connect the GitHub release to an archival service such as Zenodo after the account and repository are configured.

## D. Kaggle steps

1. Create a **Dataset** for the project; use the title and description in `KAGGLE_PAGE.md`.
2. Upload the archive or its extracted files. For notebook usability, keep `ABG_top8_benchmark/raw_results.csv`, all summary CSVs, and `plots/` available as ordinary files rather than making the notebook parse a large HTML dashboard for data.
3. Select the license that matches the one you have chosen for the project. Do not upload third-party code or figures without checking their licensing.
4. Create a separate Kaggle Notebook attached to the dataset. Follow the eight-section outline in `KAGGLE_PAGE.md` and correct the `/kaggle/input/...` slug in the first cell.
5. In the first markdown cell, summarize ABG-v30's design. In the second, show the headline ranking plot. Then validate row counts before reproducing any tables.
6. Distinguish **analysis of archived results** from **rerunning the optimizer**. If the notebook only loads the CSVs, say that clearly.
7. Include the paired-test caveat in the Kaggle description and notebook; it is part of the result interpretation.

## E. Which figures go where?

The exact numbered order is in `FIGURE_PLACEMENT_GUIDE.md`. `PAPER.md` already embeds the method schematics, protocol diagram, and result figures in that order. For README first-view, prefer the ABG workflow and overall ranking chart:

1. `plots/01_overall_ranking.png` — immediately after the headline findings.
2. `plots/landscape_contact_sheet.png` — after the function-suite description.
3. `plots/03_gap_heatmap.png` and `plots/04_category_rank_heatmap.png` — after the aggregate table.
4. `plots/07_performance_profile.png` — to qualify the rank table with a distributional view.
5. `plots/08_convergence_1.png`, `plots/10_convergence_3.png`, `plots/12_convergence_5.png` — in the convergence subsection.
6. `plots/13_noise_robustness.png` — immediately after the noisy-stress results.
7. `plots/06_paired_wins_losses.png` — beside the statistical-evidence discussion.
8. `plots/05_runtime_success.png` and `plots/14_budget_utilization.png` — in the efficiency section.

See the full captions in `PAPER.md`; do not remove captions or claim the figures prove more than their plotted metrics show.

## F. Final accuracy checklist

- [ ] Author name, date, version, and citation metadata are real; repository URL is added only after the repository exists.
- [ ] The repository owner has selected a license or consciously chosen to publish without one.
- [ ] The final `overall_ranking.csv` matches the values in `PAPER.md`.
- [ ] The paper says **300 maximum counted evaluations per run**, not exactly 300 calls in every run.
- [ ] The paper states that 0/693 comparisons were significant after global Holm correction.
- [ ] The paper states that this is not an official COCO/BBOB evaluation.
- [ ] Baselines are accurately described; the compact CMA-ES and fixed-rate Adam details are not omitted.
- [ ] All Markdown figure links point to files in the repository.
- [ ] The chosen code/data license is consistent across GitHub and Kaggle.
- [ ] The benchmark can be re-scored from shipped data, and the documented command is tested.
