# Artifact Audit — ABG-v30 Publication Package

**Audit date:** 19 October 2026  
**Purpose:** Validate consistency of the archived publication artifact before public release.

## Checks completed

| Check | Result |
|---|---:|
| Raw run-level records | 7,536 |
| Deterministic run records | 6,960 |
| Noisy-stress run records | 576 |
| Deterministic function–dimension tasks | 87 |
| Algorithm configurations in the archived comparison | 8 |
| Counted objective-evaluation budget | At most 300 per run |
| Runs exceeding the budget | 0 |
| Duplicate run keys | 0 |
| Run records with a recorded error | 0 |
| Paired test rows | 693 |
| Comparisons significant after global Holm correction at 0.05 | 0 |
| Landscape entries in the image catalog | 33 |

The leading archived metrics reproduce from the included CSV data: ABG-v30 mean task rank **2.2701**, mean success **74.48%** at normalized gap \(10^{-3}\), mean success **65.17%** at \(10^{-6}\), and top-three placement on **73/87** deterministic tasks.

## Executed validation

- `python ABG_research_v30.py --self-test` — **passed**. The test harness reported 38 function/dimension cases and 14 registered methods in the implementation module; this is an implementation smoke/invariant test, not the 8-configuration archived comparison.
- `python benchmark_top8_20plus.py --finalize-only --skip-landscapes` — **completed successfully**. This rebuilds summary tables, statistical comparisons, and aggregate plots from the archived raw CSVs. It does **not** rerun the 7,536 optimization runs.
- Markdown image/link paths and citation-file YAML syntax were checked locally; the three new diagrams have PNG and SVG versions.

## What this audit does not establish

This is a consistency and packaging audit, not independent scientific replication, peer review, or a fresh full benchmark rerun. It does not establish that ABG-v30 is statistically superior: no paired test survived the global Holm correction. The comparison is not an official COCO/BBOB run, the baseline settings are not exhaustively tuned, and the noisy subset is small. These limitations are documented in `PAPER.md` and should remain visible in public descriptions.

## Still requires the repository owner

- Add the actual GitHub repository URL to `CITATION.cff` after the repository exists.
- Choose a code/data license only after confirming the rights to license the included material. No `LICENSE` file has been added because the owner has not selected a license.
- Add the public release date to `CITATION.cff` only when the artifact is actually released.
