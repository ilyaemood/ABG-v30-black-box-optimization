# ABG Top-8 Benchmark — final report

## Headline
- **Function families:** 33 total; **33** have deterministic scalable or special-function benchmark rows.
- **Deterministic function × dimension tasks:** 87.
- **Algorithms:** 8; **trials per task:** 10; **objective-evaluation budget:** 300 per run.
- **Noisy stress tests:** 6 selected families × relative noise levels (0.01, 0.05) × 6 trials, d=10.
- **Best mean-rank method in this run:** ABG-v30 (mean rank 2.270); this is a result for this exact suite, not a universal superiority claim.

## What the plots mean
- Each landscape image has two panels: a 2D domain slice with coordinates 3..d fixed at zero, and a slice through the known global minimizer with the remaining coordinates fixed at their optimal values. The color uses log10 of a dimensionless gap for readability; visual clipping never enters benchmark scores.
- Normalized gap is max(0, f(x_best)-f*) divided by a method-independent reference scale estimated from 256 Sobol points before optimization. The same scale is used for all methods on that function instance.
- Each algorithm has the same hard cap of 300 counted objective evaluations. Deterministic exact duplicate points use the ABG harness cache for every method; noisy evaluations are not cached. Post-run clean scoring and the reference design are audit/scoring operations and are excluded from the optimizer budget for all algorithms.
- All methods are bounded to normalized coordinates [-1,1]^d. DE, Dual Annealing, DIRECT, CMA-ES, local-search baselines, Adam, and ABG have different native internal operations; fairness is defined by the same objective-evaluation cap, not equal iterations or wall-clock.
- Adam uses the transparent fixed learning rate 0.03; it is not labelled tuned. Other default algorithm settings are disclosed in the script.
- Paired tests compare ABG with every competitor on the same trial instances. P-values are Holm-adjusted across all tested function/dimension/noise comparisons; significance is not inferred from average ranks alone.

## Overall ranking (deterministic suite)
| Rank | Algorithm | Mean rank | Median rank | Mean success ≤1e-3 (%) | Mean success ≤1e-6 (%) | Median seconds/run | Tasks |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | ABG-v30 | 2.270 | 1.50 | 74.5 | 65.2 | 0.0310 | 87 |
| 2 | Dual Annealing | 3.046 | 3.00 | 60.5 | 47.1 | 0.0128 | 87 |
| 3 | DIRECT | 3.057 | 2.00 | 62.0 | 52.4 | 0.0068 | 87 |
| 4 | CMA-ES (compact implementation) | 4.609 | 4.00 | 40.2 | 18.6 | 0.0106 | 87 |
| 5 | Sobol+L-BFGS-B | 5.144 | 5.00 | 42.2 | 27.1 | 0.0160 | 87 |
| 6 | Powell multi-start | 5.425 | 6.00 | 41.0 | 24.3 | 0.0131 | 87 |
| 7 | Differential Evolution | 6.057 | 6.00 | 41.6 | 21.1 | 0.0147 | 87 |
| 8 | Adam (finite-difference, lr=0.03) | 6.391 | 7.00 | 31.5 | 8.7 | 0.0071 | 87 |

## ABG paired-test summary after global Holm correction
| Competitor | ABG wins | ABG losses | ties | Significant task-level tests |
|---|---:|---:|---:|---:|
| DIRECT | 340 | 208 | 322 | 0 |
| CMA-ES (compact implementation) | 727 | 133 | 10 | 0 |
| Differential Evolution | 799 | 68 | 3 | 0 |
| Dual Annealing | 411 | 207 | 252 | 0 |
| Sobol+L-BFGS-B | 691 | 62 | 117 | 0 |
| Powell multi-start | 681 | 63 | 126 | 0 |
| Adam (finite-difference, lr=0.03) | 797 | 73 | 0 | 0 |

## Algorithm roster
| Algorithm | Role / method |
|---|---|
| ABG-v30 | User-designed adaptive box subdivision with optional quadratic surrogate and local refinement. |
| DIRECT | Deterministic partitioning of bounded domain. |
| CMA-ES (compact implementation) | Covariance-adaptation evolutionary search, implemented in the bundled harness. |
| Differential Evolution | Population-based differential mutation/recombination. |
| Dual Annealing | Stochastic global search with generalized annealing and optional local search. |
| Sobol+L-BFGS-B | Low-discrepancy start design plus bounded local refinements. |
| Powell multi-start | Derivative-free local search from diverse Sobol starts. |
| Adam (finite-difference, lr=0.03) | Adam-style update using finite-difference gradient estimates; fixed LR. |

## Function families and plot catalog
| # | Function | Category | Benchmark settings |
|---:|---|---|---|
| 1 | Sphere (convex) | Convex | [Landscape](plots/landscapes/01_sphere__convex.png) |
| 2 | Sphere (shifted convex) | Convex, shifted | [Landscape](plots/landscapes/02_sphere__shifted_convex.png) |
| 3 | Ellipsoid (ill-conditioned convex) | Convex, ill-conditioned | [Landscape](plots/landscapes/03_ellipsoid__ill_conditioned_convex.png) |
| 4 | Rotated Ellipsoid (convex) | Convex, rotated, ill-conditioned | [Landscape](plots/landscapes/04_rotated_ellipsoid__convex.png) |
| 5 | Bent Cigar (convex) | Convex, ill-conditioned | [Landscape](plots/landscapes/05_bent_cigar__convex.png) |
| 6 | Discus (convex) | Convex, ill-conditioned | [Landscape](plots/landscapes/06_discus__convex.png) |
| 7 | Zakharov (convex) | Convex | [Landscape](plots/landscapes/07_zakharov__convex.png) |
| 8 | Rosenbrock | Nonconvex, narrow valley | [Landscape](plots/landscapes/08_rosenbrock.png) |
| 9 | Rotated Rosenbrock | Nonconvex, rotated valley | [Landscape](plots/landscapes/09_rotated_rosenbrock.png) |
| 10 | Rastrigin | Multimodal | [Landscape](plots/landscapes/10_rastrigin.png) |
| 11 | Shifted Rastrigin | Multimodal, shifted | [Landscape](plots/landscapes/11_shifted_rastrigin.png) |
| 12 | Rotated Rastrigin | Multimodal, rotated and shifted | [Landscape](plots/landscapes/12_rotated_rastrigin.png) |
| 13 | Ackley | Multimodal, flat basin | [Landscape](plots/landscapes/13_ackley.png) |
| 14 | Griewank | Multimodal | [Landscape](plots/landscapes/14_griewank.png) |
| 15 | Schwefel | Multimodal, deceptive | [Landscape](plots/landscapes/15_schwefel.png) |
| 16 | Levy | Multimodal | [Landscape](plots/landscapes/16_levy.png) |
| 17 | Styblinski-Tang | Nonconvex, separable | [Landscape](plots/landscapes/17_styblinski_tang.png) |
| 18 | Dixon-Price | Nonconvex, coupled | [Landscape](plots/landscapes/18_dixon_price.png) |
| 19 | Alpine 1 | Nonsmooth, multimodal | [Landscape](plots/landscapes/19_alpine_1.png) |
| 20 | Shifted Ellipsoid (ill-conditioned convex) | Convex, shifted, ill-conditioned | [Landscape](plots/landscapes/20_shifted_ellipsoid__ill_conditioned_convex.png) |
| 21 | Shifted Rotated Ellipsoid (convex) | Convex, shifted, rotated, ill-conditioned | [Landscape](plots/landscapes/21_shifted_rotated_ellipsoid__convex.png) |
| 22 | Shifted Bent Cigar (convex) | Convex, shifted, ill-conditioned | [Landscape](plots/landscapes/22_shifted_bent_cigar__convex.png) |
| 23 | Shifted Discus (convex) | Convex, shifted, ill-conditioned | [Landscape](plots/landscapes/23_shifted_discus__convex.png) |
| 24 | Shifted Zakharov (convex) | Convex, shifted | [Landscape](plots/landscapes/24_shifted_zakharov__convex.png) |
| 25 | Shifted Ackley | Multimodal, shifted, flat basin | [Landscape](plots/landscapes/25_shifted_ackley.png) |
| 26 | Shifted Griewank | Multimodal, shifted | [Landscape](plots/landscapes/26_shifted_griewank.png) |
| 27 | Shifted Alpine 1 | Nonsmooth, shifted multimodal | [Landscape](plots/landscapes/27_shifted_alpine_1.png) |
| 28 | Beale (2D) | Multimodal, 2D-only | [Landscape](plots/landscapes/28_beale__2d.png) |
| 29 | Himmelblau (2D) | Multimodal, 2D-only | [Landscape](plots/landscapes/29_himmelblau__2d.png) |
| 30 | Michalewicz (2D) | Multimodal, 2D-only | [Landscape](plots/landscapes/30_michalewicz__2d.png) |
| 31 | Powell Singular (4D) | Nonconvex, ill-conditioned, 4D | [Landscape](plots/landscapes/31_powell_singular__4d.png) |
| 32 | Cross-in-Tray (2D) | Highly multimodal, 2D-only | [Landscape](plots/landscapes/32_cross_in_tray__2d.png) |
| 33 | Eggholder (2D) | Highly multimodal, 2D-only | [Landscape](plots/landscapes/33_eggholder__2d.png) |

## Limitations and interpretation
- Ten stochastic trials per deterministic task provide paired evidence, but this is still a finite benchmark rather than a universal proof. Several functions are classic analytic functions rather than the full official COCO/BBOB implementation with all prescribed transformations and noise models.
- For some special functions, the global-best reference values are literature best-known values, and the plotted 2D slice can be harder or easier than the full-dimensional search.
- Wall-clock timings are environment-dependent. The ranking is based on median dimensionless optimality gap, mean task rank, and threshold success; runtime is reported as a separate efficiency measure.
- Failure rows are retained with a worst-case gap penalty, not silently dropped.

## References
- COCO BBOB test-suite overview and function families: https://coco-platform.org/testsuites/bbob/overview.html
- SciPy Differential Evolution documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
- SciPy Dual Annealing documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html

## Files
- `raw_results.csv`: run-level outcomes and budget accounting.
- `summary_by_function_dimension.csv`: per-task medians, IQRs, success rates, runtime, failures.
- `overall_ranking.csv`: aggregate ranking and coverage.
- `paired_tests_vs_ABG.csv`: paired wins/losses and Holm-adjusted significance tests.
- `convergence_traces.csv`: best-so-far trajectories on five representative tasks.
- `plots/landscapes/`: 33 two-panel landscape/slice charts.
- `plots/`: summary charts and convergence profiles.
- `benchmark_top8_20plus.py`: executable, reproducible benchmark source.