# Submission TODO

**Deadline: 23:59, 1 April 2026**

## Written answers (`written_answers.pdf`)

- [ ] Create the written answers PDF (does not exist yet)
- [ ] Q1: State the mean rate of accidents (brief)
- [ ] Q2a: Qualitative comparison of plain vs even-numbered order statistics priors (<200 words) + include plot
- [ ] Q2b: Derivation showing even-numbered order statistics prior PDF is proportional to product of gaps
- [ ] Q3a: Observations about the prior/posterior plot + state the numerical evidence Z_0
- [ ] Q4a: Explanation of MCMC algorithm/package, settings, and convergence diagnostics (<300 words) + include corner plot
- [ ] Q4b: Savage–Dickey density ratio explanation and difficulty encountered (<200 words) + include plot
- [ ] Q4c: Difficulty with the Laplace approximation for Z_1 (<200 words)
- [ ] Q4d: Nested sampling algorithm details and settings (<200 words) + state evidence ratio and posterior odds ratio
- [ ] Q5b: State the MAP number of change points + include trace plot and posterior on k
- [ ] Q5c: Explanation of how uncertainty regions were calculated (<250 words) + include rate plot with 50%/90% shading
- [ ] Include all figures/plots alongside written answers
- [ ] Add AI declaration (use of autogeneration tools) to the written answers

## MCMC best practises

- [ ] Apply thinning to emcee chains (e.g. thin by autocorrelation time) before using posterior samples
- [ ] Compute Gelman Rubin statistic and comment on it's value
- [ ] Apply thinning/burn-in best practises to RJMCMC chain (Q5)
- [ ] Verify burn-in is sufficient for all samplers (emcee, RJMCMC)
- [ ] Check effective sample size is adequate after thinning

## Notebook cleanup

- [ ] Resolve in-notebook TODO markers:
  - Cell 1: general TODO note about thinning/best practices
  - Cell 10: "add observations about the even and plain order change points"
  - Cell 16: "ADD OBSERVATIONS ABOUT THE ABOVE PLOT"
- [ ] Clean up any commented-out debug code (e.g. Cell 33)
- [ ] Clear all saved cell outputs (Kernel > Restart & Clear All Output), then re-run to verify
- [ ] Ensure all cells run cleanly top-to-bottom (restart kernel and run all)

## Dead code removal

- [ ] Remove unused function `m1_log_likelihood()` in `src/coalmine/analysis.py` (never called anywhere)
- [ ] Remove unused function `plot_rate_history()` in `src/coalmine/plotting.py` (never called anywhere)
- [ ] Remove or use unused constants `K_MAX`, `LAMBDA_K`, `SEED` in `src/coalmine/constants.py`
- [ ] Remove any stale `print()` statements used only for debugging (e.g. `print(median)` in Cell 33)

## Spelling and typos

- [ ] Fix "becaused" -> "because" in `src/coalmine/RJMCMC.py` docstring (line ~30)
- [ ] Fix "OBSERVARTIONS" -> "OBSERVATIONS" in `coursework.ipynb` Cell 16

## Code quality

- [ ] Run `python -m pytest tests/` and ensure all tests pass
- [ ] Run linter (`ruff check .`) and fix any issues
- [ ] Run formatter (`ruff format --check .`) and fix any issues
- [ ] Review that plotting uses colourblind-friendly palettes throughout
- [ ] Fix naming convention violation: `logL` -> `log_l` in `RJMCMC.py` (line ~142)

## Type hints and docstrings

- [ ] Add type hints to `RJMCMC` class methods for consistency with rest of codebase
- [ ] Add docstring to `RJMCMC.main()`

## Git hygiene

- [ ] Add `.DS_Store` to `.gitignore`
- [ ] Verify no sensitive files (`.env`, credentials) are tracked

## Repository

- [ ] Add AI declaration to the README
- [ ] Verify `written_answers.pdf` is committed to the repository
- [ ] Final check that all required files are present: code, data, `written_answers.pdf`
- [ ] Push to GitLab before deadline
