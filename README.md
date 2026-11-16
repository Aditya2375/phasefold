# Phasefold

An original learned stock-regime lab. Fit train-only standardization and two-means centroids on trailing mean/volatility features, freeze the model, then assign a chronological holdout without refitting. Native Python and browser JavaScript independently implement the same math. The static UI includes a feature scatter, assignment timeline, boundary inspector, frozen-parameter ledger and objective trace.

**Synthetic descriptive clustering, not a stock forecast or trading edge.** No actual stock observations, feed, paid data, portfolio, positions, trades, broker keys, returns strategy, PnL backtest or economic-state identification claim. Two colors are a modeling choice, not proof of two real regimes. This project actually learns its parameters, unlike a fixed threshold or supplied-parameter HMM, but that does not make its clusters predictive.

## Run

Python 3.10+ standard library only for the native core and CLI:

```sh
python3 -m phasefold --scenario alternating --seed 29 --window 20 --train-end 240
```

This emits a complete JSON result for an invented 420-observation stock log-return control. Invalid CLI parameters exit with status 2; it reads no files, writes no files and accepts no executable code. `--scenario` is alternating, shift, null or flat. The other flags are whole integers. See bounds below.

```python
from phasefold import analyze, generate

sample = generate(29, "alternating")
result = analyze(sample["returns"], window=20, train_end=240)
assert len(result["model"]["labels"]) == 220
assert sum(result["holdout_counts"]) == 180
```

For the browser:

```sh
python3 -m http.server 8000 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8000`. Inputs mark the old run stale and disable row inspection until **Fit & freeze** succeeds. Invalid input leaves the previous result visibly stale, never presented as the new run. The inspector's slider selects an exact boundary; charts, metrics, all-row ledger and parameters refer to the applied run. The page runs the independent JavaScript model locally, not a remote Python service. Changing a model after seeing holdout results is descriptive exploration, not untuned validation. Refresh/close loses everything; no upload, storage, cookies, telemetry or app network calls. Static hosting can log ordinary asset requests and source links navigate away.

## Information boundary

Observations `r[0]..r[n-1]` are daily synthetic **log** returns. At boundary `t`, available features are:

```text
mean[t] = sum(r[t-window : t]) / window
sd[t] = sqrt(sum((r[i]-mean[t])^2 for i in [t-window,t)) / (window-1))
```

Current return `r[t]` is excluded. The first `window` observations are warm-up; feature rows are t=window..n-1. There is intentionally no row at boundary n. Fit feature rows have t<train_end, holdout rows t>=train_end. All training features, scaler and centroids are fixed by the freeze boundary. The last fitting feature excludes return train_end-1; the first holdout feature includes it, which is already known at that boundary. No embargo is needed to prevent forward-outcome label leakage because there are no forward-outcome labels; **overlapping trailing windows still create dependence**, so rows are not independent validation trials.

Fit-period assignments are retrospective: their parameters use the entire training interval, including observations later than those early rows. They must not be portrayed as historically available signals. Only the frozen holdout assignments are prospective with respect to model fitting, and they are descriptions of past-window features, **not forecasts of return t or future returns**. Changing later returns cannot change earlier features, holdout assignments or fitted parameters. This is one fixed chronological split, not rolling cross-validation or expanding refitting.

## Exact model

1. Train-only feature means and **population** standard deviations (denominator N). A training feature SD <= 1e-12 uses scale 1 in raw units. The UI shows this fallback; it is not a one-basis-point scale. This stabilizes zero/near-zero variance but can alter the feature geometry. Standardized features are `(x-mean)/scale`. Rolling SD above instead uses denominator window-1.
2. Fixed K=2. Four deterministic first-center row indices: 0, floor(N/3), floor(2N/3), N-1. For each, the second is the most distant standardized row; distance ties choose the earlier row. This is **not k-means++**, random initialization or a global optimum solver.
3. Lloyd assignment to nearest squared-Euclidean centroid, center 0 on exact tie. Update each centroid to the coordinate mean of its assigned members. An empty cluster retains its previous center, rather than inventing a point or raising an error. Reassign after the update and record total nearest-center squared distance (SSE/inertia).
4. Stop when post-update assignments match the previous iteration, or after 100 updates. Return `converged=false` on the cap; no convergence is implied then. Choose the lowest final training SSE over four starts; exact SSE ties retain the earlier start. Floating-point near-ties can be platform-sensitive. The trace shows the winning restart only, after each update/reassignment, not all restarts or a probability curve.
5. Canonicalize cluster IDs by increasing fitted SD-axis centroid, then mean-axis centroid, then prior ID. The scaler is common with positive scales, so standardized sorting agrees with raw-coordinate ordering. Final C0/C1 are descriptive lower/higher volatility-centroid IDs, not guaranteed negative/positive drift, bull/bear, calm/crisis or economic states. Identical centers retain ID 0 tie behavior. A stored `tie_label` remaps the original winning center 0 into the final numbering; prediction uses that same ID on exact ties, preserving training memberships even when canonical IDs swap.
6. Freeze model/scaler and assign all rows. Holdout counts and switches cover only t>=train_end; switches count adjacent holdout label changes and exclude the last-fit/first-holdout transition. Model distance is squared nearest-center distance in train-standardized units, **not confidence, likelihood, posterior probability or calibrated OOD detection**.

The objective is descriptive fit compactness. Fixed K, feature choice, scaling, window, starts and training interval shape it. Euclidean k-means favors roughly compact isotropic clusters in the chosen coordinates; it can force a division of one continuous cloud. Four starts cannot ensure a global minimum. No silhouette optimization, statistical state-count selection, HMM transition model, confidence intervals, p-values, profit analysis or causal claim. Outliers and nearly constant training variance can dominate distances. No annualized performance metrics or risk guarantees.

### Synthetic controls

- `alternating`: generator state changes every 70 observations. State 0 mean +0.0006, SD 0.005; state 1 mean -0.0005, SD 0.018. Rolling windows mix states near changes, so learned labels are not generator truth.
- `shift`: quiet parameters until t=269, high-volatility parameters starting at t=270. At the default freeze boundary of 240 the shift is absent from fitting. A freeze boundary of 300 includes part of the shift in fitting; it is no longer unseen. A frozen classifier still forces far-away points into one of its old clusters.
- `null`: a single quiet generator state for all observations. The model may still produce two occupied clusters, which is not evidence for two regimes.
- `flat`: every return is 0, zero within-training variance, coincident centers. All assignments go to C0, and the UI reports one occupied fit cluster. No second regime is identified.

420 synthetic observations. The original reproducible generator uses `state=(1664525*state+1013904223) mod 2^32`, `u=(state+0.5)/2^32`, Box-Muller with two uniforms per draw and cosine. Return=mean+SD\*z, clipped to [-0.25, 0.25]. Clipping means distributions are bounded, not exact untruncated Gaussians. The seed is a teaching PRNG, not cryptography or a certified simulation source. Flat consumes no normal draws. Known generator labels are returned by generate for tests/background only, never used in fitting, canonical ordering or assignments; the UI displays no truth-agreement accuracy. No external data rights or empirical realism assumed.

## Library bounds

`generate(seed,scenario)` requires an integer seed from 1 to 4294967295; four scenario names above. `features(returns,window)` and `analyze(returns,window,train_end)` accept exact Python lists / ordinary JS arrays with 100 to 1000 finite numeric log returns, each in [-0.25, 0.25]. Booleans, sparse arrays, NaN, Infinity and numeric strings are rejected. Window is an integer from 5 to 60; train_end is an integer in [window+30, n-30], leaving at least 30 fit feature rows and 30 holdout rows. The UI uses 420 observations and offers windows of 5/20/40/60 and freeze boundaries of 180/240/300. Warm-up rows never enter counts.

`fit(points)` accepts 20 to 1000 two-dimensional finite numeric points bounded to an absolute value of 1e6. It returns detached means/scales/centers/labels/counts/trace. `predict(model,point)` expects an unmodified fitted model and two finite feature coordinates in its units; it is a low-level trusted helper, not an input validator. Do not mutate a model while predicting or pass hostile objects/proxies, subclasses or concurrent mutation. Core input lists/arrays are not mutated. Snapshot/output objects belong to the caller; changing them does not change the already-produced results elsewhere.

Worst-case fitting is four starts × 100 updates × 1000 points × 2 centers × 2 features, plus bounded feature calculation. There are no external callbacks, threads, user code execution, network, files, native pointers or persistence in the core. Host resource exhaustion and runtime failures are not transactionally isolated. CLI argument parsing uses argparse and JSON encoding uses finite values; it is not a shell, JSON input loader or streaming market interface.

## Verify

```sh
make test
npm ci --ignore-scripts
npx playwright install chromium
npm test
npm run test:browser
```

Native analytic tests include sample-SD feature fixtures, exact two-cloud SSE of 0, constant collapse, one constant dimension, tie behavior, translation/positive-scale equivariance, min/max data/split bounds, malformed/invalid CLI, input immutability and current/future-return invariance. 120 independent seeded point sets use Python statistics for train scaling and Euclidean distances for nearest assignments/SSE/monotone objectives. Browser-core tests compare 48 complete native/JS results across four controls, four seeds and three windows, including all rows/labels/centroids/scalers/traces/distances. Numerical parity supplements, never replaces independent fixtures. Tests tolerate floating arithmetic, not arbitrary label drift. The UI's charts use raw daily basis points; model distances are standardized. Matching axis pixels do not mean matching model distance.

Real Chromium tests at 1440/1024/768/390/320 px inspect actual chart pixels, model-derived metrics and counts, exact feature boundaries, retrospective-vs-holdout labels, stale/invalid input, seed bounds, all controls including degeneracy/null honesty, split changes, objective trace/all-row ledger, no page overflow/errors/failed or third-party requests/storage/cookies. Other browsers and assistive technology are not certified. Canvas chart information is also available in labeled inspector/parameters/all-row ledger. A 2,000-fit discrete-grid regression verifies canonical remapping preserves all fit labels and counts, including exact equal-distance points. Static analysis/finite tests are not proof of absence of bugs. CI pins actions, tests three Python versions, browser and secrets/no-network/DOM/native rules. Dev-only Playwright is locked; runtime has no packages.

## Background

Sources for concepts, not copied implementation or evidence of this lab's investment performance:

- [Scikit-learn k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means): objective, Lloyd updates, local minima and geometry limits.
- [Scikit-learn leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): train-only preprocessing and holdout separation.
- [Scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): train means/scales and zero-variance convention.
- [NIST measures of scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm): sample SD and outlier sensitivity.
- [Horvath, Issa, Muguruza (2021)](https://arxiv.org/abs/2110.11848): a different Wasserstein distribution-clustering method. This moment-feature lab does not reproduce that paper or inherit its reported results.

MIT. Original code and design by Aditya Kulkarni.
