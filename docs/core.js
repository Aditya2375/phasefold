// Original train-only two-means lab. No runtime data feed or trade execution.
function integer(v, lo, hi, label) {
  if (!Number.isInteger(v) || v < lo || v > hi)
    throw Error(`${label} must be an integer in ${lo}..${hi}`);
  return v;
}
const sum = (a) => a.reduce((s, v) => s + v, 0),
  dist = (a, b) => sum(a.map((v, j) => (v - b[j]) ** 2));
const label = (p, c, tie = 0) => {
  const a = dist(p, c[0]),
    b = dist(p, c[1]);
  return a === b ? tie : a < b ? 0 : 1;
};
export function features(returns, window) {
  integer(window, 5, 60, "window");
  if (!Array.isArray(returns) || returns.length < 100 || returns.length > 1000)
    throw Error("Need 100..1000 log returns");
  if (
    Array.from(returns).some(
      (v) => typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 0.25,
    )
  )
    throw Error("Log returns must be finite numbers within -0.25..0.25");
  const out = [];
  for (let t = window; t < returns.length; t++) {
    const data = returns.slice(t - window, t),
      mean = sum(data) / window,
      sd = Math.sqrt(sum(data.map((v) => (v - mean) ** 2)) / (window - 1));
    out.push({ t, mean, sd });
  }
  return out;
}
export function fit(points) {
  if (
    !Array.isArray(points) ||
    points.length < 20 ||
    points.length > 1000 ||
    Array.from(points).some(
      (p) =>
        !Array.isArray(p) ||
        p.length !== 2 ||
        Array.from(p).some(
          (v) =>
            typeof v !== "number" || !Number.isFinite(v) || Math.abs(v) > 1e6,
        ),
    )
  )
    throw Error("Need 20..1000 finite two-dimensional bounded points");
  const means = [0, 1].map((j) => sum(points.map((p) => p[j])) / points.length),
    scales = [0, 1]
      .map((j) =>
        Math.sqrt(
          sum(points.map((p) => (p[j] - means[j]) ** 2)) / points.length,
        ),
      )
      .map((v) => (v > 1e-12 ? v : 1)),
    scaled = points.map((p) => p.map((v, j) => (v - means[j]) / scales[j]));
  let best = null;
  for (const start of [
    0,
    Math.floor(points.length / 3),
    Math.floor((2 * points.length) / 3),
    points.length - 1,
  ]) {
    const first = [...scaled[start]];
    let farthest = 0;
    for (let i = 1; i < scaled.length; i++)
      if (dist(scaled[i], first) > dist(scaled[farthest], first)) farthest = i;
    let centers = [first, [...scaled[farthest]]],
      previous = null,
      converged = false,
      labels = [],
      objective = 0;
    const trace = [];
    for (let iteration = 0; iteration < 100; iteration++) {
      labels = scaled.map((p) => label(p, centers));
      centers = [0, 1].map((k) => {
        const members = scaled.filter((_, i) => labels[i] === k);
        return members.length
          ? [0, 1].map((j) => sum(members.map((p) => p[j])) / members.length)
          : [...centers[k]];
      });
      labels = scaled.map((p) => label(p, centers));
      objective = sum(scaled.map((p, i) => dist(p, centers[labels[i]])));
      trace.push(objective);
      if (previous && labels.every((v, i) => v === previous[i])) {
        converged = true;
        break;
      }
      previous = labels;
    }
    const counts = [0, 1].map((k) => labels.filter((v) => v === k).length),
      candidate = {
        centers,
        labels,
        counts,
        inertia: objective,
        trace,
        iterations: trace.length,
        converged,
        start,
      };
    if (best === null || objective < best.inertia) best = candidate;
  }
  const order = [0, 1].sort(
      (a, b) =>
        best.centers[a][1] - best.centers[b][1] ||
        best.centers[a][0] - best.centers[b][0] ||
        a - b,
    ),
    mapping = [];
  order.forEach((old, i) => (mapping[old] = i));
  return {
    means,
    scales,
    tie_label: mapping[0],
    ...best,
    centers: order.map((k) => best.centers[k]),
    counts: order.map((k) => best.counts[k]),
    labels: best.labels.map((v) => mapping[v]),
  };
}
export function predict(model, point) {
  return label(
    point.map((v, j) => (v - model.means[j]) / model.scales[j]),
    model.centers,
    model.tie_label,
  );
}
export function analyze(returns, window = 20, train_end = 240) {
  const rows = features(returns, window);
  integer(train_end, window + 30, returns.length - 30, "training boundary");
  const training = rows.filter((r) => r.t < train_end),
    model = fit(training.map((r) => [r.mean, r.sd]));
  for (const row of rows) {
    row.label = predict(model, [row.mean, row.sd]);
    row.split = row.t < train_end ? "fit" : "holdout";
    const p = [
      (row.mean - model.means[0]) / model.scales[0],
      (row.sd - model.means[1]) / model.scales[1],
    ];
    row.distance = Math.min(...model.centers.map((c) => dist(p, c)));
  }
  const hold = rows.filter((r) => r.split === "holdout");
  return {
    window,
    train_end,
    model,
    rows,
    holdout_counts: [0, 1].map((k) => hold.filter((r) => r.label === k).length),
    holdout_switches: hold.slice(1).filter((r, i) => r.label !== hold[i].label)
      .length,
  };
}
export function generate(seed = 29, scenario = "alternating") {
  integer(seed, 1, 4294967295, "seed");
  if (!["alternating", "shift", "null", "flat"].includes(scenario))
    throw Error("Unknown scenario");
  let state = seed;
  const uniform = () => {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      return (state + 0.5) / 4294967296;
    },
    normal = () =>
      Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const returns = [],
    truth = [];
  for (let t = 0; t < 420; t++) {
    const k =
        scenario === "alternating"
          ? Math.floor(t / 70) % 2
          : scenario === "shift"
            ? Number(t >= 270)
            : 0,
      mean = k === 0 ? 0.0006 : -0.0005,
      sd = k === 0 ? 0.005 : 0.018;
    returns.push(
      scenario === "flat"
        ? 0
        : Math.max(-0.25, Math.min(0.25, mean + sd * normal())),
    );
    truth.push(k);
  }
  return { returns, truth };
}
