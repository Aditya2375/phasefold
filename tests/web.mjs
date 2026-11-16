import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { analyze, generate, fit, features, predict } from "../docs/core.js";
let checks = 0;
function near(a, b, path = "root") {
  if (typeof a === "number") {
    assert.ok(
      Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)),
      `${path}:${a} vs ${b}`,
    );
    checks++;
  } else if (Array.isArray(a)) {
    assert.equal(a.length, b.length, path);
    a.forEach((v, i) => near(v, b[i], path + "." + i));
  } else if (a && typeof a === "object") {
    assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort());
    for (const k in a) near(a[k], b[k], path + "." + k);
  } else {
    assert.equal(a, b, path);
    checks++;
  }
}
const programs = [];
for (const scenario of ["alternating", "shift", "null", "flat"])
  for (const seed of [1, 29, 12345, 4294967295])
    for (const window of [5, 20, 60])
      programs.push({ scenario, seed, window, train_end: 240 });
const p = spawnSync("python3", ["-m", "tests.parity"], {
  input: JSON.stringify(programs),
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
assert.equal(p.status, 0, p.stderr);
const native = JSON.parse(p.stdout);
for (let i = 0; i < programs.length; i++) {
  const c = programs[i];
  near(
    analyze(generate(c.seed, c.scenario).returns, c.window, c.train_end),
    native[i],
  );
}
const flat = analyze(generate(29, "flat").returns);
assert.deepEqual(flat.model.counts, [220, 0]);
assert.equal(flat.model.inertia, 0);
let r = generate().returns,
  a = analyze(r);
r = r.slice();
r[300] = 0.2;
const b = analyze(r);
assert.deepEqual(a.model, b.model);
assert.deepEqual(a.rows.slice(0, 281), b.rows.slice(0, 281));
assert.throws(() => features(Array(100), 20));
assert.throws(() => fit(Array(20)));
console.log(
  JSON.stringify({
    scalar_checks: checks,
    full_result_native_parity: programs.length,
    status: "PASS",
  }),
);

// A swapped canonical order must preserve the training tie convention.
const tied = [
  [2, 2],
  [1, 0],
  [0, 2],
  [1, 2],
  [1, 2],
  [2, 1],
  [1, 0],
  [2, 0],
  [1, 0],
  [2, 1],
  [0, 2],
  [0, 1],
  [2, 1],
  [0, 2],
  [1, 1],
  [0, 0],
  [1, 2],
  [1, 0],
  [1, 1],
  [1, 0],
];
const tm = fit(tied);
assert.equal(tm.tie_label, 1);
assert.deepEqual(
  tm.labels,
  tied.map((p) => predict(tm, p)),
);
