import { analyze, generate } from "./core.js";
const $ = (s) => document.querySelector(s),
  colors = ["#4c8a83", "#c29459"];
let result,
  current,
  selected = 240,
  dirty = false;
function el(tag, text, className) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (className) e.className = className;
  return e;
}
function value(id) {
  const s = $("#" + id).value;
  if (!/^\d+$/.test(s)) throw Error("Use whole decimal integers only.");
  return Number(s);
}
const bps = (v) => (v * 10000).toFixed(2) + " bp";
function setDirty() {
  dirty = true;
  document.body.classList.add("pending");
  $("#dirty").hidden = false;
  $("#point").disabled = true;
}
function canvas(id) {
  const c = $("#" + id),
    r = c.getBoundingClientRect(),
    dpr = Math.min(2, devicePixelRatio || 1);
  c.width = Math.ceil(r.width * dpr);
  c.height = Math.ceil(r.height * dpr);
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  ctx.font = "9px ui-monospace,monospace";
  ctx.textBaseline = "middle";
  return { ctx, w: r.width, h: r.height };
}
function charts() {
  if (!result) return;
  const { ctx, w, h } = canvas("scatter"),
    rows = result.rows;
  const xs = rows.map((r) => r.mean),
    ys = rows.map((r) => r.sd),
    centers = result.model.centers.map((c) =>
      c.map((v, j) => v * result.model.scales[j] + result.model.means[j]),
    );
  xs.push(...centers.map((c) => c[0]));
  ys.push(...centers.map((c) => c[1]));
  let loX = Math.min(...xs),
    hiX = Math.max(...xs),
    loY = Math.min(0, ...ys),
    hiY = Math.max(...ys);
  if (hiX - loX < 1e-9) {
    loX -= 0.001;
    hiX += 0.001;
  }
  if (hiY - loY < 1e-9) hiY = 0.001;
  const padx = (hiX - loX) * 0.1,
    pady = (hiY - loY) * 0.1;
  loX -= padx;
  hiX += padx;
  hiY += pady;
  const X = (v) => 48 + ((v - loX) / (hiX - loX)) * (w - 68),
    Y = (v) => h - 38 - ((v - loY) / (hiY - loY)) * (h - 62);
  ctx.strokeStyle = "#dce7e1";
  ctx.fillStyle = "#71847d";
  for (let i = 0; i < 4; i++) {
    const y = loY + ((hiY - loY) * i) / 3,
      py = Y(y);
    ctx.beginPath();
    ctx.moveTo(45, py);
    ctx.lineTo(w - 16, py);
    ctx.stroke();
    ctx.fillText((y * 10000).toFixed(0), 7, py);
  }
  for (let i = 0; i < 3; i++) {
    const x = loX + ((hiX - loX) * i) / 2;
    ctx.textAlign = "center";
    ctx.fillText((x * 10000).toFixed(0), X(x), h - 21);
  }
  ctx.textAlign = "left";
  ctx.fillText("SD bp", 8, 12);
  ctx.textAlign = "right";
  ctx.fillText("mean bp", w - 15, h - 7);
  for (const r of rows) {
    ctx.beginPath();
    ctx.arc(X(r.mean), Y(r.sd), r.t === selected ? 4.5 : 2.2, 0, 2 * Math.PI);
    ctx.globalAlpha = r.t === selected ? 1 : 0.55;
    ctx.fillStyle = colors[r.label];
    ctx.strokeStyle = colors[r.label];
    if (r.split === "fit") ctx.fill();
    else ctx.stroke();
    if (r.t === selected) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#29383a";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.lineWidth = 1;
    }
  }
  ctx.globalAlpha = 1;
  centers.forEach((c, i) => {
    const x = X(c[0]),
      y = Y(c[1]);
    ctx.strokeStyle = "#253e3a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x - 6, y);
    ctx.lineTo(x + 6, y);
    ctx.moveTo(x, y - 6);
    ctx.lineTo(x, y + 6);
    ctx.stroke();
    ctx.lineWidth = 1;
  });
  const t = canvas("timeline"),
    c = t.ctx,
    W = t.w,
    H = t.h,
    left = 35,
    right = W - 14,
    range = 420 - current.window,
    tx = (v) => left + ((v - current.window) / range) * (right - left),
    boundary = tx(current.train_end);
  c.fillStyle = "#eaf1ed";
  c.fillRect(left, 18, boundary - left, 61);
  for (const r of rows) {
    c.fillStyle = colors[r.label];
    c.fillRect(
      tx(r.t),
      r.label === 0 ? 49 : 26,
      Math.max(1, (right - left) / range),
      19,
    );
  }
  c.strokeStyle = "#29383a";
  c.setLineDash([3, 3]);
  c.beginPath();
  c.moveTo(boundary, 12);
  c.lineTo(boundary, 85);
  c.stroke();
  c.setLineDash([]);
  c.fillStyle = "#6d7d7f";
  c.fillText("C1", 8, 35);
  c.fillText("C0", 8, 59);
  c.fillText("FIT", left, 96);
  c.textAlign = "right";
  c.fillText("HOLDOUT", right, 96);
  c.textAlign = "center";
  c.fillText("t=" + current.train_end, boundary, 9);
  c.strokeStyle = "#29383a";
  c.beginPath();
  c.moveTo(tx(selected), 20);
  c.lineTo(tx(selected), 78);
  c.stroke();
}
function inspect() {
  if (!result) return;
  const row = result.rows.find((r) => r.t === selected);
  $("#selected-label").textContent = "t = " + selected;
  const info = [
    ["Information window", `[${selected - current.window}, ${selected})`],
    [
      "Assignment",
      `C${row.label} / ${row.split === "fit" ? "retrospective fit" : "frozen holdout"}`,
    ],
    ["Trailing mean", bps(row.mean)],
    ["Trailing sample SD", bps(row.sd)],
    ["Squared distance", row.distance.toFixed(4)],
    ["Current return t", "Excluded from feature"],
  ];
  $("#point-info").replaceChildren(
    ...info.map(([a, b]) => {
      const d = el("div");
      d.append(el("span", a), el("b", b));
      return d;
    }),
  );
  charts();
}
function render() {
  const m = result.model,
    occupied = m.counts.filter((v) => v > 0).length;
  $("#run-label").textContent =
    `${current.scenario.toUpperCase()} / SEED ${current.seed}`;
  $("#model-status").textContent =
    occupied < 2
      ? "One occupied cluster."
      : `${m.converged ? "Converged" : "Iteration cap reached"} / ${m.iterations} updates`;
  $("#frozen").textContent = "FROZEN AT " + current.train_end;
  $("#fit-count").textContent = m.labels.length;
  $("#hold-count").textContent = result.holdout_counts.join(" / ");
  $("#switches").textContent = result.holdout_switches;
  $("#inertia").textContent = m.inertia.toFixed(2);
  const notes = {
    alternating:
      "Volatility alternates every 70 observations by construction. Mean/SD windows blur transitions; learned labels are not generator truth.",
    shift:
      "The generator changes at t = 270. With freeze at 240, that high-volatility state was absent from fitting. The old centroids still force every holdout row into a cluster.",
    null: "The generator has just one state. Two-means can still split random feature variation into two occupied groups. That is not evidence of two real regimes.",
    flat: "All returns are zero. The scaler falls back to one on constant features; both centroids coincide and ties go to C0. There is no identified second regime.",
  };
  $("#scenario-note").textContent = notes[current.scenario];
  $("#point").min = current.window;
  $("#point").value = selected;
  const rows = [
    ["Train mean", bps(m.means[0]), bps(m.means[1])],
    ["Train scale (population SD)", bps(m.scales[0]), bps(m.scales[1])],
    ...m.centers.map((c, k) => [
      `C${k} centroid / ${m.counts[k]} fit rows`,
      bps(c[0] * m.scales[0] + m.means[0]),
      bps(c[1] * m.scales[1] + m.means[1]),
    ]),
  ];
  $("#parameters").replaceChildren(
    ...rows.map(([name, a, b]) => {
      const d = el("div", undefined, "param-row");
      const n = el("div", name);
      d.append(n, el("span", a + " mean axis"), el("span", b + " SD axis"));
      return d;
    }),
  );
  const caption = el(
    "p",
    `Near-zero training feature SD (≤ 1e−12) uses scale 1 in raw units. Selected start index ${m.start}; ${occupied} occupied fit clusters. The objective trace is standardized, not in basis points.`,
    "hint",
  );
  $("#parameters").append(caption);
  $("#trace").replaceChildren(...m.trace.map((v) => el("li", v.toFixed(8))));
  $("#ledger").replaceChildren(
    ...result.rows.map((r) => {
      const d = el("div", undefined, "ledger-row");
      for (const [a, b] of [
        [`t = ${r.t}`, `${r.split} / C${r.label}`],
        ["Features", `${bps(r.mean)} mean / ${bps(r.sd)} SD`],
        ["Squared distance", r.distance.toFixed(4)],
      ]) {
        const c = el("div");
        c.append(el("b", a), el("span", b));
        d.append(c);
      }
      return d;
    }),
  );
  inspect();
}
function apply() {
  try {
    const proposed = {
        seed: value("seed"),
        scenario: $("#scenario").value,
        window: value("window"),
        train_end: value("train-end"),
      },
      generated = generate(proposed.seed, proposed.scenario),
      next = analyze(generated.returns, proposed.window, proposed.train_end);
    current = proposed;
    result = next;
    selected = current.train_end;
    dirty = false;
    $("#dirty").hidden = true;
    $("#error").textContent = "";
    document.body.classList.remove("pending");
    $("#point").disabled = false;
    render();
  } catch (e) {
    $("#error").textContent = e.message;
    setDirty();
  }
}
for (const id of ["seed", "scenario", "window", "train-end"])
  $("#" + id).addEventListener("input", setDirty);
$("#apply").addEventListener("click", apply);
$("#point").addEventListener("input", () => {
  if (dirty) return;
  selected = Number($("#point").value);
  inspect();
});
new ResizeObserver(() => charts()).observe($("#scatter"));
apply();
