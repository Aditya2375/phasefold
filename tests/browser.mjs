import { chromium } from "playwright";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdir } from "node:fs/promises";
const base = process.env.BASE_URL || "http://127.0.0.1:8351/";
let server;
if (!process.env.BASE_URL) {
  server = spawn(
    "python3",
    ["-m", "http.server", "8351", "--bind", "127.0.0.1", "--directory", "docs"],
    { stdio: "ignore" },
  );
  await new Promise((r) => setTimeout(r, 500));
}
const browser = await chromium.launch({ headless: true });
let count = 0;
const ok = (v, m) => {
  assert.ok(v, m);
  count++;
};
await mkdir("/tmp/phasefold-shots", { recursive: true });
try {
  for (const width of [1440, 1024, 768, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } }),
      errors = [],
      failed = [],
      third = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    page.on("requestfailed", (r) => failed.push(r.url()));
    page.on("request", (r) => {
      if (!r.url().startsWith(base)) third.push(r.url());
    });
    await page.goto(base);
    await page.waitForFunction(
      () =>
        document.querySelector("#model-status").textContent !== "Model ready.",
    );
    ok((await page.locator("#fit-count").innerText()) === "220", "fit rows");
    ok(
      (await page.locator("#frozen").innerText()) === "FROZEN AT 240",
      "freeze",
    );
    ok(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      "no page overflow",
    );
    await page.screenshot({ path: `/tmp/phasefold-shots/hero-${width}.png` });
    const actual = await page.evaluate(async () => {
      const { generate, analyze } = await import("./core.js");
      const r = analyze(generate().returns);
      return {
        hold: r.holdout_counts.join(" / "),
        sse: r.model.inertia.toFixed(2),
        switches: String(r.holdout_switches),
      };
    });
    ok(
      (await page.locator("#hold-count").innerText()) === actual.hold,
      "metrics actual model",
    );
    ok(
      (await page.locator("#inertia").innerText()) === actual.sse,
      "SSE actual",
    );
    ok(
      (await page.locator("#switches").innerText()) === actual.switches,
      "switches actual",
    );
    ok(
      await page
        .locator("#point-info")
        .innerText()
        .then((t) => t.includes("[220, 240)") && t.includes("frozen holdout")),
      "holdout feature boundary",
    );
    for (const id of ["scatter", "timeline"]) {
      ok(
        await page.locator("#" + id).evaluate((c) => {
          const d = c
            .getContext("2d")
            .getImageData(0, 0, c.width, c.height).data;
          return (
            Array.from(d).filter((v, i) => i % 4 === 3 && v > 0).length > 100
          );
        }),
        "actual chart pixels",
      );
    }
    await page.locator("#point").fill("239");
    ok(
      (await page.locator("#point-info").innerText()).includes(
        "retrospective fit",
      ),
      "training retrospective",
    );
    await page.locator("#window").selectOption("60");
    ok(await page.locator("#dirty").isVisible(), "dirty visible");
    ok(await page.locator("#point").isDisabled(), "dirty pauses inspection");
    ok(
      (await page.locator("#fit-count").innerText()) === "220",
      "old result unchanged",
    );
    await page.locator("#apply").click();
    ok(
      (await page.locator("#fit-count").innerText()) === "180",
      "applied window",
    );
    ok(
      (await page.locator("#point-info").innerText()).includes("[180, 240)"),
      "window exact",
    );
    await page.locator("#seed").fill("1e2");
    await page.locator("#apply").click();
    ok(
      (await page.locator("#error").innerText()).includes("whole decimal"),
      "strict integer",
    );
    ok(
      (await page.locator("#fit-count").innerText()) === "180",
      "invalid no mutation",
    );
    ok(await page.locator("#point").isDisabled(), "invalid paused");
    await page.locator("#seed").fill("4294967296");
    await page.locator("#apply").click();
    ok(
      (await page.locator("#error").innerText()).includes("seed must"),
      "seed bound",
    );
    await page.locator("#seed").fill("29");
    for (const scenario of ["flat", "null", "shift", "alternating"]) {
      await page.locator("#scenario").selectOption(scenario);
      await page.locator("#apply").click();
      ok((await page.locator("#error").innerText()) === "", "valid scenario");
      ok(!(await page.locator("#point").isDisabled()), "inspection enabled");
      if (scenario === "flat") {
        ok(
          (await page.locator("#model-status").innerText()) ===
            "One occupied cluster.",
          "no fabricated second state",
        );
        ok(
          (await page.locator("#inertia").innerText()) === "0.00",
          "flat objective",
        );
        ok(
          (await page.locator("#hold-count").innerText()) === "180 / 0",
          "flat tie0",
        );
        ok(
          (await page.locator("#scenario-note").innerText()).includes(
            "no identified second",
          ),
          "flat honesty",
        );
      }
      if (scenario === "null")
        ok(
          (await page.locator("#scenario-note").innerText()).includes(
            "not evidence",
          ),
          "null honesty",
        );
      if (scenario === "shift")
        ok(
          (await page.locator("#scenario-note").innerText()).includes("absent"),
          "unseen shift honesty",
        );
    }
    await page.locator("#train-end").selectOption("300");
    await page.locator("#apply").click();
    ok((await page.locator("#fit-count").innerText()) === "240", "new split");
    ok(
      (await page.locator("#frozen").innerText()) === "FROZEN AT 300",
      "new freeze",
    );
    await page.locator("#point").fill("419");
    ok(
      (await page.locator("#point-info").innerText()).includes("[359, 419)"),
      "last boundary exact",
    );
    const summaries = page.locator("summary");
    await summaries.nth(0).click();
    ok((await page.locator("#trace li").count()) >= 2, "objective trace");
    await summaries.nth(1).click();
    ok(
      (await page.locator("#ledger .ledger-row").count()) === 360,
      "full nonwarmup ledger",
    );
    await summaries.nth(1).click();
    await page.locator("#lab").scrollIntoViewIfNeeded();
    await page.screenshot({
      path: `/tmp/phasefold-shots/lab-${width}.png`,
      fullPage: true,
    });
    ok(errors.length === 0, "no errors " + errors.join(";"));
    ok(failed.length === 0, "no failed requests");
    ok(third.length === 0, "no thirdparty");
    ok(
      await page.evaluate(
        () =>
          localStorage.length === 0 &&
          sessionStorage.length === 0 &&
          document.cookie === "",
      ),
      "no persistence",
    );
    await page.close();
  }
  console.log(`${count} browser checks across 5 widths PASS`);
} finally {
  await browser.close();
  if (server) server.kill();
}
