/** Headless render check for the single-file terminal build.
 *
 * Asserts against `#root`'s rendered text only — `body.textContent` also
 * contains the inlined payload and bundle source, which would make any
 * substring assertion meaningless. */
import { JSDOM, VirtualConsole } from "jsdom";
import fs from "node:fs";

const FILE = "../reports/terminal/master-us-terminal.html";
const errors = [];
const vc = new VirtualConsole()
  .on("jsdomError", (e) => errors.push(e.message))
  .on("error", (m) => errors.push(String(m)));

const dom = new JSDOM(fs.readFileSync(new URL(FILE, import.meta.url), "utf8"), {
  runScripts: "dangerously",
  pretendToBeVisual: true,
  url: "file:///terminal.html",
  virtualConsole: vc,
});
dom.window.scrollTo = () => {};
await new Promise((r) => setTimeout(r, 1200));

const d = dom.window.document;
const root = () => d.getElementById("root");
const seen = () => root()?.textContent ?? "";
let failed = 0;
const check = (label, cond, extra = "") => {
  if (!cond) failed++;
  console.log(`${cond ? "PASS" : "FAIL"}  ${label}${extra ? "  · " + extra : ""}`);
};

check("root mounted", (root()?.children.length ?? 0) > 0);
check("no runtime errors", errors.length === 0, errors.slice(0, 2).join(" | "));
check("command bar", !!d.querySelector(".bar"));
check("panel buttons", d.querySelectorAll(".key").length === 7, `${d.querySelectorAll(".key").length}`);
check("ticker input", !!d.querySelector(".ticker-input"));
check("blotter rows", d.querySelectorAll("tbody tr").length === 9, `${d.querySelectorAll("tbody tr").length}`);
check("gate ratios shown", seen().includes("0.949") && seen().includes("0.464"));
check("MSTR.US in blotter", seen().includes("MSTR.US"));
check("no undefined rendered", !seen().includes("undefined"));
check("no NaN rendered", !seen().includes("NaN"));
check("phases 5-7 complete on MONITOR", /FACTORS|JOIN|RISK/.test(seen()) && !/AWAITING RESULTS|NOT YET RUN/.test(seen()));

const expect = {
  monitor: { minSvg: 1, label: "equity chart" },
  quote: { minSvg: 4, label: "seed strips + turnover" },
  abla: { minSvg: 1, label: "beta sweep" },
  cost: { minSvg: 1, label: "cost curve + turnover hists" },
  risk: { minSvg: 0, minRows: 4, label: "book table" },
  attr: { minSvg: 0, minRows: 8, label: "timing table" },
  data: { minSvg: 1, label: "survivorship" },
};

for (const [id, e] of Object.entries(expect)) {
  const btn = d.getElementById(`key-${id}`);
  if (!btn) { check(`panel ${id.toUpperCase()}`, false, "button missing"); continue; }
  btn.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 200));
  const svg = d.querySelectorAll("svg").length;
  const rows = d.querySelectorAll("tbody tr").length;
  const txt = seen();
  const svgOk = svg >= (e.minSvg ?? 0);
  const rowsOk = e.minRows == null || rows >= e.minRows;
  const ok =
    (root()?.children.length ?? 0) > 0 &&
    !txt.includes("undefined") &&
    !txt.includes("NaN") &&
    !/AWAITING RESULTS|NOT YET RUN/.test(txt) &&
    svgOk &&
    rowsOk;
  check(
    `panel ${id.toUpperCase()}`,
    ok,
    `${svg} svg (${e.label}), ${rows} rows`,
  );
}

// RISK must carry the book table and the specific-alpha caveat, never a pending badge
d.getElementById("key-risk").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((r) => setTimeout(r, 200));
check("RISK has book rows", d.querySelectorAll("tbody tr").length >= 4);
check("RISK shows style-neutral", /STYLE-NEUTRAL|style-neutral/i.test(seen()));
check("RISK attaches specific-alpha caveat", /NOT an alpha number|not an alpha number|assumes factor hedging is free/i.test(seen()));
check("RISK is not pending", !/AWAITING RESULTS|NOT YET RUN/.test(seen()));

d.getElementById("key-attr").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((r) => setTimeout(r, 200));
check("ATTR has timing rows", d.querySelectorAll("tbody tr").length >= 8);
check("ATTR carries method note", /SUBSTITUTE|consequence|activation regression/i.test(seen()));
check("ATTR shows eigenfactor not confirmed", /NOT confirmed|failed prediction/i.test(seen()));
check("ATTR is not pending", !/AWAITING RESULTS|NOT YET RUN/.test(seen()));

// Ticker search suggests while typing and matches variant names, not only tickers
{
  const input = d.querySelector(".ticker-input");
  const setValue = Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, "value").set;
  const type = async (q) => {
    input.focus();
    setValue.call(input, q);
    input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 100));
    return [...d.querySelectorAll('[role="option"]')].map((o) => o.textContent);
  };
  const byName = await type("ridge");
  check("ticker search matches a variant name", byName.length === 1 && byName[0].includes("RIDG.BL"), byName.join(", "));
  const byCode = await type("ng");
  check("ticker search matches the code after the dot", byCode.some((o) => o.includes("MSTR.NG")), byCode.join(", "));
  const all = await type("");
  check("ticker search lists every variant when empty", all.length === 9, `${all.length}`);
  await type("ridge");
  input.form.dispatchEvent(new dom.window.Event("submit", { bubbles: true, cancelable: true }));
  await new Promise((r) => setTimeout(r, 200));
  check(
    "ticker search opens QUOTE",
    d.querySelector('.key[aria-selected="true"]')?.id === "key-quote" && seen().includes("RIDG.BL"),
  );
}

console.log(errors.length ? "\nerrors:\n" + errors.join("\n") : "");
console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
process.exit(failed ? 1 : 0);
