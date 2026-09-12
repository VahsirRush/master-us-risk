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

const expect = {
  monitor: { svg: 1, label: "equity chart" },
  quote: { svg: 5, label: "4 seed strips + turnover" },
  abla: { svg: 1, label: "beta sweep" },
  cost: { svg: 7, label: "cost curve + turnover hists" },
  data: { svg: 1, label: "survivorship" },
  risk: { svg: 0, label: "pending state" },
  attr: { svg: 0, label: "pending state" },
};

for (const [id, e] of Object.entries(expect)) {
  const btn = d.getElementById(`key-${id}`);
  if (!btn) { check(`panel ${id.toUpperCase()}`, false, "button missing"); continue; }
  btn.dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
  await new Promise((r) => setTimeout(r, 200));
  const svg = d.querySelectorAll("svg").length;
  const txt = seen();
  const ok = (root()?.children.length ?? 0) > 0 && !txt.includes("undefined") && !txt.includes("NaN") && svg === e.svg;
  check(`panel ${id.toUpperCase()}`, ok, `${svg} svg (${e.label}), ${d.querySelectorAll("tbody tr").length} rows`);
}

// pending panels must say so, and must not carry figures
d.getElementById("key-risk").dispatchEvent(new dom.window.MouseEvent("click", { bubbles: true }));
await new Promise((r) => setTimeout(r, 200));
check("RISK declares not-run", /NOT YET RUN/.test(seen()));
check("RISK has no table", d.querySelectorAll("tbody tr").length === 0);

console.log(errors.length ? "\nerrors:\n" + errors.join("\n") : "");
console.log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
process.exit(failed ? 1 : 0);
