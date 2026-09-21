import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { JSDOM } from "jsdom";

const HTML = readFileSync(new URL("../static/index.html", import.meta.url), "utf8");
const SCRIPT = readFileSync(new URL("../static/app.js", import.meta.url), "utf8");
const STYLE = readFileSync(new URL("../static/style.css", import.meta.url), "utf8");

const CARD = "20260921-014220-キングハイド.png";
const NEXT = "20260921-015500-バルランク.png";

function open(context, { styled = false } = {}) {
  const dom = new JSDOM(HTML, { runScripts: "outside-only", url: "http://localhost/" });
  const window = dom.window;
  context.after(() => window.close());
  if (styled) {
    const sheet = window.document.createElement("style");
    sheet.textContent = STYLE;
    window.document.head.append(sheet);
  }
  let status = { running: false, image: null, error: null };
  window.fetch = async () => ({ ok: true, json: async () => status });
  window.eval(SCRIPT);
  const at = (id) => window.document.getElementById(id);
  const poll = async (state) => {
    status = { running: false, image: null, error: null, ...state };
    await window.poll();
  };
  return { window, at, poll, arrive: (image) => poll({ image }) };
}

test("nothing is offered before the first card", async (context) => {
  const { at, poll } = open(context);
  await poll({});
  assert.equal(at("book").children.length, 0);
  assert.equal(at("controls").hidden, true);
  assert.equal(at("pager").hidden, true);
});

test("the first card is the page you are on", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  assert.equal(at("book").children.length, 1);
  assert.equal(at("book").children[0].hidden, false);
  assert.equal(at("controls").hidden, false);
  assert.equal(at("pager").hidden, true);
  assert.equal(at("count").textContent, "1 / 1");
});

test("the plate is drawn from the card it was handed", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  const image = at("book").querySelector("img");
  assert.equal(image.getAttribute("src"), `/api/image/${encodeURIComponent(CARD)}`);
});

test("the file is saved under the creature alone", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  assert.equal(at("save").download, "キングハイド.png");
  assert.equal(at("save").getAttribute("href"), `/api/image/${encodeURIComponent(CARD)}`);
});

test("a name without a stamp is saved as it stands", async (context) => {
  const { at, arrive } = open(context);
  await arrive("0-ムカシガニ.png");
  assert.equal(at("save").download, "0-ムカシガニ.png");
});

test("a second card turns forward and brings the pager", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  await arrive(NEXT);
  assert.equal(at("book").children.length, 2);
  assert.equal(at("book").children[0].hidden, true);
  assert.equal(at("book").children[1].hidden, false);
  assert.equal(at("pager").hidden, false);
  assert.equal(at("count").textContent, "2 / 2");
  assert.equal(at("save").download, "バルランク.png");
});

test("the ends of the book are dead", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  assert.equal(at("back").disabled, true);
  assert.equal(at("forward").disabled, true);
  await arrive(NEXT);
  assert.equal(at("back").disabled, false);
  assert.equal(at("forward").disabled, true);
  at("back").click();
  assert.equal(at("back").disabled, true);
  assert.equal(at("forward").disabled, false);
});

test("turning back shows the earlier page again", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  await arrive(NEXT);
  at("back").click();
  assert.equal(at("book").children[0].hidden, false);
  assert.equal(at("book").children[1].hidden, true);
  assert.equal(at("count").textContent, "1 / 2");
  assert.equal(at("save").download, "キングハイド.png");
  assert.ok(at("book").children[0].className.includes("turns-back"));
});

test("the arrow keys turn the page", async (context) => {
  const { window, at, arrive } = open(context);
  await arrive(CARD);
  await arrive(NEXT);
  const press = (key) =>
    window.document.dispatchEvent(new window.KeyboardEvent("keydown", { key }));
  press("ArrowLeft");
  assert.equal(at("count").textContent, "1 / 2");
  press("ArrowRight");
  assert.equal(at("count").textContent, "2 / 2");
  press("ArrowRight");
  assert.equal(at("count").textContent, "2 / 2");
  assert.equal(at("book").children[1].hidden, false);
});

test("the poll does not drag the reader back to the newest card", async (context) => {
  const { at, arrive } = open(context);
  await arrive(CARD);
  await arrive(NEXT);
  at("back").click();
  await arrive(NEXT);
  await arrive(NEXT);
  assert.equal(at("book").children.length, 2);
  assert.equal(at("count").textContent, "1 / 2");
});

test("a generation in flight leaves the page alone and holds the button", async (context) => {
  const { at, poll, arrive } = open(context);
  await arrive(CARD);
  await poll({ running: true });
  assert.equal(at("go").disabled, true);
  assert.equal(at("book").children.length, 1);
  assert.equal(at("book").children[0].hidden, false);
  assert.ok(at("note").textContent.startsWith("探索中"));
});

test("a failed generation puts its reason on the page", async (context) => {
  const { at, poll } = open(context);
  await poll({ error: "RuntimeError: comfyui is down" });
  assert.equal(at("go").disabled, false);
  assert.equal(at("note").className, "is-alarm");
  assert.ok(at("note").textContent.includes("comfyui is down"));
});

test("the stylesheet keeps hidden ahead of the display rules of its own ids", () => {
  const forced = /\[hidden\][^{]*\{[^}]*display:\s*none\s*!important/;
  const displayed = [...STYLE.matchAll(/#([a-z]+)[^{}]*\{[^}]*display:/g)].map((hit) => hit[1]);
  assert.ok(displayed.includes("controls") && displayed.includes("pager"));
  assert.match(STYLE, forced);
});
