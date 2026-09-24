import { shell , gated } from "./app.js";

await gated();
shell("settings");
const $ = (id) => document.getElementById(id);

/* Every field the bot actually reads. The placeholders are shaped like
 * the real thing but deliberately too short to be mistaken for a key —
 * the deploy workflow refuses to publish anything key-shaped under
 * web/, and a placeholder that trips that check is how a security
 * check gets switched off. */
const GROUPS = [
  { title: "Alpaca — US equities and ETFs", note:
    "Paper keys begin with PK, live keys with AK. Paper and live are separate credentials and separate endpoints.",
    fields: [
      { k: "ALPACA_API_KEY_ID",     l: "Key ID",     p: "PK…" },
      { k: "ALPACA_API_SECRET_KEY", l: "Secret key", p: "••••", secret: true },
      { k: "ALPACA_PAPER",          l: "Paper trading", type: "select", opts: ["true", "false"] },
    ] },
  { title: "Crypto venue", note:
    "One adapter, several venues. Binance returns 451 in some regions; OKX, Kraken, KuCoin and Coinbase are configured alternatives.",
    fields: [
      { k: "EXCHANGE_ID",     l: "Venue", type: "select",
        opts: ["binance", "okx", "kraken", "kucoin", "coinbase"] },
      { k: "EXCHANGE_API_KEY",    l: "API key",    p: "…" },
      { k: "EXCHANGE_API_SECRET", l: "API secret", p: "••••", secret: true },
      { k: "EXCHANGE_PASSWORD",   l: "Passphrase (OKX, KuCoin)", p: "••••", secret: true },
    ] },
  { title: "News", note:
    "Public RSS needs no key. A key raises the rate limit and widens per-ticker coverage.",
    fields: [
      { k: "NEWSAPI_KEY",       l: "NewsAPI key",       p: "…" },
      { k: "CRYPTOPANIC_TOKEN", l: "CryptoPanic token", p: "…" },
    ] },
  { title: "Language models", note:
    "Optional. Used to summarise the morning briefing; no model is consulted before a trade is sized.",
    fields: [
      { k: "ANTHROPIC_API_KEY", l: "Anthropic key", p: "…", secret: true },
      { k: "OPENAI_API_KEY",    l: "OpenAI key",    p: "…", secret: true },
    ] },
  { title: "Run", note: "How the bot behaves on the machine it runs on.",
    fields: [
      { k: "MERIDIAN_MODE", l: "Mode", type: "select", opts: ["paper", "live"] },
      { k: "MERIDIAN_PUBLISH_BRANCH", l: "Publish dashboard to branch", p: "main" },
    ] },
];

const read = () => { try { return JSON.parse(localStorage.getItem("keys") || "{}"); } catch { return {}; } };
const write = (o) => { try { localStorage.setItem("keys", JSON.stringify(o)); } catch {} };

function preview() {
  const v = read();
  const lines = ["# Meridian — generated from the settings page.",
                 "# Keep this file out of version control.", ""];
  for (const g of GROUPS) {
    const set = g.fields.filter((f) => v[f.k]);
    if (!set.length) continue;
    lines.push(`# ${g.title}`);
    for (const f of set) lines.push(`${f.k}=${v[f.k]}`);
    lines.push("");
  }
  $("preview").textContent = lines.length > 3 ? lines.join("\n")
    : "# Nothing filled in yet — the fields above write this file.";
}

$("groups").innerHTML = GROUPS.map((g) => `
  <section class="card">
    <div class="card-head"><h2>${g.title}</h2></div>
    <div class="card-body">
      <p class="muted" style="font-size:12.5px;margin-bottom:16px">${g.note}</p>
      ${g.fields.map((f) => `<div class="field">
        <label for="${f.k}">${f.l}</label>
        ${f.type === "select"
          ? `<select id="${f.k}" data-k="${f.k}">${f.opts.map((o) => `<option>${o}</option>`).join("")}</select>`
          : `<input id="${f.k}" data-k="${f.k}" type="${f.secret ? "password" : "text"}"
               placeholder="${f.p || ""}" autocomplete="off" spellcheck="false">`}
      </div>`).join("")}
    </div>
  </section>`).join("");

const saved = read();
for (const el of document.querySelectorAll("[data-k]")) {
  if (saved[el.dataset.k] != null) el.value = saved[el.dataset.k];
  el.addEventListener("input", () => {
    const v = read();
    if (el.value) v[el.dataset.k] = el.value; else delete v[el.dataset.k];
    write(v); preview();
  });
  el.addEventListener("change", () => el.dispatchEvent(new Event("input")));
}
preview();

$("export").onclick = () => {
  const blob = new Blob([$("preview").textContent], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = ".env";
  a.click();
  URL.revokeObjectURL(a.href);
};
$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("preview").textContent);
    $("copy").textContent = "Copied";
    setTimeout(() => ($("copy").textContent = "Copy"), 1400);
  } catch { $("copy").textContent = "Select it manually"; }
};
$("clear").onclick = () => {
  if (!confirm("Clear every key stored in this browser?")) return;
  try { localStorage.removeItem("keys"); } catch {}
  for (const el of document.querySelectorAll("[data-k]")) el.value = "";
  preview();
};
