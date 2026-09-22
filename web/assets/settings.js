/* Settings.

   Keys are held in this browser's localStorage under this site's origin.
   That is a deliberate and limited choice: a static page has nowhere else
   to put them, and anywhere else would mean this project holding
   somebody's exchange credentials. It is stated plainly on the page
   rather than implied, because "your keys never leave your device" is a
   claim people are entitled to check.

   The bot itself never reads this. It takes credentials from a .env file
   on the machine it runs on. */

import { initTheme, safeGet, safeSet, safeRemove } from "./app.js";
import { renderNav } from "./nav.js";

renderNav();
initTheme();

const PREFIX = "meridian.";

/* id → the environment variable the bot expects, so Export .env produces
   something that actually works rather than a plausible-looking file. */
const FIELDS = {
  "data-url":         { env: null },
  "auto-refresh":     { env: null, type: "check" },
  "k-anthropic":      { env: "ANTHROPIC_API_KEY" },
  "k-openai":         { env: "OPENAI_API_KEY" },
  "broker":           { env: "EXCHANGE_NAME" },
  "k-exchange":       { env: "EXCHANGE_API_KEY" },
  "k-exchange-secret":{ env: "EXCHANGE_API_SECRET" },
  "k-alpaca":         { env: "ALPACA_API_KEY_ID" },
  "k-alpaca-secret":  { env: "ALPACA_API_SECRET_KEY" },
  "alpaca-paper":     { env: "ALPACA_PAPER", type: "check" },
  "k-cryptopanic":    { env: "CRYPTOPANIC_TOKEN" },
  "k-newsapi":        { env: "NEWSAPI_KEY" },
  "k-cmc":            { env: "COINMARKETCAP_KEY" },
};

function node(id) { return document.getElementById(id); }

function restore() {
  for (const [id, spec] of Object.entries(FIELDS)) {
    const input = node(id);
    if (!input) continue;
    const saved = safeGet(PREFIX + id);
    if (saved === null) continue;
    if (spec.type === "check") input.checked = saved === "true";
    else input.value = saved;
  }
}

function save() {
  let ok = true;
  for (const [id, spec] of Object.entries(FIELDS)) {
    const input = node(id);
    if (!input) continue;
    const value = spec.type === "check" ? String(input.checked) : input.value;
    if (!value) { safeRemove(PREFIX + id); continue; }
    if (!safeSet(PREFIX + id, value)) ok = false;
  }
  flash(ok ? "Saved to this browser."
           : "Could not save — this browser is blocking site storage.");
}

function clearAll() {
  if (!confirm("Remove every key and preference stored by this page?")) return;
  for (const id of Object.keys(FIELDS)) safeRemove(PREFIX + id);
  document.querySelectorAll("input").forEach((i) => {
    if (i.type === "checkbox") i.checked = i.id === "alpaca-paper" || i.id === "auto-refresh";
    else i.value = "";
  });
  flash("Cleared.");
}

function exportEnv() {
  const lines = [
    "# Written by the Meridian settings page.",
    "# Put this on the machine that runs the bot, next to config.yaml.",
    "# It is gitignored; never commit it.",
    "",
  ];
  let any = false;
  for (const [id, spec] of Object.entries(FIELDS)) {
    if (!spec.env) continue;
    const input = node(id);
    if (!input) continue;
    const value = spec.type === "check" ? String(input.checked) : input.value.trim();
    if (!value) continue;
    any = true;
    lines.push(`${spec.env}=${value}`);
  }
  if (!any) { flash("Nothing to export — no keys entered."); return; }

  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: ".env" });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  flash("Exported .env — keep it off the repository.");
}

let flashTimer = null;
function flash(message) {
  const host = node("saved");
  host.textContent = message;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { host.textContent = ""; }, 4000);
}

document.querySelectorAll("[data-reveal]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const input = node(btn.dataset.reveal);
    const hidden = input.type === "password";
    input.type = hidden ? "text" : "password";
    btn.textContent = hidden ? "Hide" : "Show";
  });
});

node("save").addEventListener("click", save);
node("clear").addEventListener("click", clearAll);
node("export").addEventListener("click", exportEnv);

restore();
