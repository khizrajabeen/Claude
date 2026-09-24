import { shell, load, explain, usd, pct, num, cls, price, compact, freshness, chipColour, base , gated } from "./app.js";

await gated();
shell("markets");
const $ = (id) => document.getElementById(id);
let all = [], sort = { k: "rank", dir: 1 }, mode = "all", traded = new Set();

function render() {
  const q = $("q").value.trim().toUpperCase();
  let rows = all.filter((r) => {
    if (q && !r.symbol.toUpperCase().includes(q)) return false;
    if (mode === "traded") return traded.has(r.symbol);
    if (mode === "new") return r.is_new;
    if (mode === "news") return r.news_articles > 0;
    return true;
  });
  rows.sort((a, b) => {
    const x = a[sort.k], y = b[sort.k];
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * sort.dir;
  });

  const t = (l, v, sub) => `<div class="card tile"><div class="label">${l}</div>
    <div class="value">${v}</div><div class="delta muted">${sub}</div></div>`;
  const vol = rows.reduce((a, r) => a + (r.quote_volume || 0), 0);
  const withNews = rows.filter((r) => r.news_articles > 0).length;
  const betas = rows.filter((r) => r.beta_r2 > 0.3).map((r) => r.beta);
  $("tiles").innerHTML =
    t("Markets shown", rows.length, `of ${all.length} screened`) +
    t("24h volume", compact(vol), "summed across the shown set") +
    t("In the news", withNews, "coins with scored coverage") +
    t("Median beta", betas.length ? num(betas.sort((a, b) => a - b)[betas.length >> 1], 2) : "—",
      "to BTC, where r² > 0.3");

  $("tbl").tBodies[0].innerHTML = rows.length ? rows.slice(0, 400).map((r) => `<tr>
    <td class="r muted num">${r.rank}</td>
    <td><span class="asset"><span class="coin" style="background:${chipColour(r.symbol)}">${base(r.symbol).slice(0,3)}</span>
      <span><span class="nm">${r.symbol}</span>
      <span class="tk">${r.is_new ? "new listing" : Math.round(r.age_days / 365) + "y old"}</span></span></span></td>
    <td class="r num">${price(r.price)}</td>
    <td class="r num ${cls(r.change_24h_pct)}">${pct(r.change_24h_pct, 1)}</td>
    <td class="r num">${compact(r.quote_volume)}</td>
    <td class="r num">${num(r.beta, 2)}</td>
    <td class="r num ${r.beta_r2 < 0.3 ? "muted" : ""}">${num(r.beta_r2, 2)}</td>
    <td class="r num muted">${num(r.age_days / 365, 1)}y</td>
    <td class="r num ${cls(r.news_score)}">${r.news_articles ? num(r.news_score, 2) : "—"}</td>
    <td class="muted" style="font-size:12px">${(r.notes || []).join(", ") || "—"}</td></tr>`).join("")
    : `<tr><td colspan="10" class="empty">Nothing matches.</td></tr>`;
}

try {
  const [s, m] = await Promise.all([load("screen"), load("meta")]);
  all = s.markets || [];
  traded = new Set([...(m.universe?.crypto_spot || []), ...(m.universe?.crypto_perp || [])]);
  const f = freshness(s.as_of);
  $("fresh").innerHTML = `<span class="dot-live ${f.state}"></span><span>${f.text}</span>`;
  $("q").oninput = render;
  $("filter").onclick = (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $("filter").children) x.setAttribute("aria-pressed", String(x === b));
    mode = b.dataset.f; render();
  };
  $("tbl").tHead.onclick = (e) => {
    const th = e.target.closest("th.sortable"); if (!th) return;
    sort = { k: th.dataset.k, dir: sort.k === th.dataset.k ? -sort.dir : -1 };
    render();
  };
  render();
} catch (e) { explain(document.querySelector(".content"), e); }
