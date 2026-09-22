/* One definition of the navigation, rendered into every page.

   Repeating the rail markup across six files guarantees they drift: a
   page gets added, five files get updated, and the sixth quietly points
   somewhere that no longer exists. */

export const PAGES = [
  { href: "index.html",      icon: "◈", label: "Overview" },
  { href: "dashboard.html",  icon: "▦", label: "Portfolio" },
  { href: "trades.html",     icon: "⇄", label: "Trades" },
  { href: "markets.html",    icon: "◎", label: "Markets" },
  { href: "strategies.html", icon: "▤", label: "Strategies" },
];

export const FOOTER = [
  { href: "settings.html", icon: "⚙", label: "Settings" },
];

export function renderNav() {
  const here = location.pathname.split("/").pop() || "index.html";

  document.querySelectorAll("[data-nav]").forEach((host) => {
    const link = (p) => `
      <a href="./${p.href}" title="${p.label}"
         ${p.href === here ? 'aria-current="page"' : ""}>${p.icon}</a>`;
    host.innerHTML =
      PAGES.map(link).join("") +
      `<span class="gap"></span>` +
      FOOTER.map(link).join("");
  });

  document.querySelectorAll("[data-topnav]").forEach((host) => {
    host.innerHTML = PAGES.map((p) => `
      <a href="./${p.href}"
         ${p.href === here ? 'aria-current="page"' : ""}>${p.label}</a>`
    ).join("");
  });
}
