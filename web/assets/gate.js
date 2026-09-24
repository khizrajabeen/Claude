/* A passcode on a static page.
 *
 * Be clear about what this is. The dashboard is files on a public host.
 * Every one of them — the page, the scripts, and every JSON file under
 * data/ — is downloadable by anyone with the URL, before a single line
 * of this file runs. A visitor who opens the network tab, or who fetches
 * `data/portfolio.json` directly with curl, sees everything regardless
 * of what is typed here.
 *
 * So this is a curtain, not a lock. It stops a shoulder-surfer and a
 * casual link-follower. It stops nobody who is trying.
 *
 * The passcode is compared as a SHA-256 hash so the literal string is
 * not sitting in the source, which is worth doing and is also not
 * security: the hash of a short passphrase falls to a wordlist in
 * seconds. It is here to avoid the worse outcome of someone reading the
 * passcode over your shoulder while you read the source.
 *
 * If this page ever needs to hold something that actually must not leak,
 * the answer is not a better gate — it is to stop publishing that thing
 * to a static host. Put it behind a server that can refuse to send it.
 */

const KEY = "gate-ok";

async function sha256(text) {
  const bytes = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function guard({ hash, title = "Meridian" }) {
  if (!hash) return true;                       // no passcode configured
  try {
    if (sessionStorage.getItem(KEY) === hash) return true;
  } catch { /* private mode — ask every time */ }

  document.documentElement.style.visibility = "visible";
  document.body.innerHTML = `
    <div style="min-height:100vh;display:grid;place-items:center;padding:24px;background:#0a0611">
      <form id="gate" style="width:100%;max-width:400px;text-align:center;color:#fff">
        <div class="mark-glyph" style="margin:0 auto 22px"></div>
        <h1 style="font-size:24px;margin-bottom:10px">${title}</h1>
        <p style="font-size:13.5px;color:rgba(255,255,255,.6);margin-bottom:22px;line-height:1.5">
          This dashboard is passcode-gated.
        </p>
        <input id="pc" type="password" autocomplete="current-password" autofocus
               placeholder="Passcode" aria-label="Passcode"
               style="width:100%;padding:12px 14px;border-radius:10px;
                      border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);
                      color:#fff;font-size:14px;text-align:center">
        <p id="err" style="min-height:18px;margin:10px 0;font-size:12.5px;color:#ff7ab2"></p>
        <button type="submit"
                style="width:100%;padding:12px;border:0;border-radius:999px;cursor:pointer;
                       background:linear-gradient(180deg,#ffa8ce,#f06aa8);color:#2b0518;
                       font-weight:600;font-size:14px">Open dashboard</button>
        <p style="font-size:11.5px;color:rgba(255,255,255,.4);margin-top:20px;line-height:1.55">
          This hides the page from casual visitors. It is not security: the
          data is served from a public host and can be fetched directly
          without answering this prompt.
        </p>
      </form>
    </div>`;

  return new Promise((resolve) => {
    document.getElementById("gate").addEventListener("submit", async (e) => {
      e.preventDefault();
      const got = await sha256(document.getElementById("pc").value);
      if (got === hash) {
        try { sessionStorage.setItem(KEY, hash); } catch { /* ignore */ }
        resolve(true);
        location.reload();
      } else {
        document.getElementById("err").textContent = "Not that one.";
        document.getElementById("pc").select();
      }
    });
  });
}

/** Read the configured hash. Empty or absent means no gate. */
export async function configuredHash() {
  try {
    const res = await fetch("gate.json", { cache: "no-store" });
    if (!res.ok) return "";
    return (await res.json()).sha256 || "";
  } catch {
    return "";
  }
}
