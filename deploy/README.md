# Running it 24/7

The dashboard is static. GitHub Pages serves files — it cannot run the
bot, hold your keys or place an order. Those need a machine you control.
This directory is how you get from one to the other:

```
  your server  ──runs the daily cycle──▶  state/  ──publish──▶  web/data/
                                                                    │
                                                              git push
                                                                    ▼
                                                        GitHub Pages (dashboard)
```

The bot trades and records. It publishes a handful of small JSON files.
The page reads them. Nothing on the public side ever holds a credential.

## 1. Put the code somewhere

```bash
git clone https://github.com/khizrajabeen/Claude.git /opt/meridian
cd /opt/meridian
pip install -r requirements.txt
```

## 2. Give it credentials

Use the **Export .env** button on the dashboard's settings page, or write
the file yourself:

```bash
# /opt/meridian/.env  — gitignored, never committed
EXCHANGE_NAME=okx
EXCHANGE_API_KEY=...
EXCHANGE_API_SECRET=...

# Optional: US stocks through Alpaca. Paper unless you say otherwise.
ALPACA_API_KEY_ID=PK...
ALPACA_API_SECRET_KEY=...
ALPACA_PAPER=true
```

Check it before trusting it:

```bash
chmod 600 .env
python main.py briefing        # reads markets, places nothing
```

**Keep paper trading on until the record convinces you.** `paper.enabled`
in `config.yaml` is the switch, and it is on by default. The measured
result over 90 days is roughly flat, which is not a reason to fund it.

## 3. Pick how it runs

### systemd timer (recommended)

One cycle a day at the UTC boundary, which is where the bot's day starts
and where perpetual funding settles.

```bash
sudo useradd --system --home /opt/meridian meridian
sudo chown -R meridian:meridian /opt/meridian
sudo cp deploy/meridian.service deploy/meridian.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now meridian.timer

systemctl list-timers meridian.timer     # when it fires next
journalctl -u meridian -f                # what it did
```

A timer is preferred over a long-running loop because a crashed loop
looks exactly like a quiet market, whereas a timer that failed to fire is
visible in the journal.

### Docker

```bash
cp .env.example .env    # then fill it in
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f
```

`MERIDIAN_MODE=run` walks day after day inside the container.
`MERIDIAN_MODE=day` runs one cycle and exits, for an external scheduler.

The journal is mounted from the host on purpose: it is the record, and
losing it loses every measurement the bot has made.

## 4. Publish to the dashboard

Set the branch the page builds from and the run script will commit and
push `web/data` after each cycle:

```bash
echo 'MERIDIAN_PUBLISH_BRANCH=main' >> .env
```

The machine needs push access — a deploy key with write scope is enough,
and is preferable to a personal token because it can be revoked for this
one repository.

Then enable Pages once, in **Settings → Pages → Source: GitHub Actions**.
The workflow in `.github/workflows/pages.yml` does the rest, and refuses
to publish if anything key-shaped appears under `web/`.

Three repository settings gate a publish, and all three fail the same
unhelpful way — a job that ends in a couple of seconds with no
downloadable logs, which reads like nothing happened at all:

- **Pages enabled**, with Actions as the source, as above.
- **A plan that allows it.** Pages from a private repository needs a
  paid plan; on a free one, make the repository public.
- **The `github-pages` environment must allow the branch.** When Pages
  is first enabled it permits the default branch only, so a deploy from
  any other branch is rejected before a step runs. Widen it under
  *Settings → Environments → github-pages → Deployment branches*, or
  deploy from the default branch.

Without push access the bot still trades and still records; only the
public dashboard goes stale. The dashboard says so rather than showing
old numbers as current.

## What can go wrong

**The dashboard shows old numbers.** The status pill turns amber past two
hours. Check `journalctl -u meridian` and whether the push succeeded.

**Alpaca returns 403.** Paper keys do not work against the live endpoint
or the reverse. `ALPACA_PAPER` decides which is used.

**The exchange is geo-blocked.** Binance returns 451 from some regions.
OKX, Kraken, KuCoin and Coinbase are configured alternatives.

**Nothing trades.** Usually correct behaviour rather than a fault: a
halted day, a market outside its session, or no signal clearing its edge
bar. `python main.py briefing` prints every symbol with the reason it was
skipped.

## The honest part

This is a paper trading system with a measured record that is roughly
flat over three months, and every statistic it reports carries an error
bar saying so. Backtests flatter — they fill at prices nobody queued for
and they know which markets survived. Run it on paper, read the record,
and decide for yourself. Nothing here is financial advice.
