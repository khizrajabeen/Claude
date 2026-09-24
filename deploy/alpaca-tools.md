# Alpaca from the terminal and from an assistant

Two separate things, both optional, neither of which the bot needs in
order to trade. The bot talks to Alpaca directly through
`bot/trading/alpaca_broker.py`. These are for *you* — checking the
account, poking at market data, asking questions in plain English —
and they run on your machine, not on the server.

## Which one to use

|              | MCP server                         | CLI                             |
|--------------|------------------------------------|---------------------------------|
| Runs as      | a background process for a session | one command, then exits         |
| Context cost | every tool schema in the window    | just the command string         |
| Needs        | an MCP client                      | any terminal                    |
| Best for     | "how did my week go?"              | scripts, cron, a focused action |

If you only want one, take the CLI. It composes with everything else
and costs nothing when idle.

## The repository already carries an MCP config

`.mcp.json` at the root configures the server for Claude Code, reading
the keys from your environment rather than storing them:

```json
"env": {
  "ALPACA_API_KEY": "${ALPACA_API_KEY_ID}",
  "ALPACA_SECRET_KEY": "${ALPACA_API_SECRET_KEY}"
}
```

That indirection is the point. The file is committed; your keys are not.
Export them first — from the same `.env` the bot uses:

```bash
set -a && . ./.env && set +a
```

### Toolsets

The config enables `account,stock-data,crypto-data,news,assets` — reading
only. Order placement is deliberately **off**.

The bot places orders under rules that were measured: a stop sized to
ATR, a risk budget, a correlation cap, a daily loss limit. An assistant
placing an order on request is none of that. Turn `trading` on if you
want it, knowing you are stepping outside every guard this project
exists to enforce:

```json
"ALPACA_TOOLSETS": "account,trading,stock-data,crypto-data,news,assets"
```

## Other clients

Same block, different file. Substitute your real keys, or export them
into the client's environment.

- **Cursor** — `~/.cursor/mcp.json`
- **Claude Desktop** — `claude_desktop_config.json`
- **VS Code** — `.vscode/mcp.json`, under `"servers"` rather than `"mcpServers"`

```bash
# Claude Code, user-wide instead of per-repository
claude mcp add alpaca --scope user --transport stdio uvx alpaca-mcp-server \
  --env ALPACA_API_KEY="$ALPACA_API_KEY_ID" \
  --env ALPACA_SECRET_KEY="$ALPACA_API_SECRET_KEY"
```

Verify by asking: *"What is my Alpaca account balance and buying power?"*

## The CLI

```bash
brew install alpacahq/tap/cli          # or: go install github.com/alpacahq/cli/cmd/alpaca@latest
alpaca version && alpaca doctor
```

Authenticate with the keys already in your environment — no profile, no
browser, nothing written to disk:

```bash
export ALPACA_API_KEY="$ALPACA_API_KEY_ID"
export ALPACA_SECRET_KEY="$ALPACA_API_SECRET_KEY"
alpaca account get
```

Useful against this bot:

```bash
# Does the account agree with what the dashboard is showing?
alpaca position list --jq '[.[] | {symbol, qty, unrealized_pl}]'

# What did it actually do today?
alpaca order list --status all --jq '[.[] | {symbol, side, filled_qty, filled_avg_price, status}]'

# The cost that decides whether a trade pays for itself
alpaca data crypto latest-quotes --symbol BTC/USD,ETH/USD,SOL/USD
```

### A warning worth repeating

`alpaca order cancel-all` and `alpaca position close-all` execute
immediately. There is no confirmation prompt, by design — which is what
makes the CLI good for scripts and dangerous in a shell where you
half-remember what is open.

Paper is the default. Live needs `--live` or `ALPACA_LIVE_TRADE=true`,
and a script that forgets to opt in hits paper rather than your money.
That default is the right way round.
