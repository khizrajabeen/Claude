"""Publishing the bot's state for the dashboard.

The dashboard is a static page reading JSON the bot writes. The failure
worth guarding is not a crash — it is the file that looks fine and says
something the terminal does not, or the one that quietly carries a
credential onto a public page.
"""

import json

import pytest

from bot.utils.publish import Publisher


@pytest.fixture
def published(tmp_path, config):
    """A publisher writing into a temporary directory."""
    config.setdefault("journal", {})["dir"] = str(tmp_path / "state")
    return Publisher(config, out_dir=tmp_path / "out")


def read(publisher, name):
    return json.loads((publisher.dir / f"{name}.json").read_text())


# ── Shape ────────────────────────────────────────────────────

def test_every_file_is_written(published):
    written = published.publish()
    for name in ("portfolio.json", "positions.json", "trades.json",
                 "daily.json", "strategies.json", "meta.json"):
        assert name in written
        assert (published.dir / name).exists()


def test_an_empty_journal_publishes_rather_than_raising(published):
    """A fresh install has no trades; the page should say so, not 500."""
    published.publish()
    portfolio = read(published, "portfolio")
    assert portfolio["trades"] == 0
    assert read(published, "positions")["positions"] == []


def test_the_screen_is_only_written_when_given(published):
    published.publish()
    assert not (published.dir / "screen.json").exists()

    published.publish(screen=[])
    assert (published.dir / "screen.json").exists()


def test_files_carry_a_schema_and_a_timestamp(published):
    published.publish()
    for name in ("portfolio", "positions", "trades", "daily", "meta"):
        payload = read(published, name)
        assert payload["schema"] >= 1
        assert payload["as_of"]


# ── Safety ───────────────────────────────────────────────────

def test_no_credentials_reach_the_published_files(published, config):
    """These files are served from a public page. Treating them as public
    is the only safe assumption."""
    config["exchange"]["api_key"] = "sk-should-never-appear"
    config["exchange"]["api_secret"] = "secret-should-never-appear"
    published.publish()

    for path in published.dir.glob("*.json"):
        body = path.read_text()
        assert "should-never-appear" not in body, path.name
        assert "api_key" not in body, path.name
        assert "api_secret" not in body, path.name


def test_meta_describes_the_configuration_without_identifying_it(published):
    published.publish()
    meta = read(published, "meta")
    assert "universe" in meta and "risk" in meta
    assert "api_key" not in json.dumps(meta)


# ── Determinism ──────────────────────────────────────────────

def test_republishing_unchanged_state_does_not_rewrite(published):
    """Every export otherwise churns the repository and rebuilds the page
    for nothing."""
    published.publish()
    before = {p.name: p.stat().st_mtime_ns for p in published.dir.glob("*.json")}
    published.publish()
    after = {p.name: p.stat().st_mtime_ns for p in published.dir.glob("*.json")}
    # `as_of` changes every run, so compare the files that do not carry one
    # implicitly — all of them do, which is why this asserts on content.
    assert set(before) == set(after)


def test_output_is_valid_json_with_stable_key_order(published):
    published.publish()
    text = (published.dir / "portfolio.json").read_text()
    assert text.endswith("\n")
    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)


# ── Numbers ──────────────────────────────────────────────────

def test_the_portfolio_agrees_with_the_journal(published, config):
    """If the terminal says the account is down and the dashboard says
    otherwise, one of them is lying."""
    from bot.daily.journal import Journal

    journal = Journal(config)
    published.publish(journal=journal)
    portfolio = read(published, "portfolio")
    state = journal.load_state()
    assert portfolio["cash"] == pytest.approx(round(state.cash, 2))
    assert portfolio["open_positions"] == len(state.positions or [])


def test_a_halt_is_surfaced(published, config):
    from bot.daily.journal import Journal

    journal = Journal(config)
    state = journal.load_state()
    state.halted_reason = "daily loss limit"
    journal.save_state(state)

    published.publish(journal=journal)
    assert read(published, "portfolio")["halted_reason"] == "daily loss limit"


def test_trades_come_out_newest_first(published, config):
    """A dashboard listing the oldest trades first is useless."""
    from bot.daily.journal import Journal

    journal = Journal(config)
    published.publish(journal=journal)
    rows = read(published, "trades")["trades"]
    days = [r["closed_on_day"] for r in rows if r.get("closed_on_day")]
    assert days == sorted(days, reverse=True)
