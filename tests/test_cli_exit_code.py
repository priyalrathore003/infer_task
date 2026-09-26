"""`voice-tau run` must fail loudly when every simulation is an infrastructure_error.

Earlier today a keyless --audio run (6/6 infrastructure_error) and the airline and
baseline rate-limit runs all exited 0. tau2's run_domain() is replaced by a stand-in
that writes a results.json with chosen termination reasons.
"""

from pathlib import Path

import pytest
import tau2.runner
import tau2.utils.utils as tu
from typer.testing import CliRunner

from voice_tau import cli

from .cli_fakes import write_fake_results

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


@pytest.fixture
def fake_tau2(monkeypatch, tmp_path):
    """Returns a setter for the termination reasons the next fake run produces."""
    outcome: list[str] = []
    monkeypatch.chdir(tmp_path)  # results/ is written relative to cwd
    monkeypatch.setattr(tu, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(cli, "_preregister_plugins", lambda cfg: None)  # covered elsewhere
    monkeypatch.setattr(
        tau2.runner, "run_domain", lambda config: write_fake_results(tu.DATA_DIR, config.save_to, outcome)
    )
    return lambda terms: outcome.extend(terms)


def _invoke(name: str):
    return CliRunner().invoke(cli.app, ["run", str(CONFIG), "--name", name])


def test_all_infrastructure_errors_exit_nonzero(fake_tau2):
    fake_tau2(["infrastructure_error"] * 6)
    r = _invoke("all_infra")
    assert r.exit_code == 1, r.output
    assert "all 6 simulations ended as infrastructure_error" in r.output
    assert Path("results/all_infra/results.json").exists()  # results still saved


def test_partial_infrastructure_errors_warn_but_exit_zero(fake_tau2):
    fake_tau2(["user_stop", "infrastructure_error", "user_stop"])
    r = _invoke("partial_infra")
    assert r.exit_code == 0, r.output
    assert "1/3 simulations ended as infrastructure_error" in r.output


def test_clean_run_exits_zero(fake_tau2):
    fake_tau2(["user_stop", "agent_stop"])
    r = _invoke("clean")
    assert r.exit_code == 0, r.output
    assert "infrastructure_error" not in r.output
