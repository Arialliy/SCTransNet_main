from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO
    / "experiments/"
    "launch_tpd_ner_v4_qfg_v2_croa_formal800_finalizer_2x5090.sh"
)
SOURCE_LOCK = (
    REPO
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _harness(tmp_path: Path) -> tuple[dict[str, str], dict[str, Path]]:
    repo = tmp_path / "repo"
    experiments = repo / "experiments"
    result_root = repo / "results"
    c_run = (
        result_root
        / "NUDT-SIRST/qfg_only/seed_42_formal800_qfg_only"
    )
    d_run = (
        result_root
        / "NUDT-SIRST/tss_qfg/seed_42_formal800_tss_qfg"
    )
    experiments.mkdir(parents=True)
    c_run.mkdir(parents=True)
    d_run.mkdir(parents=True)

    finalizer_log = tmp_path / "finalizer.json"
    finalizer = _write_executable(
        experiments / "fake_finalizer.py",
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

Path(os.environ["FAKE_FINALIZER_LOG"]).write_text(
    json.dumps({"argv": sys.argv[1:]}),
    encoding="utf-8",
)
""",
    )
    systemd_log = tmp_path / "systemd-run.json"
    systemd_run = _write_executable(
        tmp_path / "fake-systemd-run",
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

Path(os.environ["FAKE_SYSTEMD_LOG"]).write_text(
    json.dumps(sys.argv[1:]),
    encoding="utf-8",
)
""",
    )
    systemctl_log = tmp_path / "systemctl.jsonl"
    systemctl = _write_executable(
        tmp_path / "fake-systemctl",
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

log = Path(os.environ["FAKE_SYSTEMCTL_LOG"])
with log.open("a", encoding="utf-8") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
prop = next(
    (arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--property=")),
    "",
)
values = {
    "LoadState": os.environ.get("FAKE_LOAD_STATE", "not-found"),
    "ActiveState": os.environ.get("FAKE_ACTIVE_STATE", "inactive"),
    "SubState": os.environ.get("FAKE_SUB_STATE", "dead"),
    "Result": os.environ.get("FAKE_RESULT", "success"),
    "NRestarts": os.environ.get("FAKE_RESTARTS", "0"),
}
print(values.get(prop, ""))
""",
    )

    env = dict(os.environ)
    env.update(
        {
            "TPD_NER_V4_QFG_V2_CROA_REPO": str(repo),
            "TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT": str(result_root),
            "TPD_NER_V4_QFG_V2_CROA_FINALIZER": str(finalizer),
            "TPD_NER_V4_QFG_V2_CROA_FINALIZER_SYSTEMD_RUN": str(
                systemd_run
            ),
            "TPD_NER_V4_QFG_V2_CROA_FINALIZER_SYSTEMCTL": str(systemctl),
            "FAKE_FINALIZER_LOG": str(finalizer_log),
            "FAKE_SYSTEMD_LOG": str(systemd_log),
            "FAKE_SYSTEMCTL_LOG": str(systemctl_log),
        }
    )
    paths = {
        "repo": repo,
        "c_run": c_run,
        "d_run": d_run,
        "finalizer_log": finalizer_log,
        "systemd_log": systemd_log,
        "systemctl_log": systemctl_log,
    }
    return env, paths


def _run(
    env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(LAUNCHER), *args],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_launcher_shell_parses_and_source_lock_is_unchanged() -> None:
    before = _sha256(SOURCE_LOCK)
    subprocess.run(["/usr/bin/bash", "-n", str(LAUNCHER)], check=True)
    assert _sha256(SOURCE_LOCK) == before


def test_static_service_contract_and_no_gpu_polling() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for token in (
        "--preflight",
        "--status",
        "--worker",
        "--collect",
        "--property=Restart=on-failure",
        "--property=RestartPreventExitStatus=64",
        "--property=RestartSec=60",
        "--property=StartLimitIntervalSec=0",
    ):
        assert token in source
    assert 'exec "$qfg_finalize_launch_finalizer"' in source
    assert "exit 75" in source
    for forbidden in (
        "nvidia-smi",
        "query_compute_processes",
        "memory.free",
        "utilization.gpu",
        "sleep ",
        "CUDA_VISIBLE_DEVICES",
        "cuda:",
    ):
        assert forbidden not in source


def test_preflight_is_read_only_with_missing_summaries(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    result = _run(env, "--preflight")
    assert result.returncode == 0, result.stderr
    assert (
        "readiness=qfg_only=missing tss_qfg=missing" in result.stdout
    )
    assert "writes_performed=false" in result.stdout
    assert not paths["finalizer_log"].exists()
    assert not paths["systemd_log"].exists()
    assert not paths["systemctl_log"].exists()


def test_worker_returns_75_until_both_summaries_exist(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    (paths["c_run"] / "summary.json").write_text("{}\n", encoding="utf-8")
    result = _run(env, "--worker")
    assert result.returncode == 75
    assert (
        "qfg_only=complete tss_qfg=missing" in result.stderr
    )
    assert not paths["finalizer_log"].exists()
    assert not paths["systemd_log"].exists()
    assert not paths["systemctl_log"].exists()


def test_worker_rejects_nonregular_summary_permanently(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    (paths["c_run"] / "summary.json").symlink_to(tmp_path / "missing")
    result = _run(env, "--worker")
    assert result.returncode == 64
    assert "summary_nonregular" in result.stderr
    assert not paths["finalizer_log"].exists()


def test_worker_executes_finalizer_after_both_summaries(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    for run in (paths["c_run"], paths["d_run"]):
        (run / "summary.json").write_text("{}\n", encoding="utf-8")
    result = _run(env, "--worker")
    assert result.returncode == 0, result.stderr
    assert "FINALIZER_LAUNCH_HANDOFF" in result.stdout
    assert json.loads(paths["finalizer_log"].read_text()) == {"argv": []}
    assert not paths["systemd_log"].exists()
    assert not paths["systemctl_log"].exists()


def test_launch_creates_only_mocked_restartable_user_service(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    argv = json.loads(paths["systemd_log"].read_text())
    for token in (
        "--user",
        "--collect",
        "--property=Type=exec",
        "--property=Restart=on-failure",
        "--property=RestartPreventExitStatus=64",
        "--property=RestartSec=60",
        "--property=StartLimitIntervalSec=0",
    ):
        assert token in argv
    assert argv[-2:] == [str(LAUNCHER.resolve()), "--worker"]
    assert "qfg_only_gpu=2 tss_qfg_gpu=3" in result.stdout
    assert not paths["finalizer_log"].exists()


def test_launch_is_idempotent_while_unit_is_active(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    env["FAKE_ACTIVE_STATE"] = "active"
    result = _run(env)
    assert result.returncode == 0, result.stderr
    assert "FINALIZER_LAUNCH_ALREADY_ACTIVE" in result.stdout
    assert not paths["systemd_log"].exists()
    assert not paths["finalizer_log"].exists()


def test_status_is_observational_and_uses_only_mocked_systemctl(
    tmp_path: Path,
) -> None:
    env, paths = _harness(tmp_path)
    env.update(
        {
            "FAKE_LOAD_STATE": "loaded",
            "FAKE_ACTIVE_STATE": "failed",
            "FAKE_RESULT": "exit-code",
            "FAKE_RESTARTS": "17",
        }
    )
    result = _run(env, "--status")
    assert result.returncode == 0, result.stderr
    assert "load=loaded" in result.stdout
    assert "active=failed" in result.stdout
    assert "result=exit-code" in result.stdout
    assert "restarts=17" in result.stdout
    assert "qfg_only=missing tss_qfg=missing" in result.stdout
    assert not paths["systemd_log"].exists()
    assert not paths["finalizer_log"].exists()


def test_usage_error_never_calls_external_commands(tmp_path: Path) -> None:
    env, paths = _harness(tmp_path)
    result = _run(env, "--worker", "unexpected")
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert not paths["systemd_log"].exists()
    assert not paths["systemctl_log"].exists()
    assert not paths["finalizer_log"].exists()
