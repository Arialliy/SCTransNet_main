from __future__ import annotations

import copy
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FINALIZER = (
    REPO_ROOT
    / "experiments/finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh"
)
SOURCE_LOCK_NAME = (
    "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
)


def _write(path: Path, content: str, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    if executable:
        path.chmod(0o755)
    return path


@dataclass(frozen=True)
class Harness:
    repo: Path
    result_root: Path
    survival_root: Path
    source_lock: Path
    log: Path
    env: dict[str, str]
    a_run: Path
    b_run: Path
    c_run: Path
    d_run: Path
    factorial_json: Path
    factorial_md: Path
    selection_json: Path
    selection_md: Path
    deployment_artifact: Path
    deployment_manifest: Path

    @property
    def terminal_paths(self) -> tuple[Path, ...]:
        return (
            self.factorial_json,
            self.factorial_md,
            self.selection_json,
            self.selection_md,
            self.deployment_artifact,
            self.deployment_manifest,
        )

    def calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    repo = tmp_path / "fake_repo"
    experiments = repo / "experiments"
    (repo / "datasets").mkdir(parents=True)
    result_root = (
        experiments
        / "results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
    )
    survival_root = experiments / "results/tpd_ner_v4_survival_exact_v1"
    a_run = (
        survival_root
        / "NUDT-SIRST/tss_control/seed_42_formal800_control"
    )
    b_run = survival_root / "NUDT-SIRST/tss_on/seed_42_formal800_tss"
    c_run = (
        result_root
        / "NUDT-SIRST/qfg_only/seed_42_formal800_qfg_only"
    )
    d_run = (
        result_root
        / "NUDT-SIRST/tss_qfg/seed_42_formal800_tss_qfg"
    )
    for run_dir in (a_run, b_run, c_run, d_run):
        run_dir.mkdir(parents=True)
    for run_dir in (a_run, b_run):
        for name in (
            "pd_fa_sweep_best.pth.json",
            "pd_fa_sweep_best_miou.pth.json",
        ):
            (run_dir / name).write_text("{}\n", encoding="utf-8")
    for run_dir, variant in ((c_run, "qfg_only"), (d_run, "tss_qfg")):
        _write(
            run_dir / "summary.json",
            json.dumps(
                {
                    "schema": "fake_completion_v1",
                    "status": "complete",
                    "variant": variant,
                    "epochs": 800,
                },
                sort_keys=True,
            )
            + "\n",
        )
        _write(
            run_dir / "exact_journal/active.json",
            json.dumps(
                {
                    "schema": "fake_journal_v1",
                    "variant": variant,
                    "epoch": 800,
                },
                sort_keys=True,
            )
            + "\n",
        )

    source_lock = _write(experiments / SOURCE_LOCK_NAME, "{}\n")
    closure_lock = _write(
        experiments
        / "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json",
        "{}\n",
    )
    log = tmp_path / "calls.log"

    freezer = _write(
        experiments / "fake_freezer.py",
        f"""\
        #!{sys.executable}
        import argparse
        import os
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--verify", action="store_true", required=True)
        parser.add_argument("--dataset-dir", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as out:
            out.write("source_verify\\n")
        if os.environ.get("FAIL_STAGE") == "source_verify":
            raise SystemExit(31)
        if Path(args.dataset_dir).resolve() != Path(
            os.environ["EXPECTED_DATASET_DIR"]
        ).resolve():
            raise SystemExit("wrong dataset directory")
        if Path(args.output).resolve() != Path(
            os.environ["EXPECTED_SOURCE_LOCK"]
        ).resolve():
            raise SystemExit("wrong source-lock output")
        """,
    )
    completion = _write(
        experiments / "fake_completion_checker.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys

        values = sys.argv[1:]
        if len(values) != 5:
            raise SystemExit("wrong completion checker argv")
        for run_dir, variant in (
            (Path(values[0]), values[1]),
            (Path(values[2]), values[3]),
        ):
            with Path(os.environ["FAKE_CALL_LOG"]).open(
                "a", encoding="utf-8"
            ) as out:
                out.write(f"completion_{{variant}}\\n")
            summary = json.loads(
                (run_dir / "summary.json").read_text(encoding="utf-8")
            )
            marker = json.loads(
                (run_dir / "exact_journal/active.json").read_text(
                    encoding="utf-8"
                )
            )
            if (
                summary.get("status") != "complete"
                or summary.get("variant") != variant
                or summary.get("epochs") != 800
                or marker.get("variant") != variant
                or marker.get("epoch") != 800
            ):
                raise SystemExit(f"incomplete {{variant}} run")
            if os.environ.get("FAIL_STAGE") == f"completion_{{variant}}":
                raise SystemExit(32)
        if Path(values[4]).resolve() != Path(
            os.environ["EXPECTED_SOURCE_LOCK"]
        ).resolve():
            raise SystemExit("completion checker received wrong source lock")
        """,
        executable=True,
    )
    sweeps = _write(
        experiments / "fake_sweeps_runner.py",
        f"""\
        #!{sys.executable}
        import os
        from pathlib import Path
        import sys

        args = sys.argv[1:]
        if args not in ([], ["--preflight"]):
            raise SystemExit("wrong sweeps argv")
        log = Path(os.environ["FAKE_CALL_LOG"])
        with log.open("a", encoding="utf-8") as out:
            out.write("sweeps_preflight\\n")
        if os.environ.get("FAIL_STAGE") == "sweeps_preflight":
            raise SystemExit(33)
        if args == ["--preflight"]:
            raise SystemExit(0)
        with log.open("a", encoding="utf-8") as out:
            out.write("sweeps_run\\n")
        if os.environ.get("FAIL_STAGE") == "sweeps_run":
            raise SystemExit(34)
        root = Path(os.environ["TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT"])
        for relative in (
            "NUDT-SIRST/qfg_only/seed_42_formal800_qfg_only",
            "NUDT-SIRST/tss_qfg/seed_42_formal800_tss_qfg",
        ):
            run_dir = root / relative
            for name in (
                "pd_fa_sweep_best.pth.json",
                "pd_fa_sweep_best_miou.pth.json",
            ):
                path = run_dir / name
                if not path.exists():
                    path.write_text('{{"status":"complete"}}\\n', encoding="utf-8")
        """,
        executable=True,
    )
    factorial = _write(
        experiments / "fake_factorial.py",
        f"""\
        #!{sys.executable}
        import argparse
        import json
        import os
        from pathlib import Path


        def _log(value):
            with Path(os.environ["FAKE_CALL_LOG"]).open(
                "a", encoding="utf-8"
            ) as out:
                out.write(value + "\\n")


        def collect_validated_sweeps(run_directories):
            for arm in ("A", "B", "C", "D"):
                run_dir = Path(run_directories[arm])
                for name in (
                    "pd_fa_sweep_best.pth.json",
                    "pd_fa_sweep_best_miou.pth.json",
                ):
                    if not (run_dir / name).is_file():
                        raise RuntimeError(f"missing sweep {{arm}}:{{name}}")
            return {{
                arm: str(Path(path).resolve())
                for arm, path in run_directories.items()
            }}


        def build_factorial_report(records):
            _log("factorial_build")
            if os.environ.get("FAIL_STAGE") == "factorial_build":
                raise RuntimeError("injected factorial build failure")
            return {{
                "schema": "fake_factorial_v1",
                "status": "complete",
                "runs": dict(sorted(records.items())),
            }}


        def render_markdown(report):
            return "# fake factorial\\n\\n" + json.dumps(
                report, ensure_ascii=False, sort_keys=True
            ) + "\\n"


        def _write_once(path, content):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as out:
                out.write(content)


        def main():
            parser = argparse.ArgumentParser()
            action = parser.add_mutually_exclusive_group(required=True)
            action.add_argument("--preflight", action="store_true")
            action.add_argument("--aggregate", action="store_true")
            for arm in ("a", "b", "c", "d"):
                parser.add_argument(f"--{{arm}}-run-dir", required=True)
            parser.add_argument("--json-output", required=True)
            parser.add_argument("--markdown-output", required=True)
            args = parser.parse_args()
            runs = {{
                arm.upper(): Path(getattr(args, f"{{arm}}_run_dir"))
                for arm in ("a", "b", "c", "d")
            }}
            records = collect_validated_sweeps(runs)
            if args.preflight:
                _log("factorial_preflight")
                if os.environ.get("FAIL_STAGE") == "factorial_preflight":
                    raise SystemExit(35)
                return
            _log("factorial_aggregate")
            if os.environ.get("FAIL_STAGE") == "factorial_aggregate":
                raise SystemExit(36)
            report = build_factorial_report(records)
            _write_once(
                args.json_output,
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\\n",
            )
            _write_once(args.markdown_output, render_markdown(report))


        if __name__ == "__main__":
            main()
        """,
    )
    postprocess = _write(
        experiments / "fake_postprocess.py",
        f"""\
        #!{sys.executable}
        import argparse
        import json
        import os
        from pathlib import Path


        def _log(value):
            with Path(os.environ["FAKE_CALL_LOG"]).open(
                "a", encoding="utf-8"
            ) as out:
                out.write(value + "\\n")


        def build_formal_report():
            _log("postprocess_build")
            if os.environ.get("FAIL_STAGE") == "postprocess_build":
                raise RuntimeError("injected postprocess build failure")
            return {{
                "schema": "fake_final_selection_v1",
                "status": "complete",
                "decision": "FAKE_SELECTED",
            }}


        def render_markdown(report):
            return "# fake final selection\\n\\n" + json.dumps(
                report, ensure_ascii=False, sort_keys=True
            ) + "\\n"


        def _write_once(path, content):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as out:
                out.write(content)


        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--json-output", required=True)
            parser.add_argument("--markdown-output", required=True)
            args = parser.parse_args()
            _log("postprocess_publish")
            if os.environ.get("FAIL_STAGE") == "postprocess_publish":
                raise SystemExit(37)
            report = build_formal_report()
            _write_once(
                args.json_output,
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\\n",
            )
            _write_once(args.markdown_output, render_markdown(report))


        if __name__ == "__main__":
            main()
        """,
    )
    closure_freezer = _write(
        experiments / "fake_closure_freezer.py",
        f"""\
        #!{sys.executable}
        import argparse
        import os
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--verify", action="store_true", required=True)
        parser.add_argument("--output", required=True)
        args = parser.parse_args()
        with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as out:
            out.write("closure_verify\\n")
        if os.environ.get("FAIL_STAGE") == "closure_verify":
            raise SystemExit(38)
        if Path(args.output).resolve() != Path(
            os.environ["EXPECTED_CLOSURE_LOCK"]
        ).resolve():
            raise SystemExit("wrong closure-lock output")
        """,
    )
    deployer = _write(
        experiments / "fake_deployer.py",
        f"""\
        #!{sys.executable}
        import argparse
        import json
        import os
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--preflight", action="store_true")
        parser.add_argument("--verify", action="store_true")
        parser.add_argument("--selection", required=True)
        parser.add_argument("--artifact", required=True)
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--closure-source-lock", required=True)
        args = parser.parse_args()
        action = (
            "deploy_preflight"
            if args.preflight
            else "deploy_verify"
            if args.verify
            else "deploy_publish"
        )
        with Path(os.environ["FAKE_CALL_LOG"]).open("a", encoding="utf-8") as out:
            out.write(action + "\\n")
        if os.environ.get("FAIL_STAGE") == action:
            raise SystemExit(39)
        selection = Path(args.selection)
        artifact = Path(args.artifact)
        manifest = Path(args.manifest)
        if not selection.is_file():
            raise SystemExit("selection missing")
        if args.preflight:
            raise SystemExit(0)
        if args.verify:
            if not artifact.is_file() or not manifest.is_file():
                raise SystemExit("deployment incomplete")
            raise SystemExit(0)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if not artifact.exists():
            artifact.write_bytes(b"fake-inference-artifact\\n")
        if not manifest.exists():
            manifest.write_text(
                json.dumps({{"status": "complete"}}, sort_keys=True) + "\\n",
                encoding="utf-8",
            )
        """,
    )

    factorial_json = (
        result_root
        / "NUDT-SIRST/comparison_factorial_v1/"
        "tss_qfg_v2_croa_factorial_seed42.json"
    )
    factorial_md = factorial_json.with_suffix(".md")
    selection_json = (
        result_root
        / "NUDT-SIRST/final_selection/"
        "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json"
    )
    selection_md = selection_json.with_suffix(".md")
    deployment_artifact = (
        result_root
        / "NUDT-SIRST/deployment/"
        "tpd_ner_v4_qfg_v2_croa_formal800_inference.pth.tar"
    )
    deployment_manifest = (
        result_root
        / "NUDT-SIRST/deployment/"
        "tpd_ner_v4_qfg_v2_croa_formal800_deployment_manifest.json"
    )
    env = {
        **os.environ,
        "TPD_NER_V4_QFG_V2_CROA_REPO": str(repo),
        "TPD_NER_V4_QFG_V2_CROA_PYTHON": sys.executable,
        "TPD_NER_V4_QFG_V2_CROA_SOURCE_LOCK": str(source_lock),
        "TPD_NER_V4_QFG_V2_CROA_RESULT_ROOT": str(result_root),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_SURVIVAL_RESULT_ROOT": (
            str(survival_root)
        ),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_FREEZER": str(freezer),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_SWEEPS_RUNNER": str(sweeps),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_FACTORIAL_COMPARE": (
            str(factorial)
        ),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_POSTPROCESS": str(postprocess),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_CLOSURE_FREEZER": (
            str(closure_freezer)
        ),
        "TPD_NER_V4_QFG_V2_CROA_POSTTRAINING_SOURCE_LOCK": (
            str(closure_lock)
        ),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_DEPLOYER": str(deployer),
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_COMPLETION_CHECKER": (
            str(completion)
        ),
        "FAKE_CALL_LOG": str(log),
        "EXPECTED_DATASET_DIR": str(repo / "datasets"),
        "EXPECTED_SOURCE_LOCK": str(source_lock),
        "EXPECTED_CLOSURE_LOCK": str(closure_lock),
    }
    return Harness(
        repo=repo,
        result_root=result_root,
        survival_root=survival_root,
        source_lock=source_lock,
        log=log,
        env=env,
        a_run=a_run,
        b_run=b_run,
        c_run=c_run,
        d_run=d_run,
        factorial_json=factorial_json,
        factorial_md=factorial_md,
        selection_json=selection_json,
        selection_md=selection_md,
        deployment_artifact=deployment_artifact,
        deployment_manifest=deployment_manifest,
    )


def _run(
    harness: Harness,
    *args: str,
    fail_stage: str | None = None,
    drop_env: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = dict(harness.env)
    for name in drop_env:
        env.pop(name, None)
    if fail_stage is not None:
        env["FAIL_STAGE"] = fail_stage
    else:
        env.pop("FAIL_STAGE", None)
    return subprocess.run(
        [str(FINALIZER), *args],
        cwd=harness.repo,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _assert_no_terminal_outputs(harness: Harness) -> None:
    assert all(
        not path.exists() and not path.is_symlink()
        for path in harness.terminal_paths
    )


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, tuple[int, int, str]]:
    return {
        path: (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    }


def _seed_candidate_sweeps(harness: Harness) -> None:
    for run_dir in (harness.c_run, harness.d_run):
        for name in (
            "pd_fa_sweep_best.pth.json",
            "pd_fa_sweep_best_miou.pth.json",
        ):
            (run_dir / name).write_text(
                '{"status":"complete"}\n',
                encoding="utf-8",
            )


def _install_embedded_completion_fixture(harness: Harness) -> None:
    """Install a CPU-only exact-run facade for the embedded checker."""

    _write(harness.repo / "experiments/__init__.py", "")
    _write(
        harness.repo / "experiments/tpd_exact_epoch_journal.py",
        """\
        from __future__ import annotations

        import hashlib
        import json
        from pathlib import Path
        from types import SimpleNamespace

        MARKER_FILENAME = "active.json"


        class ExactEpochJournal:
            def __init__(self, root):
                self.root = Path(root)

            def load_active(self):
                marker_path = self.root / MARKER_FILENAME
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                metrics_path = self.root / marker["metrics_file"]
                metrics = metrics_path.read_bytes()
                digest = hashlib.sha256(metrics).hexdigest()
                if marker["metrics_sha256"] != digest:
                    raise ValueError("active metrics digest differs")
                if len(metrics.splitlines()) != marker["epoch"]:
                    raise ValueError("active event count differs")
                return SimpleNamespace(
                    epoch=marker["epoch"],
                    metrics_path=metrics_path,
                    metrics_boundary={"metrics_sha256": digest},
                    marker_sha256=hashlib.sha256(
                        marker_path.read_bytes()
                    ).hexdigest(),
                )
        """,
    )
    _write(
        harness.repo
        / "experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py",
        """\
        from __future__ import annotations

        import json
        from pathlib import Path
        from types import SimpleNamespace

        FORMAL_EPOCHS = 800
        SUPPORTED_CANDIDATE_VARIANTS = ("qfg_only", "tss_qfg")
        STORED_VALIDATION_METRICS = ("pd", "fa", "miou", "tiny_pd")
        COMPLETION_SUMMARY_SCHEMA = "fake_exact_completion_v1"
        SOURCE_LOCK_KEY = "tpd_ner_v4_qfg_v2_croa_exact_source_lock"
        RUN_ID_PREFIX = "tpd-ner-v4-qfg-v2-croa-exact:"


        class ExactRunnerError(ValueError):
            pass


        class Policy:
            def recompute(self, events, require_flags=False):
                if require_flags and (
                    not any(event.get("new_best_pd") for event in events)
                    or not any(event.get("new_best_miou") for event in events)
                ):
                    raise ExactRunnerError("selection flags are missing")
                primary = [
                    event for event in events if event.get("new_best_pd")
                ][-1]
                secondary = [
                    event for event in events if event.get("new_best_miou")
                ][-1]

                def record(event, role):
                    return {
                        "epoch": event["epoch"],
                        "role": role,
                        "metrics": {
                            name: event[name]
                            for name in STORED_VALIDATION_METRICS
                        },
                    }

                return {
                    "primary": record(
                        primary,
                        "best_validation_pd_primary",
                    ),
                    "secondary": record(
                        secondary,
                        "best_validation_miou_secondary",
                    ),
                }


        exact_runner = SimpleNamespace(
            METRICS_FILENAME="metrics.jsonl",
            JOURNAL_DIRECTORY="exact_journal",
            ExactRunnerError=ExactRunnerError,
            pd_miou_selection_policy=lambda **kwargs: Policy(),
        )


        def _load_complete_events(path, epochs):
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if (
                len(rows) != epochs
                or [row.get("epoch") for row in rows]
                != list(range(1, epochs + 1))
            ):
                raise ValueError("metrics are not contiguous")
            for row in rows:
                for name in STORED_VALIDATION_METRICS:
                    if name not in row:
                        raise ValueError(f"metrics lack {name}")
            return rows


        def candidate_contract(variant):
            if variant not in SUPPORTED_CANDIDATE_VARIANTS:
                raise ValueError("unsupported candidate")
            return {
                "qfg_variant": "qfg_v2_croa",
                "tss_variant": (
                    "tss_control" if variant == "qfg_only" else "tss_on"
                ),
            }


        def formal_contract():
            return {
                "dataset": "NUDT-SIRST",
                "seed": 42,
                "split_seed": 20260722,
                "epochs": 800,
                "candidate_variants": list(SUPPORTED_CANDIDATE_VARIANTS),
            }


        def require_qfg_run_identity(
            identity,
            *,
            label,
            expected_variant=None,
        ):
            if not isinstance(identity, dict):
                raise ValueError(f"{label} is missing")
            if identity.get("variant") != expected_variant:
                raise ValueError(f"{label} variant differs")
            if identity.get("dataset") != "NUDT-SIRST":
                raise ValueError(f"{label} dataset differs")
            if identity.get("seed") != 42:
                raise ValueError(f"{label} seed differs")
            if identity.get("split_seed") != 20260722:
                raise ValueError(f"{label} split seed differs")
            locks = identity.get("source_locks")
            if not isinstance(locks, dict) or SOURCE_LOCK_KEY not in locks:
                raise ValueError(f"{label} source locks differ")
            return dict(identity)
        """,
    )

    source_lock_sha256 = hashlib.sha256(
        harness.source_lock.read_bytes()
    ).hexdigest()
    stored_metrics = ("pd", "fa", "miou", "tiny_pd")
    formal_contract = {
        "dataset": "NUDT-SIRST",
        "seed": 42,
        "split_seed": 20260722,
        "epochs": 800,
        "candidate_variants": ["qfg_only", "tss_qfg"],
    }
    for run_dir, variant, tss_variant, run_tag in (
        (
            harness.c_run,
            "qfg_only",
            "tss_control",
            "formal800_qfg_only",
        ),
        (
            harness.d_run,
            "tss_qfg",
            "tss_on",
            "formal800_tss_qfg",
        ),
    ):
        events = []
        for epoch in range(1, 801):
            events.append(
                {
                    "epoch": epoch,
                    "variant": variant,
                    "pd": 188 / 189,
                    "fa": 4e-6,
                    "miou": 0.90 + epoch / 1_000_000,
                    "tiny_pd": 1.0,
                    "new_best_pd": epoch in (1, 800),
                    "new_best_miou": epoch in (1, 400),
                }
            )
        metrics_bytes = b"".join(
            (
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for event in events
        )
        (run_dir / "metrics.jsonl").write_bytes(metrics_bytes)
        journal_metrics = run_dir / "exact_journal/active.metrics.jsonl"
        journal_metrics.write_bytes(metrics_bytes)
        marker = {
            "epoch": 800,
            "metrics_file": journal_metrics.name,
            "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
            "variant": variant,
        }
        (run_dir / "exact_journal/active.json").write_text(
            json.dumps(marker, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        primary = events[799]
        secondary = events[399]
        primary_metrics = {
            name: primary[name] for name in stored_metrics
        }
        secondary_metrics = {
            name: secondary[name] for name in stored_metrics
        }
        source_locks = {
            "tpd_ner_v4_qfg_v2_croa_exact_source_lock": (
                source_lock_sha256
            )
        }
        identity = {
            "variant": variant,
            "dataset": "NUDT-SIRST",
            "seed": 42,
            "split_seed": 20260722,
            "run_id": (
                "tpd-ner-v4-qfg-v2-croa-exact:"
                f"NUDT-SIRST:{variant}:seed-42:split-20260722:{run_tag}"
            ),
            "source_locks": source_locks,
        }
        summary = {
            "schema": "fake_exact_completion_v1",
            "status": "complete",
            "variant": variant,
            "candidate_variant": variant,
            "qfg_variant": "qfg_v2_croa",
            "tss_variant": tss_variant,
            "dataset": "NUDT-SIRST",
            "seed": 42,
            "split_seed": 20260722,
            "formal_contract": formal_contract,
            "stored_validation_metrics": list(stored_metrics),
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
            "best_epoch": 800,
            "best_pd_epoch": 800,
            "best_miou_epoch": 400,
            "best_validation_metrics": primary_metrics,
            "best_pd_validation_metrics": primary_metrics,
            "best_miou_validation_metrics": secondary_metrics,
            "run_identity": identity,
            "source_locks": source_locks,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_static_contract_and_embedded_python_compile() -> None:
    assert FINALIZER.is_file()
    assert not FINALIZER.is_symlink()
    assert os.access(FINALIZER, os.X_OK)
    subprocess.run(["bash", "-n", str(FINALIZER)], check=True)
    source = FINALIZER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(blocks) == 2
    for index, block in enumerate(blocks):
        compile(block, f"<finalizer-heredoc-{index}>", "exec")

    assert "ExactEpochJournal(journal_root).load_active()" in source
    assert "active.epoch != exact.FORMAL_EPOCHS" in source
    assert "exact.COMPLETION_SUMMARY_SCHEMA" in source
    assert "exact._load_complete_events" in source
    assert "require_flags=True" in source
    assert "source_lock_sha256" in source
    assert "run_tpd_ner_v4_qfg_v2_croa_formal800_sweeps_2x5090.sh" in source
    assert "--aggregate" in source
    assert "build_formal_report()" in source
    assert "freeze_tpd_ner_v4_qfg_v2_croa_posttraining_closure.py" in source
    assert "deploy_tpd_ner_v4_qfg_v2_croa_formal800.py" in source
    assert "--closure-source-lock" in source
    assert "--verify" in source
    assert "flock -n" in source
    assert 'exec 9>>"$qfg_finalize_lock"' in source
    for forbidden in (
        "qfg_v1",
        "/home/md0",
        "SCTransNet copy",
        "nvidia-smi",
        "sleep ",
        "GPU 0",
        "GPU 1",
        "gpu0",
        "gpu1",
    ):
        assert forbidden not in source

    assert source.index("qfg_finalize_verify_source_lock") < source.index(
        "qfg_finalize_verify_completion"
    )
    assert source.index('"$qfg_finalize_sweeps_runner"') < source.index(
        "qfg_finalize_factorial_preflight"
    )


def test_embedded_completion_accepts_exact800_and_rejects_summary_drift(
    harness: Harness,
) -> None:
    _install_embedded_completion_fixture(harness)
    completion_override = (
        "TPD_NER_V4_QFG_V2_CROA_FINALIZER_COMPLETION_CHECKER"
    )
    valid = _run(
        harness,
        fail_stage="sweeps_preflight",
        drop_env=(completion_override,),
    )
    assert valid.returncode == 33
    assert valid.stdout.count(
        "TPDNERV4QFG_FINALIZE_RUN_COMPLETE"
    ) == 2
    assert "variant=qfg_only" in valid.stdout
    assert "variant=tss_qfg" in valid.stdout
    assert harness.calls() == [
        "source_verify",
        "closure_verify",
        "sweeps_preflight",
    ]
    _assert_no_terminal_outputs(harness)

    harness.log.write_text("", encoding="utf-8")
    summary_path = harness.d_run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["best_miou_epoch"] = 401
    summary_path.write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    invalid = _run(
        harness,
        fail_stage="sweeps_preflight",
        drop_env=(completion_override,),
    )
    assert invalid.returncode != 0
    assert "summary best mIoU epoch differs" in (
        invalid.stdout + invalid.stderr
    )
    assert harness.calls() == ["source_verify", "closure_verify"]
    _assert_no_terminal_outputs(harness)


def test_real_factorial_and_postprocess_read_only_build_apis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments import compare_tss_qfg_v2_croa_factorial as factorial
    from experiments import (
        postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selection,
    )

    parent_binding = {
        "parent_checkpoint_sha256": "1" * 64,
        "parent_checkpoint_state_dict_sha256": "2" * 64,
        "parent_checkpoint_role": "best_miou",
        "parent_checkpoint_epoch": 489,
    }

    def factorial_point(
        *,
        matched: int,
        unmatched: int,
        miou: float,
        threshold: float,
    ) -> dict[str, object]:
        return {
            "threshold": threshold,
            "matched_target_count": matched,
            "pd": matched / factorial.TARGET_COUNT,
            "fa": 4e-6,
            "miou": miou,
            "tiny_pd": 1.0,
            "unmatched_predicted_object_count": unmatched,
            "false_objects_per_image": (
                unmatched / factorial.VALIDATION_COUNT
            ),
            "target_count": factorial.TARGET_COUNT,
            "tiny_target_count": factorial.TINY_TARGET_COUNT,
            "matched_tiny_target_count": factorial.TINY_TARGET_COUNT,
            "predicted_object_count": matched + unmatched,
            "valid_pixel_count": (
                factorial.VALIDATION_COUNT * 256 * 256
            ),
        }

    records: dict[tuple[str, str], factorial.SweepRecord] = {}
    for arm_index, (arm, spec) in enumerate(factorial.ARM_SPECS.items()):
        run_dir = tmp_path / "factorial" / spec.variant
        for role_index, (checkpoint, role) in enumerate(
            factorial.CHECKPOINT_ROLES.items()
        ):
            fixed = factorial_point(
                matched=184 + arm_index,
                unmatched=4 - arm_index,
                miou=0.90 + 0.005 * arm_index + 0.001 * role_index,
                threshold=0.5,
            )
            budgets = {
                key: factorial_point(
                    matched=184 + arm_index,
                    unmatched=4 - arm_index,
                    miou=float(fixed["miou"]),
                    threshold=0.6 + 0.01 * budget_index,
                )
                for budget_index, key in enumerate(factorial.BUDGET_KEYS)
            }
            records[(arm, checkpoint)] = factorial.SweepRecord(
                arm=arm,
                variant=spec.variant,
                evaluator_family=spec.evaluator_family,
                run_directory=run_dir,
                checkpoint_filename=checkpoint,
                checkpoint_role=role,
                checkpoint_epoch=10 + role_index,
                checkpoint_sha256=hashlib.sha256(
                    f"{arm}:{checkpoint}".encode()
                ).hexdigest(),
                sweep_path=run_dir / f"{checkpoint}.json",
                sweep_sha256=hashlib.sha256(
                    f"sweep:{arm}:{checkpoint}".encode()
                ).hexdigest(),
                validation_split_sha256="3" * 64,
                run_identity={
                    "run_id": f"run-{arm}",
                    "variant": spec.variant,
                },
                checkpoint_identity={
                    "run_id": f"run-{arm}",
                    "variant": spec.variant,
                    **parent_binding,
                },
                parent_binding=dict(parent_binding),
                fixed=fixed,
                budgets=budgets,
            )
    factorial_report = factorial.build_factorial_report(records)
    factorial_markdown = factorial.render_markdown(factorial_report)
    factorial_preflight = factorial.preflight_manifest(records)
    assert factorial_report["status"] == "complete"
    assert factorial_preflight["validated_checkpoint_count"] == 8
    assert set(factorial_report["role_reports"]) == {
        "pd_primary",
        "miou_secondary",
    }
    assert "TSS × QFG-V2-CROA" in factorial_markdown
    json.dumps(
        factorial_report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )

    def selection_point(
        *,
        threshold: float,
        matched: int,
        fa: float,
        miou: float,
        unmatched: int,
    ) -> dict[str, object]:
        return {
            "threshold": threshold,
            "pd": matched / selection.TARGET_COUNT,
            "fa": fa,
            "miou": miou,
            "tiny_pd": 1.0,
            "false_objects_per_image": (
                unmatched / selection.VALIDATION_COUNT
            ),
            "target_count": selection.TARGET_COUNT,
            "matched_target_count": matched,
            "tiny_target_count": selection.TINY_TARGET_COUNT,
            "matched_tiny_target_count": selection.TINY_TARGET_COUNT,
            "unmatched_predicted_object_count": unmatched,
        }

    def selection_method(
        method_id: str,
        *,
        matched: int,
        fa: float,
        miou: float,
        unmatched: int,
    ) -> dict[str, object]:
        roles: dict[str, object] = {}
        for role_name, checkpoint, checkpoint_role in (
            (
                "pd_primary",
                "best.pth.tar",
                "best_validation_pd_primary",
            ),
            (
                "miou_secondary",
                "best_miou.pth.tar",
                "best_validation_miou_secondary",
            ),
        ):
            points = [
                selection_point(
                    threshold=threshold,
                    matched=matched,
                    fa=fa,
                    miou=miou,
                    unmatched=unmatched,
                )
                for threshold in (0.4, 0.5, 0.6)
            ]
            roles[role_name] = {
                "checkpoint": checkpoint,
                "checkpoint_role": checkpoint_role,
                "role_name": role_name,
                "checkpoint_epoch": 10,
                "checkpoint_sha256": "a" * 64,
                "checkpoint_path": f"/fixture/{method_id}/{checkpoint}",
                "run_directory": f"/fixture/{method_id}",
                "fixed_threshold_0_5": copy.deepcopy(points[1]),
                "fa_budget_points": {
                    key: copy.deepcopy(points[1])
                    for key in selection.BUDGET_KEYS
                },
                "raw_points": points,
                "raw_point_count": len(points),
                "sweep_binding": {
                    "path": f"/fixture/{method_id}/{checkpoint}.json",
                    "sha256": "b" * 64,
                },
            }
        return {
            "method_id": method_id,
            "display_name": method_id,
            "variant": method_id,
            "roles": roles,
        }

    method_ids = (
        "baseline",
        "v1",
        "v2",
        "v3",
        "v4",
        "a_control",
        "b_tss",
        "c_qfg_only",
        "d_tss_qfg",
    )
    methods = {
        method_id: selection_method(
            method_id,
            matched=187 + int(
                method_id in {"c_qfg_only", "d_tss_qfg"}
            ),
            fa=5e-6,
            miou=0.90 + 0.01 * int(
                method_id in {"c_qfg_only", "d_tss_qfg"}
            ),
            unmatched=5,
        )
        for method_id in method_ids
    }
    authority = {
        "methods": {
            method_id: methods[method_id]
            for method_id in ("baseline", "v1", "v2", "v3", "v4")
        },
        "bindings": {"authority": {"sha256": "c" * 64}},
        "authority_binding": {
            "path": "/fixture/authority.json",
            "sha256": "c" * 64,
        },
    }
    monkeypatch.setattr(
        selection,
        "validate_frozen_authority",
        lambda: copy.deepcopy(authority),
    )
    monkeypatch.setattr(
        selection,
        "validate_extension_method",
        lambda spec: copy.deepcopy(methods[spec.method_id]),
    )
    monkeypatch.setattr(
        selection,
        "_snapshot_bindings",
        lambda observed, bindings: {
            "synthetic_input": {
                "path": "/fixture/synthetic.json",
                "sha256": "d" * 64,
                "method_count": len(observed),
            },
        },
    )
    monkeypatch.setattr(selection, "verify_snapshot", lambda snapshot: None)
    selection_report = selection.build_formal_report()
    selection_markdown = selection.render_markdown(selection_report)
    assert selection_report["final_model_engineering_selected"]
    assert len(selection_report["methods"]) == 9
    assert "Decision F-F" in selection_markdown
    json.dumps(
        selection_report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def test_source_lock_failure_produces_no_terminal_artifact(
    harness: Harness,
) -> None:
    result = _run(harness, fail_stage="source_verify")
    assert result.returncode != 0
    assert harness.calls() == ["source_verify"]
    _assert_no_terminal_outputs(harness)


def test_posttraining_closure_lock_failure_precedes_completion_and_outputs(
    harness: Harness,
) -> None:
    result = _run(harness, fail_stage="closure_verify")
    assert result.returncode != 0
    assert harness.calls() == ["source_verify", "closure_verify"]
    _assert_no_terminal_outputs(harness)


@pytest.mark.parametrize(
    ("mutation", "expected_calls"),
    (
        ("missing_summary", []),
        (
            "epoch_799",
            [
                "source_verify",
                "closure_verify",
                "completion_qfg_only",
            ],
        ),
    ),
)
def test_incomplete_run_never_reaches_sweeps(
    harness: Harness,
    mutation: str,
    expected_calls: list[str],
) -> None:
    if mutation == "missing_summary":
        harness.c_run.joinpath("summary.json").unlink()
    else:
        marker = harness.c_run / "exact_journal/active.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["epoch"] = 799
        marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    result = _run(harness)
    assert result.returncode != 0
    assert harness.calls() == expected_calls
    assert not any(call.startswith("sweeps") for call in harness.calls())
    _assert_no_terminal_outputs(harness)


@pytest.mark.parametrize(
    ("fail_stage", "forbidden_call"),
    (
        ("sweeps_preflight", "sweeps_run"),
        ("sweeps_run", "factorial_preflight"),
        ("factorial_preflight", "factorial_aggregate"),
        ("factorial_build", "factorial_aggregate"),
        ("postprocess_build", "factorial_aggregate"),
    ),
)
def test_front_failures_publish_no_terminal_outputs(
    harness: Harness,
    fail_stage: str,
    forbidden_call: str,
) -> None:
    result = _run(harness, fail_stage=fail_stage)
    assert result.returncode != 0
    assert forbidden_call not in harness.calls()
    assert "postprocess_publish" not in harness.calls()
    _assert_no_terminal_outputs(harness)


def test_success_order_and_complete_idempotency(harness: Harness) -> None:
    first = _run(harness)
    assert first.returncode == 0, first.stderr
    assert "idempotent=false" in first.stdout
    assert all(path.is_file() and not path.is_symlink() for path in harness.terminal_paths)
    calls = harness.calls()
    for call in (
        "source_verify",
        "closure_verify",
        "completion_qfg_only",
        "completion_tss_qfg",
        "sweeps_preflight",
        "sweeps_run",
        "factorial_preflight",
        "factorial_build",
        "postprocess_build",
        "factorial_aggregate",
        "postprocess_publish",
        "deploy_publish",
        "deploy_verify",
    ):
        assert call in calls
    assert calls.index("source_verify") < calls.index("completion_qfg_only")
    assert calls.index("completion_tss_qfg") < calls.index("sweeps_preflight")
    assert calls.index("sweeps_run") < calls.index("factorial_preflight")
    assert calls.index("factorial_preflight") < calls.index("factorial_build")
    assert calls.index("postprocess_build") < calls.index("factorial_aggregate")
    assert calls.index("factorial_aggregate") < calls.index("postprocess_publish")
    assert calls.index("postprocess_publish") < calls.index("deploy_publish")
    assert calls.index("deploy_publish") < calls.index("deploy_verify")

    before = _snapshot(harness.terminal_paths)
    producer_counts = {
        name: calls.count(name)
        for name in (
            "sweeps_run",
            "factorial_aggregate",
            "postprocess_publish",
            "deploy_publish",
        )
    }
    second = _run(harness)
    assert second.returncode == 0, second.stderr
    assert "idempotent=true" in second.stdout
    assert _snapshot(harness.terminal_paths) == before
    after_calls = harness.calls()
    assert after_calls.count("sweeps_run") == producer_counts["sweeps_run"]
    assert after_calls.count("factorial_aggregate") == producer_counts[
        "factorial_aggregate"
    ]
    assert after_calls.count("postprocess_publish") == producer_counts[
        "postprocess_publish"
    ]
    assert after_calls.count("deploy_publish") == producer_counts[
        "deploy_publish"
    ]
    assert after_calls.count("deploy_verify") == (
        calls.count("deploy_verify") + 1
    )


def test_resume_preserves_valid_factorial_pair(harness: Harness) -> None:
    first = _run(harness)
    assert first.returncode == 0, first.stderr
    factorial_before = _snapshot(
        (harness.factorial_json, harness.factorial_md)
    )
    harness.selection_json.unlink()
    harness.selection_md.unlink()
    calls_before = harness.calls()

    resumed = _run(harness)
    assert resumed.returncode == 0, resumed.stderr
    assert _snapshot((harness.factorial_json, harness.factorial_md)) == (
        factorial_before
    )
    calls_after = harness.calls()
    assert calls_after.count("factorial_aggregate") == calls_before.count(
        "factorial_aggregate"
    )
    assert calls_after.count("postprocess_publish") == (
        calls_before.count("postprocess_publish") + 1
    )


def test_artifact_only_deployment_is_recovered_without_rerunning_gpu_work(
    harness: Harness,
) -> None:
    first = _run(harness)
    assert first.returncode == 0, first.stderr
    harness.deployment_manifest.unlink()
    artifact_before = _snapshot((harness.deployment_artifact,))
    calls_before = harness.calls()
    resumed = _run(harness)
    assert resumed.returncode == 0, resumed.stderr
    assert _snapshot((harness.deployment_artifact,)) == artifact_before
    assert harness.deployment_manifest.is_file()
    calls_after = harness.calls()
    for producer in (
        "sweeps_run",
        "factorial_aggregate",
        "postprocess_publish",
    ):
        assert calls_after.count(producer) == calls_before.count(producer)
    assert calls_after.count("deploy_publish") == (
        calls_before.count("deploy_publish") + 1
    )


def test_existing_report_corruption_is_rejected_without_overwrite(
    harness: Harness,
) -> None:
    assert _run(harness).returncode == 0
    harness.factorial_json.write_text("{}\n", encoding="utf-8")
    before = _snapshot(harness.terminal_paths)
    producer_counts = {
        name: harness.calls().count(name)
        for name in ("sweeps_run", "factorial_aggregate", "postprocess_publish")
    }
    result = _run(harness)
    assert result.returncode != 0
    assert "conflicts with live inputs" in (result.stdout + result.stderr)
    assert _snapshot(harness.terminal_paths) == before
    for name, count in producer_counts.items():
        assert harness.calls().count(name) == count


@pytest.mark.parametrize("state", ("partial_factorial", "orphan_selection"))
def test_inconsistent_terminal_state_is_not_repaired(
    harness: Harness,
    state: str,
) -> None:
    if state == "partial_factorial":
        _write(harness.factorial_json, "{}\n")
    else:
        _write(harness.selection_json, "{}\n")
        _write(harness.selection_md, "orphan\n")
    existing = tuple(
        path for path in harness.terminal_paths if path.exists()
    )
    before = _snapshot(existing)
    result = _run(harness)
    assert result.returncode != 0
    assert _snapshot(existing) == before
    assert "sweeps_run" not in harness.calls()
    for path in harness.terminal_paths:
        if path not in existing:
            assert not path.exists()


def test_preflight_is_read_only(harness: Harness) -> None:
    _seed_candidate_sweeps(harness)
    result = _run(harness, "--preflight")
    assert result.returncode == 0, result.stderr
    assert "writes_performed=false" in result.stdout
    _assert_no_terminal_outputs(harness)
    assert not (harness.result_root / ".finalization_locks").exists()
    assert "sweeps_preflight" in harness.calls()
    assert "sweeps_run" not in harness.calls()
    assert "factorial_preflight" in harness.calls()
    assert "factorial_aggregate" not in harness.calls()
    assert "postprocess_publish" not in harness.calls()


def test_nonblocking_global_lock_returns_retry_without_work(
    harness: Harness,
) -> None:
    lock = (
        harness.result_root
        / ".finalization_locks/formal800_2x5090_finalize.lock"
    )
    lock.parent.mkdir(parents=True)
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(harness)
    assert result.returncode == 75
    assert "finalization_claim_held" in result.stderr
    assert harness.calls() == []
    _assert_no_terminal_outputs(harness)
