from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from experiments import compute_tpd_survival_target_statistics as statistics


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ARTIFACT = REPO_ROOT / statistics.DEFAULT_OUTPUT_RELATIVE


def _write_split(path: Path, identifiers: list[str]) -> None:
    digest = statistics.ordered_identifier_sha256(identifiers)
    payload = {
        "dataset": statistics.DATASET,
        "full_internal_train_count": len(identifiers),
        "full_internal_train_ids": identifiers,
        "hashes": {
            "full_internal_train_sha256": digest,
            "used_train_sha256": digest,
        },
        "official_test_accessed": False,
        "split_seed": 7,
        "used_train_count": len(identifiers),
        "used_train_ids": identifiers,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class TargetSurvivalStatisticsTests(unittest.TestCase):
    def test_import_is_silent_and_has_no_artifact_side_effect(self) -> None:
        before = FORMAL_ARTIFACT.read_bytes() if FORMAL_ARTIFACT.exists() else None
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            importlib.reload(statistics)
        after = FORMAL_ARTIFACT.read_bytes() if FORMAL_ARTIFACT.exists() else None
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(after, before)

    def test_max_pool_presence_is_exact_at_cell_boundaries(self) -> None:
        width = 48
        height = 32
        values = bytearray(width * height)
        values[15 * width + 16] = 128
        values[31 * width + 47] = 255
        values[0] = 127
        grid = statistics.max_pool_presence_grid(
            bytes(values),
            width=width,
            height=height,
            downsample=16,
        )
        self.assertEqual(
            grid,
            (
                (False, True, False),
                (False, False, True),
            ),
        )

    def test_write_once_or_verify_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "statistics.json"
            payload = {"schema": "test", "value": 1}
            self.assertEqual(
                statistics.publish_or_verify(output, payload),
                "created",
            )
            expected = statistics.canonical_json_bytes(payload)
            self.assertEqual(output.read_bytes(), expected)
            self.assertEqual(
                statistics.publish_or_verify(output, payload),
                "verified_existing",
            )
            with self.assertRaises(statistics.TargetStatisticsError):
                statistics.publish_or_verify(
                    output,
                    {"schema": "test", "value": 2},
                )
            self.assertEqual(output.read_bytes(), expected)

    def test_nonformal_fixture_counts_exact_presence_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split.json"
            masks = root / "masks"
            masks.mkdir()
            identifiers = ["000001", "000002"]
            _write_split(split, identifiers)

            first = Image.new("L", (256, 256), 0)
            first.putpixel((0, 0), 255)
            first.putpixel((255, 255), 255)
            first.save(masks / "000001.png")
            second = Image.new("L", (256, 256), 0)
            second.putpixel((16, 15), 128)
            second.putpixel((17, 15), 127)
            second.save(masks / "000002.png")

            payload = statistics.compute_statistics(
                repo_root=root,
                split_json=split,
                masks_dir=masks,
                validate_formal=False,
            )
            self.assertEqual(payload["train_image_count"], 2)
            self.assertEqual(payload["positive_cells"], 3)
            self.assertEqual(payload["total_cells"], 512)
            self.assertEqual(payload["negative_cells"], 509)
            self.assertEqual(payload["survival_pos_weight"], 509 / 3)
            self.assertEqual(payload["image_sizes"], [[256, 256]])

    def test_formal_frozen_data_and_artifact_match(self) -> None:
        payload = statistics.compute_statistics(
            repo_root=REPO_ROOT,
            validate_formal=True,
        )
        self.assertEqual(payload["train_image_count"], 530)
        self.assertEqual(payload["positive_cells"], 1313)
        self.assertEqual(payload["negative_cells"], 134367)
        self.assertEqual(payload["total_cells"], 135680)
        self.assertEqual(
            payload["survival_pos_weight"],
            102.33587204874334,
        )
        self.assertEqual(
            payload["used_train_ids_sha256"],
            "9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b",
        )
        self.assertTrue(payload["validation"]["all_masks_256x256"])
        self.assertTrue(FORMAL_ARTIFACT.is_file())
        self.assertEqual(
            statistics.verify_artifact(FORMAL_ARTIFACT, payload),
            statistics.artifact_sha256(payload),
        )

    def test_cli_verify_is_explicit_and_machine_readable(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / statistics.GENERATOR_RELATIVE),
                "--repo-root",
                str(REPO_ROOT),
                "--verify",
                str(FORMAL_ARTIFACT),
            ],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        status = json.loads(completed.stdout)
        self.assertEqual(status["status"], "verified")
        self.assertEqual(status["positive_cells"], 1313)
        self.assertEqual(status["negative_cells"], 134367)
        self.assertEqual(status["total_cells"], 135680)
        self.assertEqual(status["output"], statistics.DEFAULT_OUTPUT_RELATIVE)


if __name__ == "__main__":
    unittest.main()
