from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from experiments.pbdr_v4_source_lock import (
    PBDRV4SourceLockError,
    build_source_lock,
    load_source_lock,
    validate_source_lock,
    write_source_lock_exclusive,
)


_ENV = {
    "python": "synthetic",
    "torch": "synthetic",
    "cuda_runtime": None,
    "cudnn": None,
    "cuda_available": False,
    "deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


class PBDRV4SourceLockTests(unittest.TestCase):
    def test_build_write_and_replay_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "b.md").write_text("b\n", encoding="utf-8")
            first = build_source_lock(
                source_root=root,
                source_relative_paths=("a.py", "b.md"),
                environment=_ENV,
            )
            second = build_source_lock(
                source_root=root,
                source_relative_paths=("b.md", "a.py"),
                environment=_ENV,
            )
            self.assertEqual(first, second)
            path = root / "lock.json"
            write_source_lock_exclusive(path, first)
            loaded = load_source_lock(path, check_environment=False)
            self.assertEqual(loaded, first)
            with self.assertRaisesRegex(PBDRV4SourceLockError, "exists"):
                write_source_lock_exclusive(path, first)

    def test_changed_source_or_external_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            external = root / "split.json"
            source.write_text("old\n", encoding="utf-8")
            external.write_text("{}\n", encoding="utf-8")
            lock = build_source_lock(
                source_root=root,
                source_relative_paths=("source.py",),
                external_files={"split": external},
                environment=_ENV,
            )
            source.write_text("new\n", encoding="utf-8")
            with self.assertRaisesRegex(PBDRV4SourceLockError, "bytes differ"):
                validate_source_lock(lock, check_environment=False)

    def test_canonical_or_tf32_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("x\n", encoding="utf-8")
            lock = build_source_lock(
                source_root=root,
                source_relative_paths=("source.py",),
                environment=_ENV,
            )
            changed = dict(lock)
            changed["tf32_disabled"] = False
            with self.assertRaisesRegex(PBDRV4SourceLockError, "canonical"):
                validate_source_lock(changed, check_environment=False)

    def test_symlink_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.py"
            target.write_text("x\n", encoding="utf-8")
            (root / "source.py").symlink_to(target)
            with self.assertRaisesRegex(PBDRV4SourceLockError, "symlink"):
                build_source_lock(
                    source_root=root,
                    source_relative_paths=("source.py",),
                    environment=_ENV,
                )


if __name__ == "__main__":
    unittest.main()
