from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from experiments import three_dataset_pbdr_v2_models_seed42_v1 as models
from experiments import train_three_dataset_pbdr_v2_tss_off_seed42_v1 as trainer
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2 import (
    PBDR_V2_INTEGRATION_VERSION,
    PBDR_V2_STATE_KEYS,
)


torch.set_num_threads(1)


class PBDRV2RegistryTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.metadata = models.build_pbdr_v2_training_model(
            "NUAA-SIRST",
            42,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        del cls.metadata

    def test_explicit_runtime_closure_and_training_identity(self) -> None:
        paths = models.runtime_source_paths()
        names = {path.name for path in paths.values()}
        self.assertIn("train_tpd_pilot.py", names)
        self.assertIn(
            "train_three_dataset_pbdr_v2_tss_off_seed42_v1.py",
            names,
        )
        self.assertTrue(all(path.is_file() for path in paths.values()))
        self.assertEqual(len(self.model.state_dict()), 573)
        self.assertEqual(self.metadata["state_key_count"], 573)
        self.assertEqual(
            self.metadata["architecture_manifest"][
                "pbdr_v2_integration_version"
            ],
            PBDR_V2_INTEGRATION_VERSION,
        )

    def test_strip_is_exact_and_inference_strict_loads(self) -> None:
        training_state = self.model.state_dict()
        stripped = models.strip_tss_for_inference_state_dict(training_state)
        self.assertEqual(len(stripped), 569)
        self.assertTrue(set(PBDR_V2_STATE_KEYS) <= set(stripped))
        inference, metadata = (
            models.build_pbdr_v2_inference_model_from_training_state_dict(
                training_state,
                dataset_name="NUAA-SIRST",
            )
        )
        self.assertEqual(len(inference.state_dict()), 569)
        self.assertTrue(metadata["strict_load"])

        invalid = dict(training_state)
        shared_name = next(
            name
            for name in invalid
            if not name.startswith("pbdr_v2.")
            and not name.startswith("target_survival.")
        )
        del invalid[shared_name]
        invalid["pbdr_v2.fake"] = torch.zeros(1)
        self.assertEqual(len(invalid), 573)
        with self.assertRaisesRegex(ValueError, "prefix set differs"):
            models.strip_tss_for_inference_state_dict(invalid)

    def test_trainer_contract_and_current_state_resume_rejection(self) -> None:
        args = trainer.parse_args(
            [
                "--dataset",
                "NUAA-SIRST",
                "--method",
                "final",
                "--device",
                "cpu",
                "--smoke",
                "--epochs",
                "1",
                "--begin-test",
                "1",
                "--eval-every",
                "1",
                "--batch-size",
                "1",
                "--workers",
                "0",
                "--max-train-images",
                "1",
                "--max-test-images",
                "1",
            ]
        )
        trainer.validate_args(args)
        identity = trainer.recipe_identity(args)
        self.assertEqual(identity["recipe_id"], models.RECIPE_ID)
        self.assertEqual(identity["pbdr_v2_parameter_count"], 19)
        self.assertEqual(identity["pbdr_v2_state_key_count"], 5)

        current_state = {
            name: value
            for name, value in self.model.state_dict().items()
            if name not in PBDR_V2_STATE_KEYS
        }
        self.assertEqual(len(current_state), 568)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)
        architecture_id, _ = trainer._architecture_binding(self.model)
        payload = {
            "schema": trainer.SCHEMA,
            "dataset": args.dataset,
            "method": trainer.METHOD,
            "seed": trainer.TRAINING_SEED,
            "protocol_sha256": "unit-test-protocol",
            "recipe": identity,
            "architecture_id": architecture_id,
            "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": models.TRAINING_STATE_KEY_COUNT,
            "planned_total_epochs": args.epochs,
            "state_dict": current_state,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pth.tar"
            torch.save(payload, path)
            with self.assertRaisesRegex(
                trainer.PBDRV2TrainingProtocolError,
                "573-key",
            ):
                trainer._load_resume_pbdr_v2(
                    args=args,
                    path=path,
                    model=self.model,
                    optimizer=optimizer,
                    device=torch.device("cpu"),
                    protocol_sha256="unit-test-protocol",
                )


if __name__ == "__main__":
    unittest.main()
