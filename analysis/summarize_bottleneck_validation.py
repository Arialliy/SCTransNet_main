#!/usr/bin/env python3
"""Combine frozen-probe diagnostics with image-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
TRANSITIONS = {
    "pool": (("x1", "p1"), ("x2", "p2"), ("x3", "p3")),
    "encoder": (("p1", "x2"), ("p2", "x3"), ("p3", "x4")),
    "stage": (("x1", "x2"), ("x2", "x3"), ("x3", "x4")),
    "embedding": (("x1", "emb1"), ("x2", "emb2"), ("x3", "emb3"), ("x4", "emb4")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "analysis" / "results" / "best_checkpoint_probe_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "image_id": raw["image_id"],
                    "object_index": int(float(raw["object_index"])),
                    "stage": raw["stage"],
                    "source": raw["source"],
                    "area": float(raw["area"]),
                    "percentile": float(raw["percentile"]),
                    "robust_cnr": float(raw["robust_cnr"]),
                }
            )
    return rows


def test_image_ids(dataset: str, excluded: Sequence[str]) -> List[str]:
    split = REPO_ROOT / "datasets" / dataset / "img_idx" / f"test_{dataset}.txt"
    excluded_set = set(excluded)
    return [
        item.strip()
        for item in split.read_text().splitlines()
        if item.strip() and item.strip() not in excluded_set
    ]


def paired_by_image(
    rows: Sequence[Mapping[str, object]],
    before: str,
    after: str,
    source: str,
    tiny_only: bool,
    value_key: str,
) -> Dict[str, np.ndarray]:
    selected: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["source"] != source or row["stage"] not in (before, after):
            continue
        if tiny_only and float(row["area"]) > 9:
            continue
        key = (str(row["image_id"]), int(row["object_index"]))
        selected[key][str(row["stage"])] = float(row[value_key])

    grouped: Dict[str, List[float]] = defaultdict(list)
    for (image_id, _object_index), values in selected.items():
        if before in values and after in values:
            # Positive means the downstream node lost target separability.
            grouped[image_id].append(values[before] - values[after])
    return {image_id: np.asarray(values, dtype=np.float64) for image_id, values in grouped.items()}


def clustered_bootstrap_mean(
    grouped: Mapping[str, np.ndarray],
    all_image_ids: Sequence[str],
    repetitions: int,
    seed: int,
) -> Tuple[int, float, float, float]:
    observed = [grouped[image_id] for image_id in all_image_ids if image_id in grouped]
    if not observed:
        return 0, float("nan"), float("nan"), float("nan")
    point = float(np.mean(np.concatenate(observed)))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    n_images = len(all_image_ids)
    for index in range(repetitions):
        draw = rng.integers(0, n_images, size=n_images)
        arrays = [grouped[all_image_ids[item]] for item in draw if all_image_ids[item] in grouped]
        samples[index] = np.mean(np.concatenate(arrays)) if arrays else np.nan
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return int(sum(item.size for item in observed)), point, float("nan"), float("nan")
    low, high = np.quantile(finite, (0.025, 0.975))
    return int(sum(item.size for item in observed)), point, float(low), float(high)


def fmt(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def main() -> None:
    args = parse_args()
    combined: List[Dict[str, object]] = []
    provenance: Dict[str, object] = {}

    for dataset_index, dataset in enumerate(DATASETS):
        dataset_dir = args.results_dir / dataset
        summary = json.loads((dataset_dir / "summary.json").read_text())
        rows = read_rows(dataset_dir / "per_object.csv")
        image_ids = test_image_ids(dataset, summary["excluded_ids"])
        provenance[dataset] = {
            "checkpoint": summary["checkpoint"],
            "checkpoint_epoch": summary["checkpoint_epoch"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "train_images": summary["train_images"],
            "test_images": summary["test_images"],
            "excluded_ids": summary["excluded_ids"],
            "probe_epoch_losses": summary["probe_epoch_losses"],
        }

        for operation, pairs in TRANSITIONS.items():
            for pair_index, (before, after) in enumerate(pairs):
                ap_before = summary["stages"][before]["probe_ap_sctb_grid_approx"]
                ap_after = summary["stages"][after]["probe_ap_sctb_grid_approx"]
                for subset, tiny_only in (("all", False), ("tiny_le9", True)):
                    probe_grouped = paired_by_image(
                        rows, before, after, "probe", tiny_only, "percentile"
                    )
                    energy_grouped = paired_by_image(
                        rows, before, after, "energy", tiny_only, "percentile"
                    )
                    local_seed = args.seed + dataset_index * 100 + pair_index * 10 + int(tiny_only)
                    n, rank_loss, rank_low, rank_high = clustered_bootstrap_mean(
                        probe_grouped, image_ids, args.bootstrap, local_seed
                    )
                    energy_n, energy_loss, energy_low, energy_high = clustered_bootstrap_mean(
                        energy_grouped, image_ids, args.bootstrap, local_seed + 1
                    )
                    combined.append(
                        {
                            "dataset": dataset,
                            "subset": subset,
                            "operation": operation,
                            "before": before,
                            "after": after,
                            "objects": n,
                            "ap_before": ap_before,
                            "ap_after": ap_after,
                            "ap_loss": ap_before - ap_after,
                            "probe_rank_loss": rank_loss,
                            "probe_rank_ci95": [rank_low, rank_high],
                            "energy_objects": energy_n,
                            "energy_rank_loss": energy_loss,
                            "energy_rank_ci95": [energy_low, energy_high],
                        }
                    )

    payload = {
        "status": "preliminary_engineering_diagnostic_not_publication_result",
        "bootstrap_unit": "test image",
        "bootstrap_repetitions": args.bootstrap,
        "positive_loss_means": "the downstream node has lower target separability",
        "provenance": provenance,
        "transitions": combined,
    }
    output_json = args.results_dir / "combined_summary.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n")

    lines = [
        "# SCTransNet best-checkpoint bottleneck screen",
        "",
        "> Exploratory engineering diagnostic only. Legacy best checkpoints were selected on test mIoU.",
        "",
        "## Checkpoints",
        "",
        "| Dataset | Epoch | Probe losses | Test images | Excluded | SHA256 |",
        "| --- | ---: | --- | ---: | --- | --- |",
    ]
    for dataset in DATASETS:
        item = provenance[dataset]
        losses = ", ".join(f"{value:.6f}" for value in item["probe_epoch_losses"])
        lines.append(
            f"| {dataset} | {item['checkpoint_epoch']} | {losses} | {item['test_images']} | "
            f"{', '.join(item['excluded_ids']) or 'none'} | `{item['checkpoint_sha256'][:12]}…` |"
        )

    for subset in ("tiny_le9", "all"):
        label = "Tiny targets (area <= 9)" if subset == "tiny_le9" else "All targets"
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "Positive loss means worse separability after the transition. Rank intervals use image-clustered bootstrap. AP is the all-target shared-grid estimate (also in the tiny table for context), from one probe seed and without a CI.",
                "",
                "| Dataset | Operation | Pair | N | All-target AP loss | Mean paired probe-rank loss [95% CI] | Mean paired energy-rank loss [95% CI] |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for item in combined:
            if item["subset"] != subset:
                continue
            rank_ci = item["probe_rank_ci95"]
            energy_ci = item["energy_rank_ci95"]
            lines.append(
                f"| {item['dataset']} | {item['operation']} | {item['before']}→{item['after']} | "
                f"{item['objects']} | {fmt(item['ap_loss'])} | {fmt(item['probe_rank_loss'])} "
                f"[{fmt(rank_ci[0])}, {fmt(rank_ci[1])}] | {fmt(item['energy_rank_loss'])} "
                f"[{fmt(energy_ci[0])}, {fmt(energy_ci[1])}] |"
            )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- The probe tests linear decodability on a common stride-16 grid; it is not mutual information.",
            "- A practical first-screen loss is AP > 0.01 plus paired rank loss > 0.05 with a CI above zero in at least two datasets.",
            "- Because this is one backbone seed and one probe seed, failure to cross that gate is inconclusive rather than proof of no loss.",
            "- `NUAA-SIRST` here is the clean213 split after excluding the known image-mask mismatch `Misc_111`; it is not directly comparable with the unclean official split.",
            "",
        ]
    )
    output_md = args.results_dir / "combined_report.md"
    output_md.write_text("\n".join(lines))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
