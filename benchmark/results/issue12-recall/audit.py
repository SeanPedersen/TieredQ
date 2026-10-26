"""Audit the delivered matrix, source snapshot, mode parity and legacy storage reconstruction."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
DATASETS = {"sift": (128, 1_000_000), "glove": (200, 100_000), "openai": (1536, 100_000)}


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audit_scan_snapshot():
    path = ROOT / "scan-opt/as-run-sources.tar.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "8d84670f36a585fa33b803f0a393acb194fd8bdcb5ce979c5b285a6d230a15e9"
    metadata = json.loads((ROOT / "scan-opt/final/meta.json").read_text())
    with tarfile.open(path) as archive:
        for name, digest in metadata["source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name


def audit_storage_refresh():
    report = json.loads((ROOT / "storage-refresh/results.json").read_text())
    assert report["status"] == "complete" and not report["validation_only"]
    budgets = {"sift": (2, 4, 8), "glove": (4, 5, 6, 8), "openai": (2, 3, 4, 8)}
    expected = {(dataset, bits, seed) for dataset, points in budgets.items() for bits in points for seed in (0, 1, 2)}
    keys = [(row["dataset"], row["bits"], row["seed"]) for row in report["rows"]]
    assert len(keys) == len(expected) and set(keys) == expected
    manifests = json.loads((ROOT / "data-manifest.json").read_text())
    for dataset, manifest in report["data_manifests"].items():
        assert manifest["base_sha256"] == manifests[dataset]["base_sha256"]
    for row in report["rows"]:
        assert row["base_n"] == DATASETS[row["dataset"]][1]
        layout = row["diagnostics"]
        assert row["effective_index_bytes"] == max(row["persisted_index_bytes"],
            layout["allocated_vector_bytes"] + layout["model_serving_memory_bytes"])
        assert row["effective_bytes_per_vector"] == (row["effective_index_bytes"] + row["adapter_bytes"]) / row["base_n"]
        assert (row["adapter_bytes"] > 0) == (row["dataset"] == "sift")
    for point in report["summary"]:
        rows = [row for row in report["rows"] if (row["dataset"], row["bits"]) == (point["dataset"], point["bits"])]
        assert point["seeds"] == [0, 1, 2]
        assert point["effective_bytes_per_vector"] == max(row["effective_bytes_per_vector"] for row in rows)


def main():
    audit_scan_snapshot()
    audit_storage_refresh()
    rows = read_rows(ROOT / "results.csv")
    keys = [(r["dataset"], r["codec"], int(r["target_bits"]), int(r["seed"])) for r in rows]
    expected = {(dataset, codec, bits, seed) for dataset in DATASETS
                for codec in ("tieredq", "rabitq") for bits in (4, 8) for seed in (0, 1, 2)}
    assert len(keys) == len(expected) and set(keys) == expected
    for row in rows:
        dimensions, count = DATASETS[row["dataset"]]
        assert int(row["d"]) == dimensions and int(row["base_n"]) == count
        assert int(row["queries"]) == 1000 and int(row["threads"]) == 8 and int(row["batch"]) == 32
        assert 0 <= float(row["recall10"]) <= 1 and float(row["effective_bytes_per_vector"]) > 0
        if row["seed"] == "0":
            assert float(row["qps"]) > 0 and float(row["p95_ms"]) >= float(row["p50_ms"]) > 0
        if row["codec"] == "rabitq":
            assert row["config"] == "flat-fht-native-bounds-hacc"
    meta = json.loads((ROOT / "meta.json").read_text())
    assert meta["status"] == "complete" and meta["cpu_affinity"] == list(range(8))
    assert meta["rabitq_search"]["extra_bit_bounds"] and meta["rabitq_search"]["high_accuracy"]
    with tarfile.open(ROOT / "as-run-sources.tar.gz") as archive:
        for path, digest in meta["source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest, path
        for name, digest in meta["harness_sha256"].items():
            assert hashlib.sha256(archive.extractfile("benchmark/" + name).read()).hexdigest() == digest, name
    normals = json.loads((ROOT / "normal-reference-final.json").read_text())["rows"]
    for normal in normals:
        primary = next(r for r in rows if r["dataset"] == normal["dataset"] and r["codec"] == "tieredq"
                       and r["target_bits"] == "8" and r["seed"] == "0")
        assert abs(float(primary["recall10"]) - normal["recall10"]) < 1e-12
        repeated = json.loads((ROOT / "repeat" / f"{normal['dataset']}.json").read_text())[0]
        assert repeated["recall10"] == float(primary["recall10"])
    repetition = json.loads((ROOT / "repeat/provenance.json").read_text())
    with tarfile.open(ROOT / "repeat/as-run-sources.tar.gz") as archive:
        for path, digest in repetition["source_sha256"].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == digest, path
    original = read_rows(ROOT.parent / "hartmut-full-single-run-20261023/output-full-single-run/results.csv")
    layouts = json.loads((ROOT / "layouts.json").read_text())["rows"]
    assert len(layouts) == 12
    for layout in layouts:
        point = next(r for r in original if r["dataset"] == layout["dataset"] and r["codec"] == "tieredq"
                     and int(r["target_bits"]) == layout["bits"])
        assert abs(float(point["effective_bytes_per_vector"]) - layout["legacy_effective_bytes_per_vector"]) < 1e-9
    phases = read_rows(ROOT / "preprocessing.csv")
    assert len(phases) == 12
    for phase in phases:
        if phase["codec"] == "tieredq":
            assert float(phase["scan_worker_ms"]) > 0 and float(phase["selection_worker_ms"]) > 0
    diagnostic = json.loads((ROOT / "full-score/meta.json").read_text())
    assert diagnostic["status"] == "complete" and not diagnostic["native_bounds"]
    full_score = read_rows(ROOT / "full-score/results.csv")
    assert len(full_score) == 3 and {r["dataset"] for r in full_score} == set(DATASETS)
    assert all(r["config"] == "flat-fht-full-score-hacc" and r["seed"] == "0" and r["target_bits"] == "8"
               and float(r["qps"]) > 0 for r in full_score)
    result = dict(status="passed", primary_rows=len(rows), seeds=[0,1,2], queries=1000,
                  native_bounds_primary=True, archived_source_and_harness_hashes=True,
                  normal_fast_recall_parity=True, repeated_seed0_recall=True,
                  reconstructed_original_storage_points=len(layouts), phase_profiles=len(phases),
                  full_score_diagnostic_separate=True, recovered_scan_snapshot_hashes=True,
                  corrected_storage_rows=33)
    (ROOT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
