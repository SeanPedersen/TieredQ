"""Trace learned layouts and reconstruct legacy x86 storage without timing searches."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from threadpoolctl import threadpool_limits
from data import prepare, codec_vectors

HERE = Path(__file__).resolve().parent
METADATA_BYTES = 9
LEVEL_COORDINATES = 4
LOOKUP_BITS = 4
LOOKUP_BLOCK = 256
LEVEL_BLOCK = 16
FLOAT_BYTES = 4
DOUBLE_BYTES = 8
ENCODER_PANEL = 48
CENTROIDS = 256


def round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data")
    parser.add_argument("--native", type=Path, default=HERE / "build-adapters")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("use a fresh output file")
    sys.path.insert(0, str(args.native.resolve()))
    import codec_native
    result = {"seed": 0, "adapter": "original unconditioned L2 lift", "rows": [],
              "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for folder in (HERE.parent / "src", HERE.parent / "include", HERE / "codec_adapters")
                                for p in folder.rglob("*") if p.is_file()}}
    with threadpool_limits(limits=1):
        for dataset in ("sift", "glove", "openai"):
            base, queries, _ = prepare(args.data, dataset, None, 1000)
            vectors, _ = codec_vectors(base, queries, dataset == "sift")
            for bits in (1, 2, 4, 8):
                codec_native.set_threads(8)
                index = codec_native.Index(vectors, bits, 0, True, dataset == "sift")
                layout = index.diagnostics()
                dimensions, count = vectors.shape[1], len(vectors)
                coded = layout["coded_dimensions"]
                lookup = all(tier["bits"] <= LOOKUP_BITS for tier in layout["tiers"])
                if lookup:
                    groups = sum((tier["dimensions"] + LOOKUP_BITS // tier["bits"] - 1) //
                                 (LOOKUP_BITS // tier["bits"]) for tier in layout["tiers"])
                    width, block = (groups + 1) // 2, LOOKUP_BLOCK
                    dimension_padding = None
                else:
                    groups = (coded + LEVEL_COORDINATES - 1) // LEVEL_COORDINATES
                    width, block = groups * LEVEL_COORDINATES, LEVEL_BLOCK
                    dimension_padding = width - coded
                padded = round_up(count, block)
                code_bytes = padded * width
                factor_bytes = padded * 2 * FLOAT_BYTES
                centroid_bytes = padded
                legacy_memory = code_bytes + factor_bytes + centroid_bytes + layout["model_memory_bytes"]
                derived = dict(fused_panels=dimensions * round_up(coded + CENTROIDS, ENCODER_PANEL) * FLOAT_BYTES,
                               centroid_projections=CENTROIDS * round_up(coded, ENCODER_PANEL) * DOUBLE_BYTES,
                               centroid_half_norms=CENTROIDS * (FLOAT_BYTES + DOUBLE_BYTES))
                row = dict(dataset=dataset, bits=bits, base_n=count, dimensions=dimensions, layout=layout,
                           legacy_representation="byte lookup" if lookup else "int8 numerical levels",
                           scan_groups=groups, coded_bytes_per_padded_vector=width,
                           coordinate_padding=dimension_padding, padded_vectors=padded,
                           legacy_codes_bytes=code_bytes, legacy_factor_correction_bytes=factor_bytes,
                           legacy_centroid_bytes=centroid_bytes,
                           legacy_row_padding_bytes=(padded-count)*(width+METADATA_BYTES),
                           shared_derived_encoder_bytes=derived,
                           legacy_effective_bytes_per_vector=legacy_memory/count)
                result["rows"].append(row)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(row), flush=True)
                del index


if __name__ == "__main__":
    main()
