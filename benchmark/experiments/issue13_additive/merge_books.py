"""Issue #13: keep the lowest-distortion additive book per width across additive_mse.py runs; write LEVELS_FILE and JSON.

Usage: merge_books.py OUT_PREFIX BOOKS.json... (only widths above the nibble limit go to LEVELS_FILE)
"""
import json
import sys

NIBBLE_BITS = 4


def main(prefix, *paths):
    best = {}
    for path in paths:
        for name, book in json.load(open(path)).items():
            bits, form = name.split("-")
            if form == "free" and (bits not in best or book["distortion"] < best[bits]["distortion"]):
                best[bits] = book
    with open(prefix + ".levels", "w") as levels:
        for bits, book in sorted(best.items(), key=lambda item: int(item[0])):
            ratio = book["distortion"] / book["lloyd"]
            print(f"{bits} bits: {book['hi_bits']}+{int(bits) - book['hi_bits']} distortion {book['distortion']:.7f} "
                  f"lloyd {book['lloyd']:.7f} ratio {ratio:.3f}")
            if int(bits) > NIBBLE_BITS:
                levels.write(bits + " " + " ".join(f"{level:.9g}" for level in book["levels"]) + "\n")
    json.dump(best, open(prefix + ".json", "w"))


if __name__ == "__main__":
    main(*sys.argv[1:])
