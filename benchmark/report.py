"""Generate measured-budget plots and operating-point comparisons without interpolation."""
import csv
import json
from pathlib import Path
import sys
from compare import operating_points, compare
from plotting import plot_results


def report(root):
    metadata = json.loads((root / 'meta.json').read_text())
    if metadata.get('status') == 'invalidated':
        raise ValueError('Refusing to report invalidated benchmark measurements')
    with (root / 'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    plot_results(root, rows)
    points = operating_points(rows)
    (root / 'summary.json').write_text(json.dumps({'operating_points': points, 'comparisons': compare(points)}, indent=2))


if __name__ == '__main__':
    report(Path(sys.argv[1]))
