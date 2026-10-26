"""Add phase and table-build diagnostics to a separate source copy only."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tiny_budgets'))
from instrument import instrument

root=Path(sys.argv[1]).resolve()
instrument(root)
header=root/'benchmark/experiments/native/profile.h'
header.write_text(header.read_text().replace('merge, count','merge, tables, count'))
path=root/'src/tieredq_scan_lookup_x86.h'
text=path.read_text()
old='  query.tables.assign('
assert text.count(old)==1
text=text.replace(old,'  experiment_profile::Phase tables_phase(experiment_profile::tables);\n'+old)
path.write_text(f'#include "{header}"\n'+text)
