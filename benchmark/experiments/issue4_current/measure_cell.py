"""Keep run.py headline measurements; separately measure native query preparation afterward."""
import json
import ctypes as ct
import os
from pathlib import Path
import sys
import tempfile
import time

B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B))
import numpy as np
import measure


def main():
    captured=[]
    original=measure.measure
    def timed(index,queries,truth,threads,batch,k=10):
        captured[:]=[index,queries]
        return original(index,queries,truth,threads,batch,k)
    measure.measure=timed
    import run
    run.main()
    cell=Path(sys.argv[sys.argv.index('--cell')+1])
    w=cell.parent.parent
    out=w/'preparation'
    out.mkdir(exist_ok=True)
    rows=json.loads(cell.read_text())
    revision=(w/'source/SOURCE_REVISION').read_text().strip()
    for row in rows:
        row['source_revision']=revision
        row['cpu_affinity']=sorted(os.sched_getaffinity(0))
        if row['codec']=='tieredq':
            row['variant']='fine'
            row['fine_tiers_requested']=True
    cell.write_text(json.dumps(rows))
    index,queries=captured
    diagnostics=index.diagnostics() if rows[0]['codec']=='tieredq' else None
    if rows[0]['codec']=='qdrant':
        import qdrant_codec
        previous=os.environ.get('QDRANT_TQ_LIB')
        os.environ['QDRANT_TQ_LIB']=str(w/'preparation-qdrant/target/release/libqdrant_bench.so')
        library=qdrant_codec._library()
        if previous is None:
            del os.environ['QDRANT_TQ_LIB']
        else:
            os.environ['QDRANT_TQ_LIB']=previous
        library.tq_prepare_only.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_size_t]
        library.tq_prepare_only.restype=None
        class DiagnosticQdrant(qdrant_codec.QdrantIndex):
            def prepare_only(self,queries,threads):
                library.tq_prepare_only(self.pointer,queries.ctypes.data,len(queries),threads)
        DiagnosticQdrant.library=library
        load=lambda path:DiagnosticQdrant.load(path)
    else:
        sys.path.insert(0,str(w/'preparation-build'))
        import codec_preparation
        load=lambda path:codec_preparation.PreparationIndex.load(str(path),rows[0]['codec']=='tieredq')
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        path=Path(tmp)/'index.bin'
        index.save(str(path))
        diagnostic=load(path)
    expected=np.empty((32,10),dtype=np.uint32)
    actual=np.empty_like(expected)
    index.search(queries[:32],expected,1)
    diagnostic.search(queries[:32],actual,1)
    np.testing.assert_array_equal(expected,actual)
    results=[]
    for row in rows:
        threads,batch=row['threads'],row['batch']
        for _ in range(3):
            diagnostic.prepare_only(queries[:batch],threads)
        total=0.
        for _ in range(3):
            for start in range(0,len(queries),batch):
                tick=time.perf_counter()
                diagnostic.prepare_only(queries[start:start+batch],threads)
                total+=time.perf_counter()-tick
        ms=total*1000/(3*len(queries))
        results.append(dict(threads=threads,batch=batch,preparation_ms_per_query=ms,
                            fraction_of_end_to_end=ms*float(row['qps'])/1000,
                            end_to_end_mean_ms_per_query=1000/float(row['qps'])))
    code=int(sys.argv[sys.argv.index('--code-bits')+1]) if '--code-bits' in sys.argv else 0
    (out/cell.name).write_text(json.dumps(dict(dataset=rows[0]['dataset'],codec=rows[0]['codec'],
        base_n=rows[0]['base_n'],queries=len(queries),
        index_diagnostics=diagnostics,
        target_bits=rows[0]['target_bits'],code_bits=code,available=True,
        search_ids_equal=True,method='Separate native preparation-only diagnostic, including call overhead; three warmups and repeats',
        schedules=results),indent=2))


if __name__=='__main__':
    main()
