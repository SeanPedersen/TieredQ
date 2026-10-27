"""Independently validate wrapper top-k with upstream full distance enumeration."""
import ctypes as ct
import csv
import json
import os
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run import HERE,load,save,sha
LIB=ct.CDLL(str(HERE/'build/libsaq_reference.so'))
LIB.saq_reference.argtypes=[ct.c_char_p,ct.c_void_p,ct.c_size_t,ct.c_size_t,ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p]
LIB.saq_last_block_regression.restype=ct.c_uint32
LIB.saq_layout.argtypes=[ct.c_char_p,ct.c_void_p]
CHECK_QUERIES=10


def main():
    assert LIB.saq_last_block_regression()==63
    report=dict(status='running',last_full_block_regression=True,reference_binary_sha256=sha(HERE/'build/libsaq_reference.so'),checks=[])
    for dataset in ('gemini','glove','openai','sift'):
        base,queries,truth=load(dataset)
        del base
        with np.load(HERE/f'pca-{dataset}.npz') as a:basis,mean=a['basis'],a['mean']
        probes=np.ascontiguousarray(queries[np.linspace(0,len(queries)-1,CHECK_QUERIES,dtype=int)])
        for bits in (1,2,4):
            for seed in (0,1,2):
                directory=HERE/f'{dataset}-saq{bits}'
                model=directory/f'model-seed{seed}.bin'
                output=np.empty((CHECK_QUERIES,10),dtype=np.uint32)
                layout=np.zeros(256,dtype=np.uint64)
                LIB.saq_layout(str(model).encode(),layout.ctypes.data)
                LIB.saq_reference(str(model).encode(),probes.ctypes.data,len(probes),int(layout[1]),basis.ctypes.data,mean.ctypes.data,queries.shape[1],output.ctypes.data)
                expected=np.load(directory/f'ids-seed{seed}.npy')[np.linspace(0,len(queries)-1,CHECK_QUERIES,dtype=int)]
                assert np.array_equal(output,expected),(dataset,bits,seed)
                record=dict(dataset=dataset,bits=bits,seed=seed,queries=CHECK_QUERIES,ids_equal=True,
                    padded_dimensions=int(layout[1]),quant_plan=[list(map(int,layout[4+2*i:6+2*i])) for i in range(int(layout[0]))],
                    quantize_config_bytes=int(layout[2]),single_config_bytes=int(layout[3]))
                save(directory/f'layout-seed{seed}.json',record)
                report['checks'].append(record)
                print('REFERENCE VERIFIED',dataset,bits,seed,flush=True)
                save(HERE/'native-reference-audit.json',report)
    report['status']='passed';save(HERE/'native-reference-audit.json',report)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
