"""ctypes contract for full compressed scoring, with timed native PCA preprocessing."""
import ctypes as ct
from pathlib import Path
import time
import numpy as np
HERE=Path(__file__).resolve().parent
LIB=ct.CDLL(str(HERE/'build/libsaq_bench.so'))
LIB.saq_build.argtypes=[ct.c_void_p,ct.c_size_t,ct.c_size_t,ct.c_void_p,ct.c_void_p,ct.c_float,ct.c_uint,ct.c_char_p]
LIB.saq_build.restype=ct.c_void_p
LIB.saq_search.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_void_p,ct.c_size_t,ct.c_bool]
LIB.saq_stats.argtypes=[ct.c_void_p,ct.c_void_p]
LIB.saq_free.argtypes=[ct.c_void_p]


class SAQ:
    def __init__(self,base,basis,mean,bits,seed,path):
        tick=time.perf_counter()
        self.pointer=LIB.saq_build(base.ctypes.data,len(base),base.shape[1],basis.ctypes.data,mean.ctypes.data,bits,seed,str(path).encode())
        assert self.pointer
        self.build_s=time.perf_counter()-tick
        values=np.empty(2,dtype=np.float64)
        LIB.saq_stats(self.pointer,values.ctypes.data)
        self.resident_bytes=int(values[0])
        self.encode_seconds=values[1]
        self.disk_bytes=path.stat().st_size+basis.nbytes+mean.nbytes
        self.bytes_per_vector=max(self.resident_bytes,self.disk_bytes)/len(base)
        self.fullscan=True

    def search(self,queries,output,threads):
        assert queries.dtype==np.float32 and queries.flags.c_contiguous
        assert output.dtype==np.uint32 and output.flags.c_contiguous and output.shape==(len(queries),10)
        LIB.saq_search(self.pointer,queries.ctypes.data,len(queries),output.ctypes.data,threads,self.fullscan)

    def close(self):
        if self.pointer:LIB.saq_free(self.pointer);self.pointer=None
