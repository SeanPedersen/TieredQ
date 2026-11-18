# Third pass: FP32 lists tiled like SQ8; TieredQ IVF-Flat (FP32, SQ8) vs pynndescent, rinnd and manifolds-rs.
set -e
cd "$(dirname "$0")"
cmake --build cbuild -j 16 --target test_tieredq > cbuild-libs.log
OMP_NUM_THREADS=8 cbuild/test_tieredq > ctest-libs.log 2>&1
grep "flat IVF" ctest-libs.log
uv sync --reinstall-package tieredq > sync-libs.log 2>&1
IVF=ivfflat-L128-p32,ivfflat-sq8-L128-p32,ivfflat-L256-p16,ivfflat-sq8-L256-p16,ivfflat-L512-p32,ivfflat-sq8-L512-p32
LIBS=pynndescent,pynndescent-n30,rinnd,rinnd-n30,rinnd-t16-i20-c30,mfrs-nndescent,mfrs-ivf,mfrs-ivf-p2x,mfrs-ivf-p4x
uv run python compare.py ../data/dbpedia100k.npy ../results/dbpedia100k-libs --suite 100k --only $IVF,$LIBS
IVF=ivfflat-L512-p8,ivfflat-sq8-L512-p8,ivfflat-L1024-p16,ivfflat-sq8-L1024-p16,ivfflat-L1024-p32,ivfflat-sq8-L1024-p32
LIBS=pynndescent,pynndescent-t16-i20-c30,rinnd,rinnd-t16-i20-c30,rinnd-n30,mfrs-nndescent,mfrs-ivf
uv run python compare.py ../data/msmarco1m.npy ../results/msmarco1m-libs --suite 1m --truth-rows 10000 --seeds 0 \
  --only $IVF,$LIBS
echo DONE
