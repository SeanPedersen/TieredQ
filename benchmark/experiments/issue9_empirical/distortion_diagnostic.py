"""Issue #9: does the encoded coordinate distribution deviate from Gaussian enough for empirical distortion to matter?

Per rank band and codebook, prints empirical / Gaussian distortion for raw PCA coordinates, shape-gain (row-RMS
normalized, as the encoder quantizes) and after random rotations within 16- or 64-rank tiers.
Usage: uv run --with numpy python distortion_diagnostic.py src/tieredq_train_codebook_data.h DATA_DIR
"""
import re, sys, numpy as np
src = open(sys.argv[1]).read()
def book(start):
    body = src[src.index(start):]; body = body[:body.index("}")]
    return np.array([float.fromhex(t) if "0x" in t else float(t) for t in re.findall(r"-?0x[0-9a-f.]+p[+-]?\d+", body)])
books = {1: [-0.7978718280792236, 0.7978972792625427], 2: [-1.5103706121444702, -0.45274654030799866, 0.452783465385437, 1.5104025602340698]}
blocks = re.findall(r"\{\s*\n(.*?)\n\s*\}", src[src.index("books{{"):src.index("additive_three_codebook")], re.S)
books[3] = sorted(float.fromhex(t) for t in re.findall(r"-?0x[0-9a-f.]+p[+-]?\d+", blocks[0]))
books[4] = [float(t) for t in re.findall(r"-?\d+\.\d+", src[src.index("{-2.73216"):src.index("2.732205629348755f}")+20])]
books["3a"] = book("static const std::vector<float> book{")
def qerr(z, levels):
    levels = np.asarray(levels); th = (levels[1:]+levels[:-1])/2
    return (z - levels[np.searchsorted(th, z)])**2
rng = np.random.default_rng(0)
gz = rng.standard_normal(2_000_000)
gauss = {b: qerr(gz, l).mean() for b, l in books.items()}
for name, d in [("glove",200),("sift",128),("openai",1536),("gemini",3072)]:
    x = np.fromfile(f"{sys.argv[2]}/{name}/base.f32", np.float32).reshape(-1, d)
    x = x[rng.permutation(len(x))[:16384]].astype(np.float64)
    c = x[rng.choice(len(x), 256, replace=False)]
    for _ in range(8):
        ids = np.argmin((c**2).sum(1)[None] - 2*x@c.T, 1)
        for k in range(256):
            m = ids == k
            if m.any(): c[k] = x[m].mean(0)
    r = x - c[ids]; r -= r.mean(0)
    w, V = np.linalg.eigh(r.T @ r / len(r)); V = V[:, ::-1]; w = w[::-1]
    z = (r @ V) / np.sqrt(np.maximum(w, 1e-30))
    rms = np.sqrt((z**2).mean(1, keepdims=True))
    print(f"== {name}  rows={len(z)}  row-rms p5/p50/p95 = {np.percentile(rms,[5,50,95]).round(2)}")
    print("  ranks   b   gauss  raw/g  shape-gain/g  rot16/g  rot64/g   (variance-weighted)")
    for lo, hi in [(0, d//8), (d//8, d//2), (d//2, d)]:
        sl = slice(lo, hi); wt = w[sl] / w[sl].sum()
        for b in [1, 2, "3a", 4]:
            raw = (qerr(z[:, sl], books[b]).mean(0) * wt).sum()
            sg = ((rms**2 * qerr(z[:, sl]/rms, books[b])).mean(0) * wt).sum()
            rot = []
            for width in (16, 64):
                errs = []
                for s in range(lo, hi, width):
                    e = min(hi, s+width); q, _ = np.linalg.qr(rng.standard_normal((e-s, e-s)))
                    y = (r @ V[:, s:e]) @ q; sd = y.std(0); y /= sd
                    errs.append((qerr(y, books[b]).mean(0) * sd**2).sum())
                rot.append(sum(errs) / w[sl].sum())
            print(f"  {lo:4d}-{hi:<4d} {b!s:>3} {gauss[b]:.4f}  {raw/gauss[b]:.3f}  {sg/gauss[b]:.3f}        {rot[0]/gauss[b]:.3f}    {rot[1]/gauss[b]:.3f}")
