import gzip
import shutil
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "index")
emb_p, jl, gz = d / "embeddings.npy", d / "chunks.jsonl", d / "chunks.jsonl.gz"
if jl.exists():
    with open(jl, "rb") as src, gzip.open(gz, "wb", compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)
    jl.unlink()
emb = np.load(emb_p)
if emb.dtype != np.float16:
    np.save(emb_p, emb.astype(np.float16))
for f in sorted(d.iterdir()):
    print(f"{f.name:<22} {f.stat().st_size / 1e6:8.1f} MB")