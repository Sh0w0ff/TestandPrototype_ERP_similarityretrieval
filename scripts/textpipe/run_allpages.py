"""Driver: build the all-pages (drawing-level UNION) signal for every stem that already has a
page-1 signal.json. Writes signal_allpages.json ALONGSIDE signal.json (page-1 baseline untouched),
so the all-pages-vs-page-1 E1 ablation reads either file. RESUMABLE: skips stems whose
signal_allpages.json already exists, so an interrupted run can just be re-launched.

Stem names carry spaces/semicolons (KC), so we enumerate the text_pipe dir in Python rather than
passing 1827 names on the command line. PaddleOCR loads once in this process (KC later pages are
path-rendered -> OCR), so keep it a single long-running process.

Run:  python scripts/textpipe/run_allpages.py
"""
import sys, time
from pathlib import Path

sys.argv = [sys.argv[0], "--all-pages"]          # make build_signal pick up ALL_PAGES at import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_signal as B

stems = sorted(d.name for d in B.OUT.iterdir()
               if d.is_dir() and d.name != "images" and (d / "signal.json").exists())
todo = [s for s in stems if not (B.OUT / s / "signal_allpages.json").exists()]
print(f"{len(stems)} stems with page-1 signal; {len(todo)} to build "
      f"({len(stems) - len(todo)} already done) -> {B.SIG_NAME}", flush=True)

t0 = time.time()
ok = err = 0
for i, s in enumerate(todo, 1):
    try:
        B.build(s)
        ok += 1
    except Exception as e:
        err += 1
        import traceback; traceback.print_exc()
        print(f"  ERR {s[:48]}: {type(e).__name__}: {e}", flush=True)
    if i % 25 == 0 or i == len(todo):
        dt = time.time() - t0
        rate = i / dt if dt else 0
        eta = (len(todo) - i) / rate if rate else 0
        print(f"[{i}/{len(todo)}] ok={ok} err={err} {rate:.2f}/s ETA {eta/60:.1f}m", flush=True)

print(f"DONE ok={ok} err={err} in {(time.time()-t0)/60:.1f}m", flush=True)
