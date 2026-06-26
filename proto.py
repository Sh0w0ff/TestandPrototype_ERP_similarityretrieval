"""proto.py — command-line entry for the three pipeline functions.

    python proto.py make-database [--rebuild]
    python proto.py train [--epochs 120] [--scope eval|deploy|both]
    python proto.py infer STEM_OR_PDF [--k 5] [--mode test|deploy] [--pool auto|all|train]
"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prototype import pipeline as P


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("make-database").add_argument("--rebuild", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=P.WP.EPOCHS)
    t.add_argument("--scope", choices=["eval", "deploy", "both"], default="both")
    i = sub.add_parser("infer")
    i.add_argument("drawing"); i.add_argument("--k", type=int, default=5)
    i.add_argument("--mode", choices=["test", "deploy"], default="test")
    i.add_argument("--pool", choices=["auto", "all", "train"], default="auto",
                   help="retrieval pool override (default auto = match mode: test->train, deploy->all)")
    args = ap.parse_args()

    if args.cmd == "make-database":
        P.make_database(rebuild=args.rebuild)
    elif args.cmd == "train":
        P.train(epochs=args.epochs, scope=args.scope)
    elif args.cmd == "infer":
        pool = None if args.pool == "auto" else args.pool
        r = P.infer(args.drawing, k=args.k, mode=args.mode, pool=pool)
        print(f"\nDRAWING {r['stem']}  (in_corpus={r['in_corpus']}, mode={r['mode']}, "
              f"model={r['scope']}, pool={r['pool']})")
        print("similar drawings (pooled union):")
        for v in r["visual"]:
            print(f"  {v['stem'][:42]:44s} src={v['source']:8s} score={v['score']:.3f}")
        print("work-phases (classifier):", [f"{p}({pr:.2f})" for p, pr in r["workphases"]])
        print("BOM pool:", [c for c, _ in r["bom_pool"]][:12])
        if r.get("truth"):
            print("TRUE phases:", r["truth"]["phases"])
            print("scoring   :", {k: (round(v, 3) if isinstance(v, float) else v)
                                    for k, v in r["scoring"].items()})


if __name__ == "__main__":
    main()
