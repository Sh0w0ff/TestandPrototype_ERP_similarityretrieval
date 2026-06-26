"""train_simcse.py — UNSUPERVISED SimCSE (Gao et al. 2021) domain-adaptation of a sentence encoder
on our cleaned drawing-text corpus (textsim.build_corpus). Label-free: each drawing text is its own
positive (two dropout views), other in-batch texts are negatives (MultipleNegativesRankingLoss).
Never sees BOM/work-phase or the human similarity judgments -> downstream rerank eval is leak-free.

Run: /opt/anaconda3/envs/fyp/bin/python scripts/part1/train_simcse.py [--base MODEL] [--epochs N]
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "part1"))
import textsim


def _arg(f, d):
    return sys.argv[sys.argv.index(f) + 1] if f in sys.argv else d


BASE = _arg("--base", "all-MiniLM-L6-v2")
EPOCHS = int(_arg("--epochs", 3))
OUT = ROOT / "models" / "simcse_drawing"


def main():
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    corpus = textsim.build_corpus()
    texts = [t for t in corpus.values() if len(t.strip()) >= 15]   # skip near-empty drawings
    print(f"SimCSE: {len(texts)} drawing texts (of {len(corpus)})  base={BASE}  epochs={EPOCHS}")
    examples = [InputExample(texts=[t, t]) for t in texts]         # unsupervised: same text twice

    model = SentenceTransformer(BASE, device="cpu")
    loader = DataLoader(examples, shuffle=True, batch_size=64, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup = int(0.1 * len(loader) * EPOCHS)
    model.fit(train_objectives=[(loader, loss)], epochs=EPOCHS, warmup_steps=warmup,
              optimizer_params={"lr": 3e-5}, show_progress_bar=True)
    OUT.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT))
    print(f"saved SimCSE model -> {OUT}")


if __name__ == "__main__":
    main()
