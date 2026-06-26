"""app.py — Streamlit UI for the drawing retrieval + ERP proposal prototype.

    streamlit run app.py

Pick a drawing from the corpus (or upload a new PDF), choose Test or Deploy mode, and the app shows:
  - the query drawing and its pooled-union similar drawings (mean-pool + self-attention encoders)
  - the content extracted from the drawing's own text (title-block fields + internal BOM table)
  - the ERP proposal: work-phase classifier prediction + BOM candidate pool
  - in Test mode: the drawing's true ERP data and a correctness readout
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import streamlit as st

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from prototype import pipeline as P
from prototype import engine as E

st.set_page_config(page_title="Drawing → ERP prototype", layout="wide")


@st.cache_resource(show_spinner="Loading database…")
def _db():
    return P.make_database(verbose=False)


def _thumb_path(stem):
    p = E.paths.VISUAL_PIPE / stem / "polygon_removed_super.png"
    return str(p) if p.exists() else None


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Drawing → ERP")
db = _db()
st.sidebar.caption(f"{db['n']} drawings in the retrieval pool · "
                   f"classifier {'trained' if db['trained'] else 'NOT trained'}")
if not db["trained"]:
    st.sidebar.warning("Run `python proto.py train` to enable work-phase prediction.")

mode = st.sidebar.radio("Mode", ["test", "deploy"],
                        format_func=lambda x: {"test": "Test (held-out evaluation)",
                                               "deploy": "Deploy (new drawing)"}[x],
                        help="test = eval model (train split) + train-only retrieval + ground truth · "
                             "deploy = all-data model + whole-corpus retrieval + proposal only")
pool = st.sidebar.radio("Retrieval pool", ["auto", "all", "train"],
                        format_func=lambda x: {"auto": "Auto (match mode)", "all": "All data",
                                               "train": "Train split only"}[x],
                        help="override the pool the mode would pick (test->train, deploy->all)")
k = st.sidebar.slider("Neighbours per encoder (k)", 3, 10, 5)

src = st.sidebar.radio("Input", ["Pick a corpus drawing", "Upload a new PDF"])
query = None
if src == "Pick a corpus drawing":
    stems = sorted(db["stems"])
    default = stems.index("3AUA0000038918") if "3AUA0000038918" in stems else 0
    query = st.sidebar.selectbox("Drawing", stems, index=default)
else:
    up = st.sidebar.file_uploader("Drawing PDF", type=["pdf"])
    if up is not None:
        dest = REPO / "scratch" / up.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(up.read())
        query = str(dest)
        st.sidebar.info("New PDF will run the full pipeline (slower: view-seg + embedding).")

run = st.sidebar.button("Run", type="primary", use_container_width=True)


# --------------------------------------------------------------------------- main
st.title("Retrieval and ERP proposal from a 2D drawing")

if run and query:
    with st.spinner("Running inference…"):
        r = P.infer(query, k=k, mode=mode, pool=(None if pool == "auto" else pool))

    st.subheader(f"Query: {r['stem']}")
    st.caption(f"model: **{r['scope']}** ({'train-split' if r['scope']=='eval' else 'all-data'})  ·  "
               f"retrieval pool: **{r['pool']}**")
    qpath = _thumb_path(r["stem"])
    cols = st.columns([1, 2])
    with cols[0]:
        if qpath:
            st.image(qpath, caption="query drawing", use_container_width=True)
    with cols[1]:
        content = r["content"] or {}
        fields = content.get("fields", {})
        st.markdown("**Extracted content** (from the drawing's own text)")
        GROUPS = [
            ("Identity",      [("part_name", "name"), ("part_type", "type")]),
            ("Material",      [("material", "material"), ("material_class", "class"), ("coating", "coating")]),
            ("Manufacturing", [("general_tolerance", "tolerance"), ("scale", "scale"), ("weight_kg", "weight (kg)")]),
            ("Standards",     [("standards", "standards")]),
        ]
        groups, labels, values = [], [], []
        for gname, rowspec in GROUPS:
            for key, lab in rowspec:
                v = fields.get(key)
                if not v:
                    continue
                groups.append(gname); labels.append(lab)
                values.append(", ".join(map(str, v)) if isinstance(v, list) else str(v))
        if labels:
            st.table({"group": groups, "field": labels, "value": values})
        ib = content.get("internal_bom", [])
        if ib:
            st.caption(f"Internal on-drawing BOM table: {len(ib)} cells — {', '.join(ib[:8])}")

    # ---- ERP proposal ----
    st.markdown("---")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**Work-phase proposal** (classifier)")
        if r["workphases"]:
            st.table({"production line": [p for p, _ in r["workphases"]],
                      "confidence": [f"{pr:.2f}" for _, pr in r["workphases"]]})
        else:
            st.caption("classifier unavailable (train it first)")
    with ec2:
        st.markdown("**BOM candidate pool** (from similar drawings)")
        if r["bom_pool"]:
            st.table({"component": [c for c, _ in r["bom_pool"]],
                      "neighbour votes": [n for _, n in r["bom_pool"]]})
        else:
            st.caption("no recurring components among the neighbours (long-tail BOM)")

    # ---- ground truth (test mode) ----
    if r.get("truth"):
        s = r["scoring"]
        st.markdown("---")
        st.markdown("**Ground truth** (test mode)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("phase F1", f"{s['phase_f1']:.2f}")
        m2.metric("phase exact", str(s["phase_exact"]))
        m3.metric("true phases", len(r["truth"]["phases"]))
        m4.metric("BOM pool recall", "n/a" if s["bom_pool_recall"] is None else f"{s['bom_pool_recall']:.2f}")
        st.caption("true work-phases: " + ", ".join(r["truth"]["phases"]))

    # ---- similar drawings ----
    st.markdown("---")
    st.markdown(f"**Similar drawings** — pooled union (top-{k} per encoder, source-tagged)")
    grid = st.columns(5)
    for i, v in enumerate(r["visual"]):
        with grid[i % 5]:
            tp = _thumb_path(v["stem"])
            if tp:
                st.image(tp, use_container_width=True)
            st.caption(f"`{v['source']}`  {v['score']:.3f}\n\n{v['stem'][:34]}")
else:
    st.info("Choose a drawing in the sidebar and press **Run**.")
