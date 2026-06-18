"""One-shot repair: add graph nodes missing from canonical_keywords.json.

Run once after applying the canonicalizer fix. Embeds missing nodes,
appends them, and rebuilds canonical_embeddings.npy + keyword_index.faiss.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.knowledge_graph.io import build_keyword_index


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-dir", default="data/knowledge_graph/runs/latest")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent.parent
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir

    graph_nodes = {
        n["id"]
        for n in json.loads((run_dir / "graph.json").read_text())["nodes"]
    }
    canonical_kws: list[str] = json.loads((run_dir / "canonical_keywords.json").read_text())
    canonical_emb: np.ndarray = np.load(run_dir / "canonical_embeddings.npy")

    missing = sorted(graph_nodes - set(canonical_kws))
    print(f"Graph nodes:        {len(graph_nodes)}")
    print(f"Canonical keywords: {len(canonical_kws)}")
    print(f"Missing:            {len(missing)}")

    if not missing:
        print("Nothing to repair.")
        return

    model = SentenceTransformer(args.embed_model)
    print(f"Embedding {len(missing)} missing keywords with {args.embed_model}...")
    missing_emb: np.ndarray = model.encode(missing, show_progress_bar=True)

    # Build updated sorted list and aligned embedding matrix
    kw_to_emb: dict[str, np.ndarray] = {kw: canonical_emb[i] for i, kw in enumerate(canonical_kws)}
    for kw, vec in zip(missing, missing_emb):
        kw_to_emb[kw] = vec

    updated_kws = sorted(kw_to_emb.keys())
    updated_emb = np.stack([kw_to_emb[kw] for kw in updated_kws], axis=0).astype(np.float32)

    print(f"Updated canonical keywords: {len(updated_kws)}")
    print(f"Updated embeddings shape:   {updated_emb.shape}")

    (run_dir / "canonical_keywords.json").write_text(
        json.dumps(updated_kws, indent=2, ensure_ascii=False)
    )
    np.save(run_dir / "canonical_embeddings.npy", updated_emb)
    build_keyword_index(updated_emb, str(run_dir))

    print("Saved canonical_keywords.json, canonical_embeddings.npy, keyword_index.faiss")

    # Verify
    still_missing = graph_nodes - set(updated_kws)
    print(f"Verification — still missing: {len(still_missing)}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
