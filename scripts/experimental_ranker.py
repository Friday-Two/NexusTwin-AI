import numpy as np
import torch


def rank_global_shortlist(r, query: str, top_k: int = 20) -> list[dict]:
    """Experimental shortlist reranking using Hybrid's global score scales."""
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    top_k = max(1, min(int(top_k), 20))
    shortlist_size = min(len(r.corpus), max(100, top_k * 10))

    neural_results = r.rank_neural_only(query, top_k=shortlist_size)
    id_to_index = {api_id: i for i, api_id in enumerate(r.api_ids)}
    indices = [id_to_index[x["document_id"]] for x in neural_results]

    # Global neural percentiles, matching the production rank() method.
    query_embedding = r.encoder.encode(
        query, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)

    scores_all = []
    with torch.inference_mode():
        for start in range(0, len(r.embeddings), 512):
            end = min(start + 512, len(r.embeddings))
            docs = torch.from_numpy(r.embeddings[start:end]).to(r.device if hasattr(r, "device") else next(r.model.parameters()).device)
            queries = torch.from_numpy(
                np.repeat(query_embedding[None, :], end - start, axis=0)
            ).to(docs.device)
            features = torch.cat([queries, docs, torch.abs(queries - docs)], dim=1)
            scores_all.extend(r.model(features).squeeze(-1).cpu().numpy().tolist())

    neural_scores = np.asarray(scores_all, dtype=np.float32)
    order = np.argsort(neural_scores, kind="stable")
    percentiles = np.empty(len(neural_scores), dtype=np.float32)
    percentiles[order] = np.linspace(0.0, 1.0, len(neural_scores))

    metadata_scores = np.asarray(
        [r._metadata_score(query, meta) for meta in r.metadata],
        dtype=np.float32,
    )
    low, high = float(metadata_scores.min()), float(metadata_scores.max())
    metadata_norm = (
        (metadata_scores - low) / (high - low)
        if high > low else np.zeros_like(metadata_scores)
    )

    final_scores = np.asarray([
        0.80 * percentiles[i] + 0.20 * metadata_norm[i]
        for i in indices
    ])

    positions = np.argsort(final_scores, kind="stable")[::-1][:top_k]
    results = []
    for pos in positions:
        i = indices[int(pos)]
        row = r.corpus.iloc[i]
        results.append({
            "document_id": r.api_ids[i],
            "description": str(row["document_content"]),
            "api_text": str(row["api_text"]),
            "ranking_score": float(final_scores[int(pos)]),
        })
    return results
