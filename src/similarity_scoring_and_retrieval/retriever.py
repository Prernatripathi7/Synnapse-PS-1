# src/similarity_scoring_and_retrieval/retriever.py
import numpy as np
import faiss

class FaissRetriever:
    """
    Cosine similarity retrieval using FAISS (IndexFlatIP with L2-normalized vectors).
    """
    def __init__(self, emb_path, ids_path, normalize=True):
        self.emb = np.load(emb_path).astype("float32")
        self.ids = np.load(ids_path, allow_pickle=True)
        self.normalize = normalize

        if self.emb.ndim != 2:
            raise ValueError(f"Embeddings must be 2D (N,D). Got {self.emb.shape}")

        if self.normalize:
            faiss.normalize_L2(self.emb)

        D = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(D)
        self.index.add(self.emb)

    def search(self, q_emb, k=5, extra=50):
        q_emb = np.asarray(q_emb, dtype="float32")
        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]

        if self.normalize:
            faiss.normalize_L2(q_emb)

       search_k = min(self.emb.shape[0], k + extra)
       scores, idxs = self.index.search(q_emb, search_k)
       top_ids = self.ids[idxs[0]]
       return top_ids, scores[0], idxs[0]
