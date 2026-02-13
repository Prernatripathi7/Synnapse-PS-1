import numpy as np
import faiss

class FaissRetriever:
    """
    Cosine similarity retrieval using FAISS.
    """

    def __init__(self, emb_path, ids_path, normalize=True):
        self.emb = np.load(emb_path).astype("float32")
        self.ids = np.load(ids_path, allow_pickle=True)
        self.normalize = normalize

        if self.normalize:
            faiss.normalize_L2(self.emb)

        D = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(D)
        self.index.add(self.emb)

    def search(self, q_emb, k=5):
        q_emb = np.asarray(q_emb, dtype="float32")

        if q_emb.ndim == 1:
            q_emb = q_emb[None, :]

        if self.normalize:
            faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(q_emb, k)

        return self.ids[idxs[0]], scores[0], idxs[0]
