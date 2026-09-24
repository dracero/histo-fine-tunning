"""
FAISS Semantic Similarity Labeler for Histology.

Provides a pure-embedding pipeline for labeling cells by semantic similarity:
1. A pathologist marks a prototype cell and assigns a label (e.g., "Espermatogonia A Clara").
2. The system extracts embeddings for all segmented detections using pathology
   foundation models (Virchow2 1280d primary, UNI 1024d fallback).
3. A FAISS index enables fast cosine-similarity search.
4. All detections above the similarity threshold receive the same label.
5. Multi-prototype support: mark N examples of the same class and average their
   embeddings for a more robust centroid.

No LLM (Gemini/GPT) is used at any stage.  Classification is purely based on
visual embedding distance.
"""

import difflib
import gc
import logging
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import faiss
except ImportError:
    faiss = None  # Will raise clear error on use

try:
    from backend.pathology_models import (
        VirchowModelWrapper,
        UniModelWrapper,
        ConchModelWrapper,
        extract_crops_from_detections,
    )
except ImportError:
    from pathology_models import (
        VirchowModelWrapper,
        UniModelWrapper,
        ConchModelWrapper,
        extract_crops_from_detections,
    )

logger = logging.getLogger("sam3-backend.faiss-similarity")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Prototype:
    """A user-marked prototype: one or more detection indices sharing a label."""

    label: str
    color: str
    detection_indices: List[int] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None  # L2-normalised centroid embedding


@dataclass
class SimilarityResult:
    """Result of a FAISS similarity search for a single prototype."""

    prototype_label: str
    prototype_color: str
    matches: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core labeler
# ---------------------------------------------------------------------------

class FAISSSimilarityLabeler:
    """
    FAISS-powered semantic similarity labeler optimised for histology.

    The labeler holds:
    - A matrix of L2-normalised embeddings (one row per detection).
    - A ``faiss.IndexFlatIP`` for exact cosine-similarity search.
    - A registry of user-defined prototypes.

    Typical lifecycle:
        labeler = FAISSSimilarityLabeler()
        labeler.build_index(image, detections)
        labeler.add_prototype("Espermatogonia A", "#e11d48", [42])
        results = labeler.search_similar("Espermatogonia A", threshold=0.85)
        labeled = labeler.propagate_all_labels(detections, threshold=0.85)
    """

    # Supported embedding models ordered by preference
    MODEL_PRIORITY = ("virchow", "uni", "conch")

    def __init__(
        self,
        primary_model: str = "virchow",
        batch_size: int = 32,
    ) -> None:
        if faiss is None:
            raise ImportError(
                "faiss-cpu (or faiss-gpu) is required.  "
                "Install with: pip install faiss-cpu>=1.7.4"
            )

        if primary_model not in self.MODEL_PRIORITY:
            raise ValueError(
                f"Unknown model '{primary_model}'.  "
                f"Choose from {self.MODEL_PRIORITY}"
            )

        self.primary_model = primary_model
        self.batch_size = batch_size

        # State — populated by build_index()
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float32, L2-norm
        self._index: Optional[faiss.Index] = None
        self._dim: int = 0
        self._num_detections: int = 0
        self._prototypes: Dict[str, Prototype] = {}  # label -> Prototype
        self._crops: List[Image.Image] = []
        self._detections: List[Dict[str, Any]] = []

    def get_crop(self, detection_index: int) -> Optional[Image.Image]:
        """Return the extracted PIL crop for a given detection index, if available."""
        if 0 <= detection_index < len(self._crops):
            return self._crops[detection_index]
        return None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
        *,
        margin_ratio: float = 0.85,
        min_crop_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Extract embeddings for *all* detections and build the FAISS index.

        Returns a summary dict with timing and embedding metadata.
        """
        if not detections:
            return {
                "success": True,
                "total_detections": 0,
                "embedding_dim": 0,
                "index_build_time_ms": 0,
            }

        t0 = time.time()

        # 1. Crop detections from the source image
        crops = extract_crops_from_detections(
            image,
            detections,
            margin_ratio=margin_ratio,
            min_size=min_crop_size,
        )

        # 2. Encode crops with the preferred foundation model
        embeddings_tensor = self._encode_crops(crops)

        # 3. L2-normalise and convert to float32 numpy (FAISS requirement)
        embeddings_tensor = F.normalize(embeddings_tensor.float(), dim=-1)
        embeddings_np: np.ndarray = embeddings_tensor.cpu().numpy().astype(np.float32)

        self._embeddings = embeddings_np
        self._dim = embeddings_np.shape[1]
        self._num_detections = embeddings_np.shape[0]
        self._crops = crops
        self._detections = detections

        # 4. Build FAISS IndexFlatIP (cosine sim on L2-normed vectors)
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings_np)

        # 5. Clear GPU cache after heavy encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"FAISS index built: {self._num_detections} vectors x {self._dim}d "
            f"in {elapsed_ms:.0f}ms (model={self.primary_model})"
        )

        return {
            "success": True,
            "total_detections": self._num_detections,
            "embedding_dim": self._dim,
            "model_used": self.primary_model,
            "index_build_time_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # Prototype management
    # ------------------------------------------------------------------

    def _normalize_str(self, s: str) -> str:
        """Strip accents, lower-case and normalize whitespace for resilient matching."""
        if not s:
            return ""
        s_norm = unicodedata.normalize("NFKD", s)
        s_clean = "".join(c for c in s_norm if not unicodedata.combining(c))
        return " ".join(s_clean.strip().lower().split())

    def get_prototype(self, label: str) -> Optional[Prototype]:
        """
        Lookup prototype with resilient multi-stage matching:
        1. Exact match
        2. Normalized match (case-insensitive, diacritic-insensitive, trimmed whitespace)
        3. Fuzzy match (difflib) for minor typos (e.g. 'espermatogonai' -> 'Espermatogonia A Clara')
        """
        if not label:
            return None

        # 1. Exact match
        if label in self._prototypes:
            return self._prototypes[label]

        # 2. Normalized match
        target_norm = self._normalize_str(label)
        norm_map = {self._normalize_str(k): k for k in self._prototypes.keys()}
        if target_norm in norm_map:
            return self._prototypes[norm_map[target_norm]]

        # 3. Fuzzy match for typos
        candidates = list(norm_map.keys())
        if candidates:
            close = difflib.get_close_matches(target_norm, candidates, n=1, cutoff=0.72)
            if close:
                matched_key = norm_map[close[0]]
                logger.info(
                    f"Fuzzy matched prototype query '{label}' -> registered '{matched_key}'"
                )
                return self._prototypes[matched_key]

        return None

    def add_prototype(
        self,
        label: str,
        color: str,
        detection_indices: List[int],
    ) -> Dict[str, Any]:
        """
        Register one or more detections as a named prototype.

        If a prototype with the same *label* already exists its detection
        indices are extended (multi-prototype accumulation) and the centroid
        is recomputed.
        """
        self._require_index()

        # Validate indices
        valid_indices = [
            i for i in detection_indices
            if 0 <= i < self._num_detections
        ]
        if not valid_indices:
            return {
                "success": False,
                "error": f"No valid detection indices provided. Index has {self._num_detections} detections (received {detection_indices}).",
            }

        clean_label = label.strip()
        existing_proto = self.get_prototype(clean_label)

        if existing_proto is not None:
            proto = existing_proto
            # Avoid duplicate indices
            current_set = set(proto.detection_indices)
            new_indices = [i for i in valid_indices if i not in current_set]
            proto.detection_indices.extend(new_indices)
            proto.color = color  # allow colour update
            self._prototypes[clean_label] = proto
        else:
            proto = Prototype(label=clean_label, color=color, detection_indices=list(valid_indices))
            self._prototypes[clean_label] = proto

        # Recompute centroid as the L2-normalised mean of all prototype embeddings
        proto_embeddings = self._embeddings[proto.detection_indices]  # (K, D)
        centroid = proto_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        proto.centroid = centroid

        logger.info(
            f"Prototype '{clean_label}': {len(proto.detection_indices)} examples, "
            f"centroid norm={np.linalg.norm(proto.centroid):.4f}"
        )

        return {
            "success": True,
            "label": clean_label,
            "color": color,
            "num_examples": len(proto.detection_indices),
        }

    def remove_prototype(self, label: str) -> bool:
        """Remove a registered prototype by label."""
        clean_label = label.strip()
        proto = self.get_prototype(clean_label)
        if proto is not None:
            keys_to_remove = [k for k, p in self._prototypes.items() if p.label == proto.label or k == clean_label]
            removed = False
            for k in keys_to_remove:
                if self._prototypes.pop(k, None) is not None:
                    removed = True
            return removed
        return self._prototypes.pop(clean_label, None) is not None

    def list_prototypes(self) -> List[Dict[str, Any]]:
        """Return metadata for all registered prototypes."""
        seen_labels = set()
        res = []
        for p in self._prototypes.values():
            if p.label in seen_labels:
                continue
            seen_labels.add(p.label)
            res.append({
                "label": p.label,
                "color": p.color,
                "num_examples": len(p.detection_indices),
                "detection_indices": p.detection_indices,
            })
        return res

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        label: str,
        *,
        threshold: float = 0.80,
        top_k: int = 0,
    ) -> SimilarityResult:
        """
        Search for detections similar to the prototype identified by *label*.

        Args:
            label: Name of a registered prototype.
            threshold: Minimum cosine similarity (0-1) to consider a match.
            top_k: If >0, return at most this many matches (ordered by
                   similarity).  0 means return all above threshold.

        Returns:
            ``SimilarityResult`` with matched detection indices and scores.
        """
        self._require_index()

        proto = self.get_prototype(label.strip())
        if proto is None or proto.centroid is None:
            available = list({p.label for p in self._prototypes.values()})
            if available:
                raise ValueError(
                    f"Prototype '{label}' not found. Clases registradas disponibles: {available}. "
                    f"Verifica el nombre o regístralo primero con 'Registrar Prototipo'."
                )
            raise ValueError(
                f"Prototype '{label}' not found. No hay ninguna clase registrada aún. "
                f"Selecciona una célula y haz clic en 'Registrar Prototipo' primero."
            )

        # FAISS search -- query is the prototype centroid (1, D)
        query = proto.centroid.reshape(1, -1).astype(np.float32)
        k_search = self._num_detections  # search all
        distances, indices = self._index.search(query, k_search)

        matches: List[Dict[str, Any]] = []
        proto_set = set(proto.detection_indices)

        for rank in range(k_search):
            det_idx = int(indices[0, rank])
            sim_score = float(distances[0, rank])

            if sim_score < threshold:
                # Since IndexFlatIP returns results in descending similarity,
                # once we drop below threshold all subsequent will be lower.
                break

            matches.append({
                "detection_index": det_idx,
                "similarity": round(sim_score, 4),
                "is_prototype": det_idx in proto_set,
            })

        if top_k > 0:
            matches = matches[:top_k]

        return SimilarityResult(
            prototype_label=proto.label,
            prototype_color=proto.color,
            matches=matches,
        )

    def search_similar_by_index(
        self,
        detection_index: int,
        *,
        threshold: float = 0.80,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Ad-hoc similarity search using a single detection embedding as query
        (no prototype registration required).

        Useful for "click and find similar" one-shot exploration.
        """
        self._require_index()

        if detection_index < 0 or detection_index >= self._num_detections:
            raise IndexError(f"detection_index {detection_index} out of range [0, {self._num_detections})")

        query = self._embeddings[detection_index].reshape(1, -1).astype(np.float32)
        k_search = min(top_k if top_k > 0 else self._num_detections, self._num_detections)
        distances, indices = self._index.search(query, k_search)

        matches: List[Dict[str, Any]] = []
        for rank in range(k_search):
            det_idx = int(indices[0, rank])
            sim_score = float(distances[0, rank])
            if sim_score < threshold:
                break
            matches.append({
                "detection_index": det_idx,
                "similarity": round(sim_score, 4),
                "is_query": det_idx == detection_index,
            })

        return matches

    # ------------------------------------------------------------------
    # Label propagation
    # ------------------------------------------------------------------

    def propagate_label(
        self,
        label: str,
        detections: List[Dict[str, Any]],
        *,
        threshold: float = 0.80,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Propagate a single prototype's label to all similar detections.

        Mutates ``detections`` in-place by setting ``category_name``, ``color``,
        ``similarity_score``, and ``labeled_by`` on each matched detection.

        Returns (detections, num_labeled).
        """
        clean_label = label.strip()
        result = self.search_similar(clean_label, threshold=threshold)
        proto = self.get_prototype(clean_label)
        if proto is None:
            proto = self._prototypes.get(clean_label) or self._prototypes[list(self._prototypes.keys())[0]]

        num_labeled = 0
        for m in result.matches:
            idx = m["detection_index"]
            if 0 <= idx < len(detections):
                detections[idx]["category_name"] = proto.label
                detections[idx]["color"] = proto.color
                detections[idx]["similarity_score"] = m["similarity"]
                detections[idx]["labeled_by"] = "faiss_similarity"
                detections[idx]["is_prototype"] = m.get("is_prototype", False)
                num_labeled += 1

        return detections, num_labeled

    def propagate_all_labels(
        self,
        detections: List[Dict[str, Any]],
        *,
        threshold: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Propagate labels from ALL registered prototypes.

        When a detection matches multiple prototypes, the one with the
        highest similarity wins (winner-takes-all).

        Returns a summary dict.
        """
        self._require_index()

        if not self._prototypes:
            return {
                "success": True,
                "total_labeled": 0,
                "total_unlabeled": len(detections),
                "labels": {},
            }

        # Build score matrix: for each detection, record the best matching
        # prototype label and score.
        best_label: List[Optional[str]] = [None] * len(detections)
        best_score: List[float] = [-1.0] * len(detections)

        for label in self._prototypes:
            result = self.search_similar(label, threshold=threshold)
            for m in result.matches:
                idx = m["detection_index"]
                if 0 <= idx < len(detections) and m["similarity"] > best_score[idx]:
                    best_score[idx] = m["similarity"]
                    best_label[idx] = label

        # Apply labels
        label_counts: Dict[str, int] = {}
        total_labeled = 0
        for idx in range(len(detections)):
            lbl = best_label[idx]
            if lbl is not None:
                proto = self._prototypes[lbl]
                detections[idx]["category_name"] = lbl
                detections[idx]["color"] = proto.color
                detections[idx]["similarity_score"] = round(best_score[idx], 4)
                detections[idx]["labeled_by"] = "faiss_similarity"
                detections[idx]["is_prototype"] = idx in set(proto.detection_indices)
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                total_labeled += 1

        return {
            "success": True,
            "total_labeled": total_labeled,
            "total_unlabeled": len(detections) - total_labeled,
            "labels": label_counts,
            "detections": detections,
        }

    # ------------------------------------------------------------------
    # Auto-clustering (unsupervised)
    # ------------------------------------------------------------------

    def auto_cluster(
        self,
        n_clusters: int = 5,
        *,
        max_iter: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Run k-means clustering on the embedding index to produce *suggested*
        groups without any user-labeled prototypes.

        Uses FAISS' built-in k-means for speed and consistency.

        Returns a list of cluster dicts with member indices and centroid info.
        """
        self._require_index()

        if self._num_detections < n_clusters:
            n_clusters = max(1, self._num_detections)

        kmeans = faiss.Kmeans(
            d=self._dim,
            k=n_clusters,
            niter=max_iter,
            verbose=False,
            gpu=False,
            seed=42,
        )
        kmeans.train(self._embeddings)

        # Assign each detection to its nearest centroid
        _, assignments = kmeans.index.search(self._embeddings, 1)
        assignments = assignments.flatten()

        palette = [
            "#e11d48", "#8b5cf6", "#06b6d4", "#f59e0b", "#10b981",
            "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16",
        ]

        clusters: List[Dict[str, Any]] = []
        for c_id in range(n_clusters):
            member_indices = np.where(assignments == c_id)[0].tolist()
            if not member_indices:
                continue

            centroid = kmeans.centroids[c_id]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Compute intra-cluster cohesion (mean similarity to centroid)
            member_embeddings = self._embeddings[member_indices]
            sims = member_embeddings @ centroid_norm
            mean_sim = float(sims.mean())

            clusters.append({
                "cluster_id": c_id,
                "suggested_label": f"Grupo {c_id + 1}",
                "color": palette[c_id % len(palette)],
                "count": len(member_indices),
                "detection_indices": member_indices,
                "mean_similarity": round(mean_sim, 4),
            })

        # Sort by count descending
        clusters.sort(key=lambda c: c["count"], reverse=True)

        logger.info(
            f"Auto-cluster: {n_clusters} clusters over {self._num_detections} "
            f"detections, sizes={[c['count'] for c in clusters]}"
        )

        return clusters

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def get_embedding(self, detection_index: int) -> Optional[List[float]]:
        """Return the L2-normalised embedding for a single detection."""
        if self._embeddings is None or detection_index < 0 or detection_index >= self._num_detections:
            return None
        return self._embeddings[detection_index].tolist()

    def get_pairwise_similarity(self, idx_a: int, idx_b: int) -> float:
        """Cosine similarity between two detections."""
        self._require_index()
        a = self._embeddings[idx_a]
        b = self._embeddings[idx_b]
        return float(np.dot(a, b))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_index(self) -> None:
        if self._index is None or self._embeddings is None:
            raise RuntimeError(
                "FAISS index not built.  Call build_index() first."
            )

    def _encode_crops(self, crops: List[Image.Image]) -> torch.Tensor:
        """
        Encode crops using the primary model with automatic fallback.

        Tries models in order of preference until one succeeds.
        """
        models_to_try: List[str] = [self.primary_model] + [
            m for m in self.MODEL_PRIORITY if m != self.primary_model
        ]

        last_error: Optional[Exception] = None

        for model_name in models_to_try:
            try:
                if model_name == "virchow":
                    wrapper = VirchowModelWrapper.get_instance()
                    if not wrapper.is_loaded:
                        wrapper.load()
                    return wrapper.encode_crops(crops, batch_size=self.batch_size)

                elif model_name == "uni":
                    wrapper = UniModelWrapper.get_instance()
                    if not wrapper.is_loaded:
                        wrapper.load()
                    return wrapper.encode_crops(crops, batch_size=self.batch_size)

                elif model_name == "conch":
                    wrapper = ConchModelWrapper.get_instance()
                    if not wrapper.is_loaded:
                        wrapper.load()
                    return wrapper.encode_image_crops(crops, batch_size=self.batch_size)

            except Exception as e:
                logger.warning(f"Model '{model_name}' failed to encode crops: {e}")
                last_error = e
                continue

        raise RuntimeError(
            f"All foundation models failed to encode crops.  "
            f"Last error: {last_error}"
        )
