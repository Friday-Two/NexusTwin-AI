from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from database.models.entities import Service, UserPreference

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "ranking" / "preference_ranker.pt"
DEVICE = torch.device("cpu")


class PairRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


class PreferenceMLRanker:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing preference checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=True
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid preference-ranker checkpoint.")

        required = {"state_dict", "input_dim", "encoder", "threshold"}
        missing = required - checkpoint.keys()
        if missing:
            raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

        self.encoder = SentenceTransformer(
            checkpoint["encoder"], device=str(DEVICE)
        )
        self.model = PairRanker(int(checkpoint["input_dim"])).to(DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.threshold = float(checkpoint["threshold"])

    def rank(
        self,
        services: list[Service],
        preferences: list[UserPreference],
    ) -> list[tuple[Service, float]]:
        safe_preferences = [
            p.preference_value.strip()
            for p in preferences
            if not p.is_sensitive and p.preference_value.strip()
        ]

        if not services:
            return []

        if not safe_preferences:
            return [(service, 0.0) for service in sorted(
                services, key=lambda s: (s.name.lower(), s.id)
            )]

        query_text = " ".join(safe_preferences)
        service_texts = [
            f"{service.name}. Category: {service.category}. {service.description or ''}"
            for service in services
        ]

        query_vectors = self.encoder.encode(
            [query_text] * len(services),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        service_vectors = self.encoder.encode(
            service_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        features = np.concatenate(
            [
                query_vectors,
                service_vectors,
                np.abs(query_vectors - service_vectors),
            ],
            axis=1,
        ).astype(np.float32)

        expected_dim = self.model.network[0].in_features
        if features.shape[1] != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {expected_dim}, "
                f"got {features.shape[1]}"
            )

        with torch.no_grad():
            logits = self.model(
                torch.tensor(features, dtype=torch.float32, device=DEVICE)
            )
            scores = torch.sigmoid(logits).cpu().numpy()

        ranked = [
            (service, round(float(score), 4))
            for service, score in zip(services, scores)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0].name.lower(), item[0].id))
        return ranked


@lru_cache(maxsize=1)
def get_preference_ml_ranker() -> PreferenceMLRanker:
    """Load once when explicitly requested; errors are handled by the caller."""
    return PreferenceMLRanker()


def rank_services_with_fallback(
    services: list[Service],
    preferences: list[UserPreference],
    use_ml: bool = False,
) -> list[tuple[Service, float]]:
    """Use ML only when explicitly enabled; otherwise use deterministic ranking."""
    from core.negotiation.service_ranker import rank_services

    if not use_ml:
        return rank_services(services, preferences)

    try:
        return get_preference_ml_ranker().rank(services, preferences)
    except Exception as exc:
        # Keep discovery available if the experimental model cannot run.
        import logging
        logging.getLogger(__name__).warning(
            "Preference ML ranking failed; using deterministic fallback: %s",
            exc,
        )
        return rank_services(services, preferences)
