from functools import lru_cache
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "toolbench_ranker"

CHECKPOINT_PATH = MODEL_DIR / "nexustwin_api_ranker.pt"
EMBEDDINGS_PATH = MODEL_DIR / "api_embeddings.npy"
CORPUS_PATH = MODEL_DIR / "corpus_clean.csv"

DEVICE = torch.device("cpu")

STOPWORDS = {
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "by",
    "and", "or", "is", "are", "with", "from", "under", "per",
    "find", "get", "please", "want", "need", "api", "me", "my",
    "that", "this", "can", "could", "show", "give", "search",
}

SYNONYMS = {
    "hotel": {"hotel", "hotels", "accommodation", "lodging", "room", "rooms"},
    "hotels": {"hotel", "hotels", "accommodation", "lodging", "room", "rooms"},
    "accommodation": {"hotel", "hotels", "accommodation", "lodging", "room", "rooms"},
    "flight": {"flight", "flights", "airline", "airlines", "airfare"},
    "flights": {"flight", "flights", "airline", "airlines", "airfare"},
    "restaurant": {"restaurant", "restaurants", "food", "dining"},
    "restaurants": {"restaurant", "restaurants", "food", "dining"},
    "price": {"price", "prices", "cost", "rate", "rates", "fare"},
    "cheap": {"cheap", "budget", "affordable", "price", "cost"},
    "booking": {"booking", "reservation", "reserve", "book"},
    "book": {"booking", "reservation", "reserve", "book"},
}

CATEGORY_KEYWORDS = {
    "hotel": {"travel", "accommodation", "hotel", "search"},
    "accommodation": {"travel", "accommodation", "hotel", "search"},
    "flight": {"travel", "flight", "airline", "transport"},
    "restaurant": {"restaurant", "food", "dining", "location"},
    "shopping": {"shopping", "product", "ecommerce", "store"},
    "transport": {"transport", "travel", "vehicle", "logistics"},
}

ACTION_KEYWORDS = {
    "search": {"search", "find", "lookup", "list", "discover"},
    "details": {"detail", "details", "information", "info", "retrieve"},
    "book": {"book", "booking", "reserve", "reservation"},
    "price": {"price", "prices", "cost", "rate", "rates", "fare"},
    "track": {"track", "tracking", "status", "shipment"},
}


class APIRelevanceNetwork(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


def tokenize(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))
    return expanded


class NeuralAPIRanker:
    def __init__(self):
        for path in (CHECKPOINT_PATH, EMBEDDINGS_PATH, CORPUS_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Missing model file: {path}")

        self.checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        if not isinstance(self.checkpoint, dict):
            raise ValueError("Invalid checkpoint format.")

        required_keys = {
            "model_state_dict",
            "input_dim",
            "embedding_dim",
            "encoder_name",
            "api_ids",
        }
        missing = required_keys - self.checkpoint.keys()
        if missing:
            raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")

        self.corpus = pd.read_csv(CORPUS_PATH).fillna("")
        self.embeddings = np.load(EMBEDDINGS_PATH).astype(
            np.float32, copy=False
        )

        required_columns = {"docid", "document_content", "api_text"}
        if not required_columns.issubset(self.corpus.columns):
            raise ValueError(
                f"Corpus must contain columns: {sorted(required_columns)}"
            )

        self.api_ids = [str(value) for value in self.checkpoint["api_ids"]]
        corpus_ids = self.corpus["docid"].astype(str).tolist()

        if corpus_ids != self.api_ids:
            raise ValueError(
                "Corpus document IDs/order do not match checkpoint api_ids."
            )

        embedding_dim = int(self.checkpoint["embedding_dim"])
        input_dim = int(self.checkpoint["input_dim"])

        if embedding_dim != self.embeddings.shape[1]:
            raise ValueError(
                "Checkpoint embedding_dim does not match embedding file."
            )

        if input_dim != embedding_dim * 3:
            raise ValueError(
                f"Expected input_dim = 3 * embedding_dim; got "
                f"{input_dim} and {embedding_dim}."
            )

        if len(self.corpus) != len(self.embeddings):
            raise ValueError("Corpus and embedding row counts differ.")

        self.encoder = SentenceTransformer(
            self.checkpoint["encoder_name"],
            device=str(DEVICE),
        )

        self.model = APIRelevanceNetwork(input_dim).to(DEVICE)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

        # Parse structured metadata once, rather than for every query.
        self.metadata = []
        for _, row in self.corpus.iterrows():
            try:
                doc = json.loads(str(row["document_content"]))
                if not isinstance(doc, dict):
                    doc = {}
            except (json.JSONDecodeError, TypeError, ValueError):
                doc = {}

            api_name = str(doc.get("api_name", ""))
            tool_name = str(doc.get("tool_name", ""))
            category = str(doc.get("category_name", ""))
            description = str(doc.get("api_description", ""))
            api_text = str(row["api_text"])

            combined = " ".join(
                [api_name, tool_name, category, description, api_text]
            )

            self.metadata.append({
                "document": doc,
                "name": api_name,
                "tool": tool_name,
                "category": category,
                "description": description,
                "tokens": tokenize(combined),
                "name_tokens": tokenize(api_name + " " + tool_name),
                "text": combined,
            })

        print(
            f"Neural API ranker loaded: {len(self.corpus)} documents "
            f"on {DEVICE}."
        )

    def _metadata_score(self, query: str, meta: dict) -> float:
        query_tokens = tokenize(query)
        expanded = expand_query_tokens(query_tokens)

        if not expanded:
            return 0.0

        document_tokens = meta["tokens"]
        name_tokens = meta["name_tokens"]

        # Measure query coverage and reward important matches in API names.
        matched = expanded & document_tokens
        coverage = len(matched) / len(expanded)

        name_matches = expanded & name_tokens
        name_score = min(len(name_matches) / max(len(expanded), 1), 1.0)

        score = 0.55 * coverage + 0.45 * name_score
        query_lower = query.lower()
        text_lower = meta["text"].lower()

        # Match the requested action to what the endpoint actually does.
        for action, terms in ACTION_KEYWORDS.items():
            if any(term in query_tokens for term in terms):
                if any(term in text_lower for term in terms):
                    score += 0.12
                elif action == "search":
                    # Search requests should not prefer detail-only endpoints.
                    if any(term in text_lower for term in ACTION_KEYWORDS["details"]):
                        score -= 0.10

        # Prefer primary result-producing operations over helper endpoints.
        query_words = query.lower()
        api_name = meta["name"].lower()
        description = meta["description"].lower()
        operation_text = f"{api_name} {description}"

        search_intent = any(
            phrase in query_words
            for phrase in ("find ", "search ", "look for", "show me", "list ")
        )

        if search_intent:
            primary_terms = (
                "hotel search", "hotels search", "search hotel",
                "search hotels", "available hotels", "search results",
                "flight search", "search flights", "product search",
                "search products", "car rental search",
            )
            helper_terms = (
                "filters for search", "search filters",
                "filter options", "filter list", "regions search",
                "search locations", "location lookup", "get config",
                "markets", "autocomplete",
            )

            if any(term in api_name for term in primary_terms):
                score += 0.20

            if any(term in api_name for term in helper_terms):
                score -= 0.20

            # A filter endpoint may mention search without returning results.
            if "filter" in api_name and not any(
                term in api_name for term in primary_terms
            ):
                score -= 0.15

        # Give a modest boost when the requested service category matches.
        for keyword, category_terms in CATEGORY_KEYWORDS.items():
            if keyword in query_tokens or keyword in query_lower:
                category_text = (
                    meta["category"] + " " + meta["name"] + " " + meta["tool"]
                ).lower()
                if any(term in category_text for term in category_terms):
                    score += 0.15
                else:
                    score -= 0.08

        # Product search and price-comparison intent.
        product_query = any(term in query_lower for term in (
            "product", "products", "shopping",
            "compare prices", "price comparison",
            "compare product prices", "product offers",
        ))

        price_comparison = any(term in query_lower for term in (
            "compare prices", "price comparison",
            "compare product prices", "compare their prices",
            "lowest price", "cheapest",
        ))

        product_domain = any(term in text_lower for term in (
            "ecommerce", "e-commerce", "commerce",
            "product", "products", "shopping",
        ))

        if product_query:
            score += 0.18 if product_domain else -0.08

        if price_comparison:
            supports_price = any(term in text_lower for term in (
                "price", "prices", "min_price", "max_price",
                "sort_by", "lowest_price", "highest_price",
                "priceasc", "pricedesc",
            ))

            if supports_price and (not product_query or product_domain):
                score += 0.20
            elif not supports_price:
                score -= 0.10

        # Distinguish actual car-rental search from unrelated car APIs.
        car_rental_query = any(term in query_lower for term in (
            "car rental", "car rentals", "rental car",
            "rental cars", "rent a car", "car hire",
            "vehicle rental", "vehicle hire",
        ))

        if car_rental_query:
            rental_intent = any(term in text_lower for term in (
                "car rental search", "rental car search",
                "search rental cars", "search for rental cars",
                "car hire search", "vehicle rental search",
                "rental availability", "rental booking",
            ))

            vacation_rental = any(term in text_lower for term in (
                "vacation rental", "vacation rentals",
                "holiday rental", "holiday rentals",
            ))

            car_listing = (
                "search cars" in text_lower
                or "car listings" in text_lower
                or "cars for sale" in text_lower
            )

            if vacation_rental:
                score -= 0.35
            elif rental_intent:
                score += 0.35
            elif car_listing:
                score -= 0.20

        # Penalize deprecated endpoints; do not blindly remove them.
        if "deprecated" in text_lower:
            score -= 0.50

        # Prefer actual flight-search endpoints for flight-search requests.
        flight_terms = {"flight", "flights", "airfare"}
        if query_tokens & flight_terms:
            if any(term in api_name for term in (
                "flight search", "search flight", "flight fare search"
            )):
                score += 0.25
            if "search" in description and "flight" in description:
                score += 0.15
            if any(term in api_name for term in (
                "list-by-airline", "list-in-boundary", "aircraftsearch"
            )):
                score -= 0.20

        # Prefer transit-search APIs for transport availability requests.
        transport_intent = any(term in query_lower for term in (
            "available transport", "transport between",
            "find transport", "public transport",
            "bus between", "train between", "search trips",
            "available trips", "transport options",
        ))

        if transport_intent:
            transit_terms = (
                "search trips", "search trip", "transit",
                "public transportation", "bus", "railway",
                "train", "transportation search",
            )
            route_terms = (
                "directions between 2 locations",
                "distance matrix", "getdistance",
            )

            if any(term in text_lower for term in transit_terms):
                score += 0.25

            if any(term in text_lower for term in route_terms):
                score -= 0.20
        return float(score)

    @torch.inference_mode()
    def rank_neural_only(self, query: str, top_k: int = 10) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        # This internal method must support larger candidate pools for evaluation.
        top_k = max(1, min(int(top_k), len(self.corpus)))

        query_embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores_all = []

        for start in range(0, len(self.embeddings), 512):
            end = min(start + 512, len(self.embeddings))

            docs = torch.from_numpy(
                self.embeddings[start:end]
            ).to(DEVICE)

            queries = torch.from_numpy(
                np.repeat(
                    query_embedding[None, :],
                    end - start,
                    axis=0,
                )
            ).to(DEVICE)

            features = torch.cat(
                [queries, docs, torch.abs(queries - docs)],
                dim=1,
            )

            scores = self.model(features).squeeze(-1).cpu().numpy()
            scores_all.extend(scores.tolist())

        top_indices = np.argsort(
            np.asarray(scores_all), kind="stable"
        )[::-1][:top_k]

        return [
            {
                "document_id": self.api_ids[int(index)],
                "description": str(
                    self.corpus.iloc[int(index)]["document_content"]
                ),
                "api_text": str(
                    self.corpus.iloc[int(index)]["api_text"]
                ),
                "ranking_score": float(scores_all[int(index)]),
            }
            for index in top_indices
        ]

    @torch.inference_mode()
    def rank(self, query: str, top_k: int = 5) -> list[dict]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        top_k = max(1, min(int(top_k), 20))

        query_embedding = self.encoder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        if query_embedding.shape != (self.embeddings.shape[1],):
            raise ValueError("Query embedding dimension mismatch.")

        scores_all = []

        for start in range(0, len(self.embeddings), 512):
            end = min(start + 512, len(self.embeddings))

            docs = torch.from_numpy(self.embeddings[start:end]).to(DEVICE)
            queries = torch.from_numpy(
                np.repeat(query_embedding[None, :], end - start, axis=0)
            ).to(DEVICE)

            features = torch.cat(
                [queries, docs, torch.abs(queries - docs)],
                dim=1,
            )

            scores = self.model(features).squeeze(-1).cpu().numpy()
            scores_all.extend(scores.tolist())

        neural_scores = np.asarray(scores_all, dtype=np.float32)

        # Convert neural scores into a comparable 0–1 percentile.
        order = np.argsort(neural_scores, kind="stable")
        neural_percentile = np.empty(len(neural_scores), dtype=np.float32)
        neural_percentile[order] = np.linspace(
            0.0, 1.0, num=len(neural_scores), endpoint=True
        )

        metadata_scores = np.asarray(
            [
                self._metadata_score(query, meta)
                for meta in self.metadata
            ],
            dtype=np.float32,
        )

        # Normalize metadata scores robustly to 0–1.
        low = float(metadata_scores.min())
        high = float(metadata_scores.max())
        if high > low:
            metadata_normalized = (
                (metadata_scores - low) / (high - low)
            )
        else:
            metadata_normalized = np.zeros_like(metadata_scores)

        # Keep the neural ranking influential, while adding metadata relevance.
        final_scores = (
            0.80 * neural_percentile
            + 0.20 * metadata_normalized
        )

        top_indices = np.argsort(
            final_scores, kind="stable"
        )[::-1][:top_k]

        results = []
        for index in top_indices:
            row = self.corpus.iloc[int(index)]

            results.append({
                "document_id": self.api_ids[int(index)],
                "description": str(row["document_content"]),
                "api_text": str(row["api_text"]),
                "ranking_score": float(final_scores[int(index)]),
            })

        return results

    @torch.inference_mode()
    def rank_neural_first(self, query: str, top_k: int = 10) -> list[dict]:
        """Experimental reranking: neural shortlist followed by metadata."""
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty.")

        top_k = max(1, min(int(top_k), 20))

        # Retrieve a broader shortlist using the existing neural model.
        shortlist_size = min(len(self.corpus), max(100, top_k * 10))
        neural_results = self.rank_neural_only(query, top_k=shortlist_size)

        id_to_index = {
            api_id: index for index, api_id in enumerate(self.api_ids)
        }

        candidate_indices = [
            id_to_index[result["document_id"]]
            for result in neural_results
        ]

        metadata_scores = np.asarray(
            [
                self._metadata_score(query, self.metadata[index])
                for index in candidate_indices
            ],
            dtype=np.float32,
        )

        # Normalize metadata scores within the shortlist only.
        low = float(metadata_scores.min())
        high = float(metadata_scores.max())

        if high > low:
            metadata_normalized = (metadata_scores - low) / (high - low)
        else:
            metadata_normalized = np.zeros_like(metadata_scores)

        neural_scores = np.asarray(
            [result["ranking_score"] for result in neural_results],
            dtype=np.float32,
        )

        # Normalize neural scores within the same shortlist.
        n_low = float(neural_scores.min())
        n_high = float(neural_scores.max())

        if n_high > n_low:
            neural_normalized = (
                (neural_scores - n_low) / (n_high - n_low)
            )
        else:
            neural_normalized = np.zeros_like(neural_scores)

        final_scores = (
            0.80 * neural_normalized
            + 0.20 * metadata_normalized
        )

        order = np.argsort(final_scores, kind="stable")[::-1][:top_k]
        results = []

        for position in order:
            index = candidate_indices[int(position)]
            row = self.corpus.iloc[index]

            results.append({
                "document_id": self.api_ids[index],
                "description": str(row["document_content"]),
                "api_text": str(row["api_text"]),
                "ranking_score": float(final_scores[int(position)]),
            })

        return results


@lru_cache(maxsize=1)
def get_retriever() -> NeuralAPIRanker:
    return NeuralAPIRanker()