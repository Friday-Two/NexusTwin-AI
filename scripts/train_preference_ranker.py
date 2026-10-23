"""Train a small MLP ranker on frozen SentenceTransformer embeddings.

Designed for Colab Free (GPU if available) and CPU fallback. The starter dataset is
synthetic; benchmark claims require replacing it with a documented real dataset.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REQUIRED_COLUMNS = {
    "query_id", "preference_text", "service_name", "category", "description", "label"
}


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_features(encoder, frame: pd.DataFrame, batch_size: int, device: str) -> torch.Tensor:
    query_texts = frame["preference_text"].astype(str).tolist()
    service_texts = (
        frame["service_name"].astype(str) + ". Category: "
        + frame["category"].astype(str) + ". " + frame["description"].astype(str)
    ).tolist()
    query_vectors = encoder.encode(
        query_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    service_vectors = encoder.encode(
        service_texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    features = np.concatenate(
        [query_vectors, service_vectors, np.abs(query_vectors - service_vectors)],
        axis=1,
    ).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


def ndcg_at_k(frame: pd.DataFrame, scores: np.ndarray, k: int = 5) -> float:
    values = []
    working = frame[["query_id", "label"]].copy()
    working["score"] = scores
    for _, group in working.groupby("query_id"):
        group = group.sort_values("score", ascending=False).head(k)
        gains = group["label"].to_numpy(dtype=np.float64)
        discounts = np.log2(np.arange(2, len(gains) + 2))
        dcg = float(np.sum((2**gains - 1) / discounts))
        ideal = np.sort(group["label"].to_numpy(dtype=np.float64))[::-1]
        ideal_discounts = np.log2(np.arange(2, len(ideal) + 2))
        idcg = float(np.sum((2**ideal - 1) / ideal_discounts))
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/ranking/starter_examples.csv")
    parser.add_argument("--output-dir", default="artifacts/ranking")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(data_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty or not set(frame["label"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Dataset must contain rows with binary labels 0 and 1.")
    frame["label"] = frame["label"].astype(int)
    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain both positive and negative labels.")
    if frame["query_id"].nunique() < 4:
        raise ValueError("Need at least four query_id groups for a grouped train/test split.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(frame, frame["label"], groups=frame["query_id"]))
    train_frame = frame.iloc[train_idx].reset_index(drop=True)
    test_frame = frame.iloc[test_idx].reset_index(drop=True)
    if train_frame["label"].nunique() != 2 or test_frame["label"].nunique() != 2:
        raise ValueError("Grouped split must contain both classes in train and test. Add more query groups.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Rows: train={len(train_frame)}, test={len(test_frame)}")
    print(f"Unique query groups: train={train_frame.query_id.nunique()}, test={test_frame.query_id.nunique()}")

    encoder = SentenceTransformer(args.encoder, device=device)
    train_x = make_features(encoder, train_frame, args.batch_size, device)
    test_x = make_features(encoder, test_frame, args.batch_size, device)
    train_y = torch.tensor(train_frame["label"].to_numpy(), dtype=torch.float32, device=device)
    test_y = test_frame["label"].to_numpy()

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    model = PairRanker(train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} - loss={np.mean(losses):.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(test_x)).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "dataset": str(data_path),
        "dataset_rows": len(frame),
        "dataset_query_groups": int(frame["query_id"].nunique()),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "device": device,
        "encoder": args.encoder,
        "seed": args.seed,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision": float(precision_score(test_y, predictions, zero_division=0)),
        "recall": float(recall_score(test_y, predictions, zero_division=0)),
        "f1": float(f1_score(test_y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "ndcg_at_5": ndcg_at_k(test_frame, probabilities, k=5),
        "data_warning": "Synthetic starter examples are for pipeline smoke tests only; not a benchmark result.",
    }

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "input_dim": int(train_x.shape[1]),
        "encoder": args.encoder,
        "threshold": 0.5,
    }, output_dir / "preference_ranker.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {output_dir / 'preference_ranker.pt'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
