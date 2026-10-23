
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.retrieval.toolbench_ranker import get_retriever


QUERIES = [
    "Find hotels in Hyderabad under 3000 rupees per night",
    "Find accommodation with available rooms and prices",
    "Get details about a specific hotel",
    "Search for flights from Hyderabad to Delhi",
    "Compare flight fares between two cities",
    "Find flights when the destination is flexible",
    "Find restaurants in a given city",
    "Find restaurants with cuisine and price filters",
    "Get reviews for a specific restaurant",
    "Search for products by name",
    "Find products and compare their prices",
    "Track a shipment using its tracking number",
    "Find nearby places using geographic coordinates",
    "Search for rental cars in a city",
    "Find available transport between two locations",
]


def main():
    retriever = get_retriever()
    rows = []

    for query in QUERIES:
        print(f"Retrieving candidates: {query}")
        results = retriever.rank(query, top_k=10)

        for rank, result in enumerate(results, start=1):
            rows.append({
                "query": query,
                "rank": rank,
                "document_id": str(result["document_id"]),
                "api_text": result["api_text"],
                "ranking_score": result["ranking_score"],
                "relevant": "",
            })

    output_path = PROJECT_ROOT / "scripts" / "retrieval_labeling_sheet.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nCreated: {output_path}")
    print(f"Queries: {len(QUERIES)}")
    print(f"Candidates: {len(rows)}")
    print("Open the CSV and fill in the 'relevant' column:")
    print("1 = relevant, 0 = not relevant")
    print("Leave blank if you are unsure.")


if __name__ == "__main__":
    main()
