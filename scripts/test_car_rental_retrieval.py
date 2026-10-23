from core.retrieval.toolbench_ranker import NeuralAPIRanker

ranker = NeuralAPIRanker()
results = ranker.rank("Search for rental cars in a city", top_k=10)

for item in results:
    print(
        "ID:", item["document_id"],
        "| Score:", round(item["ranking_score"], 4),
        "|", item["api_text"][:220]
    )

