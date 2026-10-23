from core.retrieval.toolbench_ranker import get_retriever

retriever = get_retriever()

queries = [
    "Search for rental cars in a city",
    "Find restaurants in a given city",
    "Find products and compare their prices",
]

for query in queries:
    print(f"\nQUERY: {query}")
    results = retriever.rank_neural_first(query, top_k=5)

    for result in results:
        print(
            result["document_id"],
            round(result["ranking_score"], 4),
            result["api_text"][:180],
        )
