from core.retrieval.toolbench_ranker import get_retriever

retriever = get_retriever()

results = retriever.rank(
    "Find an API that can search for hotels and accommodation",
    top_k=5,
)

for i, result in enumerate(results, start=1):
    print(
        f"{i}. score={result['ranking_score']:.4f} | "
        f"{result['description'][:160]}"
    )
