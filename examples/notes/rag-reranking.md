# Reranking in RAG pipelines

Retrieval usually happens in two stages. A fast **bi-encoder** embeds the query and every
document separately, so the documents can be indexed ahead of time and searched with
approximate nearest neighbour. It's quick, but it never sees the query and the document
together, so fine-grained relevance gets lost.

A **cross-encoder** reads the query and a candidate passage in one forward pass, so attention
runs across both. That's far more accurate, but you can't precompute it. You only run it on the
top 20–50 candidates from the first stage.

Rule of thumb: retrieve wide (high recall), rerank narrow (high precision).

## Notes from experiments
- `ms-marco-MiniLM-L-6-v2` is small enough to run on CPU at interactive speed.
- Reranking helped most on questions where the answer used different wording from the query.
- Over-fetching ~4x the final result count gave the reranker enough to work with.
