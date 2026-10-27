"""Generate a small synthetic placeholder dataset. Replace with your real labels."""
import random
from pathlib import Path

import pandas as pd

random.seed(0)
topics = ["RAG", "vector databases", "BM25", "embeddings", "reranking", "chunking",
          "LLM agents", "fine-tuning", "knowledge graphs", "query expansion"]
simple = ["What is {}?", "Define {}.", "Who created {}?", "Give a short definition of {}.",
          "What does {} stand for?"]
moderate = ["How does {} improve retrieval quality?", "Explain the main steps of {}.",
            "What are the pros and cons of {}?", "How do I set up {} for a small project?",
            "Why is {} used in search pipelines?"]
difficult = ["Compare {} and {} and justify which is better for multi-hop legal question answering under latency limits.",
             "Design an evaluation framework for {} combined with {}, covering failure modes and cost tradeoffs.",
             "Analyze how {} interacts with {} when evidence conflicts, and propose mitigations.",
             "Synthesize recent research on {} versus {} and recommend an architecture."]

rows = []
for t in topics:
    for tpl in simple:
        rows.append((tpl.format(t), "simple"))
    for tpl in moderate:
        rows.append((tpl.format(t), "moderate"))
    for tpl in difficult:
        rows.append((tpl.format(t, random.choice([x for x in topics if x != t])), "difficult"))

out = Path(__file__).resolve().parents[1] / "data" / "queries.csv"
out.parent.mkdir(exist_ok=True)
pd.DataFrame(rows, columns=["query", "label"]).to_csv(out, index=False)
print(f"Wrote {len(rows)} rows to {out}")
