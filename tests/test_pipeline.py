import pandas as pd

from query_router.config import load_config
from query_router.model import build_pipeline


def test_pipeline_fits_and_predicts():
    cfg = load_config()
    df = pd.DataFrame({
        "query": ["what is python", "define rag", "who is ada",
                  "compare rag and finetuning for legal domains",
                  "how does reranking affect recall", "explain dense vs sparse retrieval",
                  "design a multi-hop retrieval system and justify tradeoffs across latency and cost",
                  "analyze why agentic rag fails on conflicting evidence and propose fixes",
                  "synthesize findings from several papers on long context vs retrieval"],
        "label": ["simple"] * 3 + ["moderate"] * 3 + ["difficult"] * 3,
    })
    pipe = build_pipeline(cfg).fit(df["query"], df["label"])
    pred = pipe.predict(["what is rag"])
    assert pred[0] in cfg["labels"]
    assert pipe.predict_proba(["x"]).shape == (1, 3)
