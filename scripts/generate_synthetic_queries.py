"""Generate synthetic routing queries grounded in the concepts of
'The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs'.

Labels:
  simple    -> single-fact lookup / definition (1 chunk is enough)
  moderate  -> explanation, how-to, or a single comparison (a few chunks)
  difficult -> multi-concept reasoning, trade-off analysis, design (multi-hop)

Usage: python scripts/generate_synthetic_queries.py [--n 1000] [--out data/synthetic_queries.csv]
"""
import argparse
import random
from pathlib import Path

import pandas as pd

# (concept, chapter) -- taken from the guide's table of contents
CONCEPTS = [
    ("statistical language models", "1 Introduction"),
    ("neural language models", "1 Introduction"),
    ("pre-trained language models", "1 Introduction"),
    ("large language models", "1 Introduction"),
    ("fine-tuning", "1 Introduction"),
    ("unsupervised fine-tuning", "1 Introduction"),
    ("supervised fine-tuning (SFT)", "1 Introduction"),
    ("instruction fine-tuning", "1 Introduction"),
    ("pre-training", "1 Introduction"),
    ("Retrieval Augmented Generation (RAG)", "1 Introduction"),
    ("the traditional RAG pipeline", "1 Introduction"),
    ("the challenges of serving RAG", "1 Introduction"),
    ("the seven-stage fine-tuning pipeline", "2 Seven Stage Pipeline"),
    ("dataset preparation", "2 Seven Stage Pipeline"),
    ("model initialisation", "2 Seven Stage Pipeline"),
    ("training environment setup", "2 Seven Stage Pipeline"),
    ("partial versus full fine-tuning", "2 Seven Stage Pipeline"),
    ("evaluation and validation", "2 Seven Stage Pipeline"),
    ("model deployment", "2 Seven Stage Pipeline"),
    ("monitoring and maintenance", "2 Seven Stage Pipeline"),
    ("data collection", "3 Data Preparation"),
    ("data preprocessing and formatting", "3 Data Preparation"),
    ("handling data imbalance", "3 Data Preparation"),
    ("dataset splitting", "3 Data Preparation"),
    ("data annotation", "3 Data Preparation"),
    ("data augmentation", "3 Data Preparation"),
    ("synthetic data generation using LLMs", "3 Data Preparation"),
    ("ethical data handling", "3 Data Preparation"),
    ("high-quality data collection", "3 Data Preparation"),
    ("tools and libraries for model initialisation", "4 Model Initialisation"),
    ("challenges in model initialisation", "4 Model Initialisation"),
    ("hyperparameter tuning", "5 Training Setup"),
    ("random search for hyperparameters", "5 Training Setup"),
    ("grid search for hyperparameters", "5 Training Setup"),
    ("Bayesian optimisation for hyperparameters", "5 Training Setup"),
    ("gradient descent", "5 Training Setup"),
    ("stochastic gradient descent (SGD)", "5 Training Setup"),
    ("mini-batch gradient descent", "5 Training Setup"),
    ("the AdaGrad optimiser", "5 Training Setup"),
    ("the RMSprop optimiser", "5 Training Setup"),
    ("the AdaDelta optimiser", "5 Training Setup"),
    ("the Adam optimiser", "5 Training Setup"),
    ("the AdamW optimiser", "5 Training Setup"),
    ("loss functions", "5 Training Setup"),
    ("task-specific fine-tuning", "6 Fine-Tuning"),
    ("domain-specific fine-tuning", "6 Fine-Tuning"),
    ("parameter-efficient fine-tuning (PEFT)", "6 Fine-Tuning"),
    ("adapters", "6 Fine-Tuning"),
    ("Low-Rank Adaptation (LoRA)", "6 Fine-Tuning"),
    ("QLoRA", "6 Fine-Tuning"),
    ("DoRA (weight-decomposed low-rank adaptation)", "6 Fine-Tuning"),
    ("fine-tuning with multiple adapters", "6 Fine-Tuning"),
    ("Half Fine-Tuning (HFT)", "6 Fine-Tuning"),
    ("Lamini memory tuning", "6 Fine-Tuning"),
    ("Mixture of Experts (MoE)", "6 Fine-Tuning"),
    ("the Mixtral 8x7B architecture", "6 Fine-Tuning"),
    ("Mixture of Agents (MoA)", "6 Fine-Tuning"),
    ("Proximal Policy Optimisation (PPO)", "6 Fine-Tuning"),
    ("Direct Preference Optimisation (DPO)", "6 Fine-Tuning"),
    ("Odds-Ratio Preference Optimization (ORPO)", "6 Fine-Tuning"),
    ("pruning LLMs", "6 Fine-Tuning"),
    ("cross-entropy loss", "7 Evaluation"),
    ("advanced LLM evaluation metrics", "7 Evaluation"),
    ("training loss curves", "7 Evaluation"),
    ("overfitting", "7 Evaluation"),
    ("noisy gradients", "7 Evaluation"),
    ("validation loops", "7 Evaluation"),
    ("benchmarking fine-tuned LLMs", "7 Evaluation"),
    ("safety benchmarks for fine-tuned LLMs", "7 Evaluation"),
    ("Llama Guard", "7 Evaluation"),
    ("ShieldGemma", "7 Evaluation"),
    ("WildGuard", "7 Evaluation"),
    ("cloud-based providers for LLM deployment", "8 Deployment"),
    ("on-premises GPU-based LLM deployment", "8 Deployment"),
    ("distributed torrent-style LLM deployment", "8 Deployment"),
    ("WebGPU-based LLM deployment", "8 Deployment"),
    ("WebLLM", "8 Deployment"),
    ("quantised LLMs", "8 Deployment"),
    ("inference optimisation", "8 Deployment"),
    ("functional monitoring", "9 Monitoring"),
    ("prompt monitoring", "9 Monitoring"),
    ("response monitoring", "9 Monitoring"),
    ("alerting mechanisms and thresholds", "9 Monitoring"),
    ("multimodal LLMs", "10+ Advanced Topics"),
    ("fine-tuning for audio and speech processing", "10+ Advanced Topics"),
    ("scalability challenges in fine-tuning", "10+ Advanced Topics"),
    ("privacy in LLM fine-tuning", "10+ Advanced Topics"),
    ("accountability in LLM fine-tuning", "10+ Advanced Topics"),
]

SIMPLE = [
    "What is {}?", "Define {}.", "what is {}", "Can you define {} briefly?",
    "Give a one-line definition of {}.", "What does the guide say {} is?",
    "In which chapter of the guide is {} discussed?", "What is the purpose of {}?",
    "{} - what is it?", "Briefly, what do we mean by {}?",
    "Is {} covered in the fine-tuning guide?", "Name the main idea behind {}.",
]

MODERATE = [
    "How does {} work?", "Explain {} in the context of fine-tuning LLMs.",
    "What are the main benefits and limitations of {}?",
    "Why is {} important when fine-tuning a large language model?",
    "What challenges should I expect when dealing with {}?",
    "What best practices does the guide recommend regarding {}?",
    "How does {} fit into the overall LLM fine-tuning pipeline?",
    "Describe how {} affects model quality.",
    "What common mistakes occur with {} and how can they be avoided?",
    "Summarise the role of {} in fine-tuning.",
    "What does the guide explain about {}?",
]

MODERATE_PAIR = [
    "What is the difference between {a} and {b}?",
    "How does {a} compare to {b}?",
    "When would you choose {a} over {b}?",
    "how are {a} and {b} related?",
]

DIFFICULT_PAIR = [
    "Compare {a} and {b} in terms of compute cost, accuracy and operational complexity, and recommend which to use for a low-resource team.",
    "How do {a} and {b} interact in a full fine-tuning workflow, and what failure modes arise when they are combined?",
    "Analyse the trade-offs between {a} and {b} when fine-tuning under a tight GPU memory budget, and justify a recommendation.",
    "Design an end-to-end strategy that integrates {a} with {b} for a production LLM, covering data, training, evaluation and monitoring.",
    "Critically evaluate whether {a} can replace {b}, citing the strengths and weaknesses of each and when each fails.",
    "If {a} degrades model quality, how would you diagnose whether the cause lies in {b}, and what experiments would you run?",
    "Synthesise what the guide says about {a} and {b} into a decision framework for choosing a fine-tuning approach.",
    "How does the choice of {a} change the requirements for {b}, and how would you validate that the combination generalises?",
]

DIFFICULT_SCENARIO = [
    "A company has 5,000 labeled domain documents and a single 24GB GPU. Using {a} and {b}, how would you plan the complete fine-tuning pipeline from data preparation to deployment and monitoring?",
    "Our fine-tuned model overfits and also produces unsafe outputs. Considering {a} and {b}, what would you change in the pipeline and how would you verify the fix?",
    "Should we use RAG or fine-tuning for a fast-changing knowledge base? Reason through the role of {a} and {b} and the cost, latency and maintenance trade-offs.",
    "We must align a model to human preferences while keeping inference cheap. Explain how {a} and {b} would each contribute, and where they conflict.",
    "Evaluate the risks to privacy, scalability and accountability when applying {a} together with {b}, and propose mitigations.",
]


def noisy(q: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.06:
        return q.lower().rstrip("?.")
    if r < 0.10:
        return q.rstrip("?.")
    return q


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    per = [n // 3 + (1 if i < n % 3 else 0) for i in range(3)]
    rows, seen = [], set()

    def add(q, label, src):
        q = noisy(q, rng)
        key = q.lower()
        if key in seen:
            return False
        seen.add(key)
        rows.append((q, label, src))
        return True

    def fill(label, count, make):
        got, tries = 0, 0
        while got < count and tries < count * 200:
            tries += 1
            q, src = make()
            got += add(q, label, src)
        if got < count:
            raise RuntimeError(f"could only create {got}/{count} unique '{label}' queries")

    def one():
        return rng.choice(CONCEPTS)

    def two():
        a, b = rng.sample(CONCEPTS, 2)
        return a, b

    def make_simple():
        c, ch = one()
        return rng.choice(SIMPLE).format(c), ch

    def make_moderate():
        if rng.random() < 0.25:
            (a, cha), (b, chb) = two_c()
            return rng.choice(MODERATE_PAIR).format(a=a, b=b), f"{cha} | {chb}"
        c, ch = one()
        return rng.choice(MODERATE).format(c), ch

    def two_c():
        return rng.sample(CONCEPTS, 2)

    def make_difficult():
        (a, cha), (b, chb) = two_c()
        tpl = rng.choice(DIFFICULT_PAIR + DIFFICULT_SCENARIO)
        return tpl.format(a=a, b=b), f"{cha} | {chb}"

    fill("simple", per[0], make_simple)
    fill("moderate", per[1], make_moderate)
    fill("difficult", per[2], make_difficult)

    df = pd.DataFrame(rows, columns=["query", "label", "source_chapter"])
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "synthetic_queries.csv"))
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}")
    print(df["label"].value_counts().to_string())
