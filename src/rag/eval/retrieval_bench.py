from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from datasets import load_dataset
from rag.ingest import Chunk
from rag.retrieve import RetrievalResult


@dataclass
class BenchmarkSample:
    question: str
    answer: str
    supporting_facts: list[str]


def load_hotpotqa_subset(n: int = 200) -> list[BenchmarkSample]:
    """
    HotpotQA (Yang et al. 2018): https://arxiv.org/abs/1809.09600
    Gold-standard multi-hop QA. Enables direct comparison with published RAG papers.
    Uses its own Wikipedia corpus — benchmarks retrieval component in isolation.
    """
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    samples = []
    for row in ds.select(range(min(n, len(ds)))):
        support = []
        for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
            if title in row["supporting_facts"]["title"]:
                support.extend(sents)
        samples.append(BenchmarkSample(row["question"], row["answer"], support[:3]))
    return samples


def load_frames_subset(n: int = 200) -> list[BenchmarkSample]:
    """
    FRAMES (Google DeepMind, 2024): https://arxiv.org/abs/2409.12941
    Unlike BEIR (retrieval only), FRAMES tests the full pipeline —
    factuality, retrieval, and multi-step reasoning. More realistic than BEIR.
    """
    ds = load_dataset("google/frames-benchmark", split="test")
    samples = []
    for row in ds.select(range(min(n, len(ds)))):
        facts = [str(l) for l in (row.get("wiki_links") or [])[:3]]
        samples.append(BenchmarkSample(row["Prompt"], row["Answer"], facts))
    return samples


def _recall_at_k(retrieved: list[RetrievalResult], facts: list[str], k: int) -> float:
    if not facts:
        return 0.0
    top_texts = [r.text.lower() for r in retrieved[:k]]
    hits = sum(1 for f in facts if any(f.lower()[:60] in t for t in top_texts))
    return hits / len(facts)


def run_retrieval_benchmark(
    samples: list[BenchmarkSample],
    corpus: list[Chunk],
    retrieve_fn: Callable[[str, int], list[RetrievalResult]],
    k_values: list[int] = [3, 5, 10],
) -> dict[str, float]:
    sums = {k: 0.0 for k in k_values}
    for s in samples:
        results = retrieve_fn(s.question, max(k_values))
        for k in k_values:
            sums[k] += _recall_at_k(results, s.supporting_facts, k)
    return {f"recall@{k}": sums[k] / len(samples) for k in k_values}
