from __future__ import annotations
from typing import Callable

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from rag.retrieve import RetrievalResult


def generate_testset(
    docs,
    n: int = 100,
    llm_model: str = "gpt-4o-mini",
) -> list[dict]:
    """
    RAGAS TestsetGenerator (paper: https://arxiv.org/abs/2309.15217).
    Synthesizes QA pairs without human annotation.
    Question types:
      simple (40%): single-hop, directly answerable from one passage
      reasoning (30%): requires inference beyond literal text
      multi_context (30%): answer requires multiple passages
    Always spot-check ~25 questions manually before trusting the testset.
    """
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.schema import Document

    gen_llm = ChatOpenAI(model=llm_model)
    critic_llm = ChatOpenAI(model=llm_model)
    generator = TestsetGenerator.from_langchain(gen_llm, critic_llm, OpenAIEmbeddings())
    langchain_docs = [Document(page_content=d.text if hasattr(d, "text") else d) for d in docs]
    testset = generator.generate_with_langchain_docs(
        langchain_docs, test_size=n,
        distributions={simple: 0.4, reasoning: 0.3, multi_context: 0.3},
    )
    return testset.to_pandas().to_dict(orient="records")


def run_ragas_eval(
    testset: list[dict],
    retrieve_fn: Callable[[str], list[RetrievalResult]],
    generate_fn: Callable,
    k: int = 5,
) -> dict[str, float]:
    questions, answers, contexts, ground_truths = [], [], [], []
    for sample in testset:
        q = sample["question"]
        results = retrieve_fn(q)
        ans = generate_fn(q, results)
        questions.append(q)
        answers.append(ans.answer if hasattr(ans, "answer") else str(ans))
        contexts.append([r.text for r in results])
        ground_truths.append(sample.get("ground_truth", ""))

    ds = Dataset.from_dict({
        "question": questions, "answer": answers,
        "contexts": contexts, "ground_truth": ground_truths,
    })
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_recall, context_precision])
    return {k: float(v) for k, v in result.items()}
