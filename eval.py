"""
eval.py – RAGAS evaluation harness for the AWS RAG Chatbot.

Generates synthetic test questions from the document corpus, runs them
through the RAG pipeline, and scores Faithfulness + Context Precision.

Respects Groq free-tier rate limits (8 000 TPM) by batching and sleeping.

Usage:
    uv run python eval.py --limit 3          # smoke test
    uv run python eval.py --limit 50         # full eval
"""

import argparse
import os
import random
import sys
import time
import tomllib
from pathlib import Path

import pandas as pd
from datasets import Dataset
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, faithfulness

from rag_chain import answer, load_resources

# ---------------------------------------------------------------------------
# API Key resolution
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key:
        return key
    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
            key = secrets.get("GROQ_API_KEY", "")
    if not key or key == "your-groq-api-key-here":
        print("ERROR: GROQ_API_KEY not found in env or .streamlit/secrets.toml")
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Testset Generation (lightweight, direct from chunks)
# ---------------------------------------------------------------------------

def generate_testset(
    llm: ChatGroq,
    total: int = 50,
    batch: int = 10,
    sleep_between: float = 2.0,
    batch_pause: float = 15.0,
) -> pd.DataFrame:
    """
    Generate *total* synthetic QA pairs directly from randomly sampled
    document chunks using Groq. Stays well under 8 000 TPM.
    """
    res = load_resources()
    total = min(total, len(res["texts"]))
    indices = random.sample(range(len(res["texts"])), total)

    system_prompt = (
        "You are an AWS exam question writer. Given a text snippet from AWS documentation, "
        "write ONE clear, specific question that can be answered using ONLY the snippet. "
        "Also write the concise ground-truth answer. "
        "Output format (strict):\n"
        "QUESTION: <your question>\n"
        "ANSWER: <ground truth answer>"
    )

    rows = []
    print(f"\n[Testset] Generating {total} questions from corpus chunks …")

    for i, idx in enumerate(indices):
        chunk_text = res["texts"][idx]
        chunk_source = res["sources"][idx]

        print(f"  [{i + 1}/{total}] Generating from {chunk_source} …", flush=True)
        try:
            resp = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Snippet:\n{chunk_text[:1200]}"),
            ])
            lines = resp.content.strip().split("\n")
            q_text = ""
            a_text = ""
            for line in lines:
                if line.startswith("QUESTION:"):
                    q_text = line.replace("QUESTION:", "").strip()
                elif line.startswith("ANSWER:"):
                    a_text = line.replace("ANSWER:", "").strip()

            if q_text and a_text:
                rows.append({
                    "question": q_text,
                    "ground_truth": a_text,
                    "source": chunk_source,
                    "reference_chunk": chunk_text,
                })
        except Exception as e:
            print(f"    ✗ Failed: {e}")

        # Rate-limiting pauses
        if (i + 1) % batch == 0 and (i + 1) < total:
            print(f"  ⏸ Batch pause ({batch_pause}s) to stay under TPM limit …", flush=True)
            time.sleep(batch_pause)
        else:
            time.sleep(sleep_between)

    print(f"[Testset] ✓ Generated {len(rows)} questions successfully.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    testset_df: pd.DataFrame,
    api_key: str,
    llm_wrapper: LangchainLLMWrapper,
    k: int = 5,
    sleep_between: float = 2.0,
) -> pd.DataFrame:
    """
    Feed each question through rag_chain.answer(), assemble the dataset,
    and score with RAGAS Faithfulness + Context Precision.
    """
    questions = testset_df["question"].tolist()
    ground_truths = testset_df["ground_truth"].tolist()

    answers = []
    contexts_list = []

    print(f"\n[RAG] Answering {len(questions)} questions via rag_chain …")
    for i, q in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] {q[:70]}…", flush=True)
        try:
            resp_text, ctxs = answer(q, api_key=api_key, k=k)
            answers.append(resp_text)
            # Truncate each context chunk to 400 chars so the RAGAS prompt
            # stays well under Groq's 8 000 TPM limit
            contexts_list.append([c["text"][:400] for c in ctxs[:3]])
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            answers.append("Error generating response.")
            contexts_list.append([])

        if i < len(questions) - 1:
            time.sleep(sleep_between)

    # Build HuggingFace Dataset for RAGAS
    eval_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(eval_dict)

    # Configure metrics with the LLM wrapper
    faithfulness.llm = llm_wrapper
    context_precision.llm = llm_wrapper

    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1, max_retries=3, timeout=60)

    print("\n[RAGAS] Scoring with Faithfulness + Context Precision (sequential) …", flush=True)
    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision],
        llm=llm_wrapper,
        run_config=run_config,
    )
    return results.to_pandas()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation for AWS RAG Chatbot")
    parser.add_argument("--limit", type=int, default=50,
                        help="Number of test questions (default: 50)")
    parser.add_argument("--batch", type=int, default=10,
                        help="Batch size for question generation (default: 10)")
    parser.add_argument("--output", type=str, default="testset_50.csv",
                        help="Output path for testset CSV")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip question generation; read from --output file")
    args = parser.parse_args()

    api_key = _resolve_api_key()

    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key, temperature=0.3)
    llm_wrapper = LangchainLLMWrapper(llm)

    # 1. Generate or load testset
    if args.skip_generate and os.path.exists(args.output):
        print(f"[Main] Loading testset from {args.output}")
        testset_df = pd.read_csv(args.output)
    else:
        testset_df = generate_testset(
            llm=llm,
            total=args.limit,
            batch=args.batch,
        )
        testset_df.to_csv(args.output, index=False)
        print(f"[Main] Testset saved to {args.output}")

    if testset_df.empty:
        print("[Main] No test questions available. Exiting.")
        sys.exit(1)

    # 2. Run evaluation
    results_df = run_evaluation(
        testset_df=testset_df,
        api_key=api_key,
        llm_wrapper=llm_wrapper,
    )

    # 3. Save & report
    results_df.to_csv("eval_results.csv", index=False)
    print("\n[Main] Results saved to eval_results.csv")
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for metric_name in ["faithfulness", "context_precision"]:
        if metric_name in results_df.columns:
            mean_val = results_df[metric_name].mean()
            print(f"  {metric_name.replace('_', ' ').title():<30s} {mean_val:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
