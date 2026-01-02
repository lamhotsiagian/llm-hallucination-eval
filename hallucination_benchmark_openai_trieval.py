#!/usr/bin/env python3
"""
hallucination_benchmark_openai_trieval.py

TriEval-RAGHB (OpenAI edition): 30-sample controlled hallucination benchmark
comparing 3 evaluators on identical (question, context, answer) logs:

  (1) RAGAS Faithfulness  (collections scorer API: Faithfulness(...).ascore)
  (2) DeepEval HallucinationMetric (OpenAI judge model via GPTModel)
  (3) LLM-as-judge Groundedness (OpenAI judge with JSON output)

Outputs:
  - results/hallucination_results.json
  - results/hallucination_results.csv

Install:
  pip install -U openai ragas deepeval datasets python-dotenv pydantic

Run:
  export OPENAI_API_KEY="sk-..."
  python hallucination_benchmark_openai_trieval.py \
    --gen_model gpt-4o-mini \
    --judge_model gpt-4o-mini \
    --out_dir results \
    --timeout 60 \
    --max_retries 3 \
    --ragas_concurrency 8
"""

import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

FALLBACK_ANSWER = "I don't know based on the provided documents."

# -----------------------------
# Study-case dataset (3 cases x 10 = 30)
# -----------------------------

STUDY_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "product_faq",
        "context": """\
Product: DataFlowX
Version: 3.2
Features:
- Real-time data streaming
- Kafka integration
- Role-based access control
Deployment:
- Cloud-hosted SaaS
- Supports Kubernetes-based deployments
Release Date: March 2024
Supported Databases:
- PostgreSQL
- MySQL
""",
        "examples": [
            # grounded (5)
            {"label": "grounded", "question": "What streaming capability does DataFlowX provide?",
             "ground_truth": "DataFlowX provides real-time data streaming."},
            {"label": "grounded", "question": "Which message system does DataFlowX integrate with?",
             "ground_truth": "DataFlowX integrates with Kafka."},
            {"label": "grounded", "question": "What kind of access control does DataFlowX support?",
             "ground_truth": "DataFlowX supports role-based access control."},
            {"label": "grounded", "question": "Name one type of deployment supported by DataFlowX.",
             "ground_truth": "DataFlowX supports cloud-hosted SaaS deployments."},
            {"label": "grounded", "question": "Which relational databases are listed as supported by DataFlowX?",
             "ground_truth": "DataFlowX supports PostgreSQL and MySQL."},

            # unsupported (5)
            {"label": "unsupported", "question": "Does DataFlowX provide automated anomaly detection?",
             "ground_truth": "The documentation does not mention automated anomaly detection."},
            {"label": "unsupported", "question": "Is DataFlowX compliant with SOC 2 Type II?",
             "ground_truth": "The documentation does not mention any SOC 2 compliance."},
            {"label": "unsupported", "question": "Does DataFlowX include a built-in dashboard for business users?",
             "ground_truth": "The documentation does not mention any built-in dashboard for business users."},
            {"label": "unsupported", "question": "Can DataFlowX run as an on-premises Windows service?",
             "ground_truth": "The documentation does not mention on-premises Windows service support."},
            {"label": "unsupported", "question": "What is the maximum throughput of DataFlowX in events per second?",
             "ground_truth": "The documentation does not specify any maximum throughput in events per second."},
        ],
    },
    {
        "case_id": "biography",
        "context": """\
Sarah Lin is a data scientist at GreenTech Labs in Austin.
She specializes in climate data modeling.
Her work focuses on predicting extreme weather events using historical data.
She previously worked at EcoAI in Seattle for four years.
She has a Master's degree in Statistics.
She frequently collaborates with environmental scientists and policy makers.
""",
        "examples": [
            # grounded (5)
            {"label": "grounded", "question": "Where does Sarah Lin currently work?",
             "ground_truth": "Sarah Lin currently works at GreenTech Labs in Austin."},
            {"label": "grounded", "question": "What is Sarah Lin's area of specialization?",
             "ground_truth": "She specializes in climate data modeling."},
            {"label": "grounded", "question": "What kind of events does Sarah try to predict in her work?",
             "ground_truth": "She focuses on predicting extreme weather events."},
            {"label": "grounded", "question": "Where did Sarah Lin work before joining GreenTech Labs?",
             "ground_truth": "She previously worked at EcoAI in Seattle."},
            {"label": "grounded", "question": "What is Sarah Lin's highest degree mentioned in the bio?",
             "ground_truth": "She has a Master's degree in Statistics."},

            # unsupported (5)
            {"label": "unsupported", "question": "Where did Sarah Lin complete her PhD studies?",
             "ground_truth": "The biography does not mention Sarah Lin completing any PhD studies."},
            {"label": "unsupported", "question": "How old is Sarah Lin?",
             "ground_truth": "The biography does not mention Sarah Lin's age."},
            {"label": "unsupported", "question": "What is Sarah Lin's annual salary at GreenTech Labs?",
             "ground_truth": "The biography does not mention Sarah Lin's salary."},
            {"label": "unsupported", "question": "Is Sarah Lin married?",
             "ground_truth": "The biography does not mention anything about Sarah Lin's marital status."},
            {"label": "unsupported", "question": "Which university did Sarah attend for her undergraduate degree?",
             "ground_truth": "The biography does not mention where she completed her undergraduate degree."},
        ],
    },
    {
        "case_id": "transformer_def",
        "context": """\
A transformer model uses self-attention to process sequences.
Self-attention allows each token to attend to all other tokens in the sequence.
Transformers replaced recurrent neural networks in many natural language processing tasks.
Transformers are commonly used for machine translation, text summarization, and question answering.
The attention mechanism can be computed in parallel, which makes transformers efficient on modern hardware.
""",
        "examples": [
            # grounded (5)
            {"label": "grounded", "question": "What mechanism do transformer models use to process sequences?",
             "ground_truth": "Transformer models use self-attention to process sequences."},
            {"label": "grounded", "question": "What does self-attention allow each token to do?",
             "ground_truth": "Self-attention allows each token to attend to all other tokens in the sequence."},
            {"label": "grounded", "question": "Which older family of models have transformers replaced in many NLP tasks?",
             "ground_truth": "Transformers have replaced recurrent neural networks in many NLP tasks."},
            {"label": "grounded", "question": "Name one application area where transformers are commonly used.",
             "ground_truth": "Transformers are commonly used for tasks such as machine translation, text summarization, or question answering."},
            {"label": "grounded", "question": "Why are transformers efficient on modern hardware?",
             "ground_truth": "Transformers are efficient because the attention mechanism can be computed in parallel."},

            # unsupported (5)
            {"label": "unsupported", "question": "Who first invented the transformer architecture?",
             "ground_truth": "The provided context does not specify who invented transformers."},
            {"label": "unsupported", "question": "In which year was the original transformer paper published?",
             "ground_truth": "The provided context does not specify the year of the original transformer paper."},
            {"label": "unsupported", "question": "How many layers does a typical transformer encoder have?",
             "ground_truth": "The provided context does not specify how many layers a typical transformer encoder has."},
            {"label": "unsupported", "question": "What is the exact complexity of self-attention in terms of sequence length?",
             "ground_truth": "The provided context does not specify the exact computational complexity of self-attention."},
            {"label": "unsupported", "question": "Which optimizer is most commonly used to train transformer models?",
             "ground_truth": "The provided context does not mention any optimizer used to train transformers."},
        ],
    },
]

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def extract_json_object(text: str) -> Optional[str]:
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return text[i:j+1]

def avg_ignore_nan(vals: List[float]) -> float:
    clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(sum(clean) / len(clean)) if clean else float("nan")

def is_hallucinated_rule(label: str, answer: str) -> bool:
    return (label == "unsupported") and (answer.strip() != FALLBACK_ANSWER)

# -----------------------------
# OpenAI wrapper (sync + async)
# -----------------------------

class OpenAICompat:
    def __init__(self, model: str, timeout: int):
        from openai import OpenAI, AsyncOpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()

    async def achat(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = await self.async_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (r.choices[0].message.content or "").strip()

# -----------------------------
# 1) Generate answers
# -----------------------------

def generate_rag_answer(gen: OpenAICompat, question: str, context: str) -> str:
    system = (
        "You are a QA assistant for a Retrieval-Augmented Generation (RAG) system.\n"
        "You MUST answer ONLY using the provided context.\n"
        f"If the context does not contain the answer, respond EXACTLY with:\n\"{FALLBACK_ANSWER}\""
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return gen.chat(system=system, user=user, temperature=0.0)

def run_benchmark(gen: OpenAICompat) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case in STUDY_CASES:
        ctx = case["context"]
        cid = case["case_id"]
        for ex in case["examples"]:
            ans = generate_rag_answer(gen, ex["question"], ctx)
            rows.append({
                "case_id": cid,
                "label": ex["label"],
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
                "answer": ans,
                "contexts": [ctx],
            })
    return rows

# -----------------------------
# 2) RAGAS Faithfulness (collections scorer API)
# -----------------------------

async def ragas_faithfulness_scores(
    rows: List[Dict[str, Any]],
    judge_model: str,
    timeout: int,
    max_retries: int,
    concurrency: int,
) -> Tuple[List[float], float, int]:
    """
    Uses the newer scorer API:
      scorer = Faithfulness(llm=llm)
      await scorer.ascore(user_input=..., response=..., retrieved_contexts=[...])
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    ragas_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=timeout)
    llm = llm_factory(judge_model, client=ragas_client)

    scorer = Faithfulness(llm=llm)
    sem = asyncio.Semaphore(concurrency)

    async def score_one(r: Dict[str, Any]) -> float:
        async with sem:
            for attempt in range(max_retries):
                try:
                    res = await scorer.ascore(
                        user_input=r["question"],
                        response=r["answer"],
                        retrieved_contexts=r["contexts"],
                    )
                    return float(res.value)
                except Exception:
                    if attempt == max_retries - 1:
                        return float("nan")
                    await asyncio.sleep(0.5 * (2 ** attempt))
            return float("nan")

    scores = await asyncio.gather(*[score_one(r) for r in rows])
    nan_count = sum(1 for s in scores if isinstance(s, float) and math.isnan(s))
    return scores, avg_ignore_nan(scores), nan_count

# -----------------------------
# 3) DeepEval HallucinationMetric (OpenAI judge)
# -----------------------------

def deepeval_hallucination_scores(rows: List[Dict[str, Any]], judge_model: str) -> Tuple[List[float], float]:
    from deepeval.metrics import HallucinationMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.models import GPTModel

    model = GPTModel(model=judge_model)
    metric = HallucinationMetric(model=model, threshold=0.5)

    scores: List[float] = []
    for r in rows:
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            context=r["contexts"],
        )
        metric.measure(tc)
        scores.append(float(metric.score))

    avg = float(sum(scores) / len(scores)) if scores else float("nan")
    return scores, avg

# -----------------------------
# 4) LLM-as-judge groundedness (JSON)
# -----------------------------

def llm_judge_groundedness(judge: OpenAICompat, question: str, answer: str, contexts: List[str]) -> Dict[str, Any]:
    system = (
        "You are an evaluator for a RAG system.\n"
        "Decide if the ANSWER is fully supported by the CONTEXT.\n"
        "If any factual claim in the answer cannot be directly inferred from the context, it is hallucinated.\n"
        "Return ONLY JSON with keys: score (0..1), label ('grounded'|'hallucinated'), explanation (short)."
    )
    ctx = "\n\n".join(contexts)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return JSON only."
    )
    raw = judge.chat(system=system, user=user, temperature=0.0)
    js = extract_json_object(raw)
    if js is None:
        return {"score": 0.0, "label": "parse_error", "explanation": raw}

    try:
        data = json.loads(js)
        score = float(data.get("score", 0.0))
        label = str(data.get("label", "unknown")).lower()
        explanation = str(data.get("explanation", ""))
        score = max(0.0, min(1.0, score))
        if label not in {"grounded", "hallucinated"}:
            label = "unknown"
        return {"score": score, "label": label, "explanation": explanation}
    except Exception:
        return {"score": 0.0, "label": "parse_error", "explanation": raw}

def judge_groundedness_scores(rows: List[Dict[str, Any]], judge: OpenAICompat) -> Tuple[List[float], List[str], float]:
    scores, labels = [], []
    for r in rows:
        res = llm_judge_groundedness(judge, r["question"], r["answer"], r["contexts"])
        scores.append(float(res["score"]))
        labels.append(str(res["label"]))
    avg = float(sum(scores) / len(scores)) if scores else float("nan")
    return scores, labels, avg

# -----------------------------
# Save results
# -----------------------------

def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_model", default="gpt-4o-mini")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--ragas_concurrency", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it first.")

    ensure_dir(args.out_dir)

    gen = OpenAICompat(model=args.gen_model, timeout=args.timeout)
    judge = OpenAICompat(model=args.judge_model, timeout=args.timeout)

    print("=== Running 30 samples. Provider=openai ===")
    rows = run_benchmark(gen)
    print(f"=== Ran {len(rows)} samples (expected 30). ===\n")

    # (1) RAGAS
    print("=== (1) RAGAS Faithfulness (collections scorer) ===")
    t0 = time.time()
    ragas_scores, ragas_avg, ragas_nan = asyncio.run(
        ragas_faithfulness_scores(
            rows=rows,
            judge_model=args.judge_model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            concurrency=args.ragas_concurrency,
        )
    )
    print(f"RAGAS avg faithfulness: {ragas_avg:.3f} (nan={ragas_nan})  time={time.time()-t0:.1f}s\n")

    # (2) DeepEval
    print("=== (2) DeepEval HallucinationMetric ===")
    de_scores, de_avg = deepeval_hallucination_scores(rows, args.judge_model)
    print(f"DeepEval avg hallucination: {de_avg:.3f} (lower is better)")
    print(f"DeepEval avg non-hallucination: {(1.0 - de_avg):.3f} (higher is better)\n")

    # (3) LLM-as-judge
    print("=== (3) LLM-as-judge Groundedness ===")
    jc_scores, jc_labels, jc_avg = judge_groundedness_scores(rows, judge)
    print(f"Judge avg groundedness: {jc_avg:.3f}\n")

    merged: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        merged.append({
            "index": i,
            "case_id": r["case_id"],
            "label": r["label"],
            "question": r["question"],
            "answer": r["answer"],
            "ground_truth": r["ground_truth"],
            "rule_hallucinated": is_hallucinated_rule(r["label"], r["answer"]),
            "ragas_faithfulness": float(ragas_scores[i]),
            "deepeval_hallucination": float(de_scores[i]),
            "deepeval_non_hallucination": 1.0 - float(de_scores[i]),
            "judge_groundedness": float(jc_scores[i]),
            "judge_label": str(jc_labels[i]),
        })

    print("=== Per-case summary (first 10 rows) ===")
    for row in merged[:10]:
        print(
            f"[{row['index']:02d}] {row['case_id']:<15} label={row['label']:<11} "
            f"rule={row['rule_hallucinated']} "
            f"RAGAS={row['ragas_faithfulness']:.3f} "
            f"DE(h)={row['deepeval_hallucination']:.3f} "
            f"Judge={row['judge_groundedness']:.3f} "
            f"Judge_label={row['judge_label']}"
        )

    out_json = os.path.join(args.out_dir, "hallucination_results.json")
    out_csv = os.path.join(args.out_dir, "hallucination_results.csv")
    save_json(out_json, merged)
    save_csv(out_csv, merged)

    print("\n✅ Results written to:")
    print(f"  JSON: {os.path.abspath(out_json)}")
    print(f"  CSV : {os.path.abspath(out_csv)}")
    print("\nDone.")

if __name__ == "__main__":
    main()
