# TriEval-RAGHB (OpenAI Edition) — Hallucination Benchmark for RAG Under an Abstention Policy

A **30-sample controlled benchmark** that compares **three hallucination/groundedness evaluators** on identical `(question, context, answer)` logs using OpenAI models:

1) **RAGAS Faithfulness** (collections scorer API: `Faithfulness(...).ascore`)  
2) **DeepEval HallucinationMetric** (OpenAI judge model via `GPTModel`)  
3) **LLM-as-Judge Groundedness** (OpenAI judge with strict JSON output)

This benchmark enforces an **abstention policy** (fixed fallback response) to study evaluator behavior when a RAG system **refuses** instead of hallucinating.

---

## Related paper

**Benchmarking Hallucination Evaluation for RAG Under an Abstention Policy: A Controlled 30-Query Study with RAGAS, DeepEval, and LLM-as-Judge**  
Lamhot Siagian — Jan 2026 — DOI: `10.13140/RG.2.2.23948.78726`

Paper (ResearchGate):  
https://www.researchgate.net/publication/399331938_Benchmarking_Hallucination_Evaluation_for_RAG_Under_an_Abstention_Policy_A_Controlled_30-Query_Study_with_RAGAS_DeepEval_and_LLM-as-Judge

---

## What’s included

- **Dataset**: 3 study cases × 10 queries each = **30 total**
  - `product_faq` (DataFlowX mini-doc)
  - `biography` (short bio snippet)
  - `transformer_def` (transformer definition snippet)
- Each case includes:
  - **5 grounded** questions (answerable from context)
  - **5 unsupported** questions (should abstain / fallback)

---

## Abstention Policy

The answer generator is forced to output EXACTLY:

```text
I don't know based on the provided documents.
```

when the context does not contain the answer.

---

## Requirements

- Python **3.9+**
- OpenAI API key set as `OPENAI_API_KEY`

---

## Install

```bash
pip install -U openai ragas deepeval datasets python-dotenv pydantic
```

---

## Run

```bash
export OPENAI_API_KEY="sk-..."
python hallucination_benchmark_openai_trieval.py \
  --gen_model gpt-4o-mini \
  --judge_model gpt-4o-mini \
  --out_dir results \
  --timeout 60 \
  --max_retries 3 \
  --ragas_concurrency 8
```

---

## CLI arguments

| Argument | Default | Description |
|---|---:|---|
| `--gen_model` | `gpt-4o-mini` | Model used to generate RAG answers under abstention policy |
| `--judge_model` | `gpt-4o-mini` | Model used by evaluators (RAGAS, DeepEval, and LLM-as-judge) |
| `--timeout` | `60` | Request timeout in seconds |
| `--out_dir` | `results` | Output directory |
| `--max_retries` | `3` | Retry attempts for RAGAS scoring (with exponential backoff) |
| `--ragas_concurrency` | `8` | Async concurrency for RAGAS scoring |

---

## Outputs

After a successful run:

- `results/hallucination_results.json`
- `results/hallucination_results.csv`

Each row contains:

- `case_id`, `label` (`grounded` / `unsupported`)
- `question`, `answer`, `ground_truth`
- `rule_hallucinated`  
  - `True` if label is `unsupported` and answer is **not** the fallback
- Evaluator fields:
  - `ragas_faithfulness` (higher is better)
  - `deepeval_hallucination` (lower is better)
  - `deepeval_non_hallucination` (1 - hallucination; higher is better)
  - `judge_groundedness` (higher is better)
  - `judge_label` (`grounded` / `hallucinated` / parse-related values)

---

## How it works

### Step 1 — Generate answers (RAG-style)
For each `(question, context)` pair:
- Answer **only** from context  
- Otherwise output the fixed abstention fallback

### Step 2 — Evaluate the same logs 3 ways
- **RAGAS Faithfulness**: async scoring via `ragas.metrics.collections.Faithfulness`
- **DeepEval HallucinationMetric**: OpenAI judge model + threshold (`0.5`)
- **LLM-as-Judge**: strict JSON output:
  - `score` in `[0..1]`
  - `label`: `grounded` or `hallucinated`
  - `explanation`: short rationale

---

## Reproducibility notes

- Contexts are embedded directly in the script (fully controlled)
- Generator uses `temperature=0.0`
- Evaluators share the same `--judge_model`

Judge-based scoring may still vary slightly due to model-side nondeterminism.

---

## Troubleshooting

### `OPENAI_API_KEY is not set`
```bash
export OPENAI_API_KEY="sk-..."
```

### RAGAS returns NaNs / timeouts
Try:
- Increase `--timeout`
- Reduce `--ragas_concurrency`
- Increase `--max_retries`

Example:
```bash
python hallucination_benchmark_openai_trieval.py --timeout 120 --ragas_concurrency 4 --max_retries 5
```

### LLM-as-judge JSON parse errors
- Keep `temperature=0.0`
- Use a stronger judge model
- Tighten the system prompt (“Return JSON only. No markdown.”)

---

## How to cite

### BibTeX
```bibtex
@article{siagian2026trieval,
  title   = {Benchmarking Hallucination Evaluation for RAG Under an Abstention Policy: A Controlled 30-Query Study with RAGAS, DeepEval, and LLM-as-Judge},
  author  = {Siagian, Lamhot},
  year    = {2026},
  month   = jan,
  doi     = {10.13140/RG.2.2.23948.78726},
  note    = {Preprint, ResearchGate},
  url     = {https://www.researchgate.net/publication/399331938_Benchmarking_Hallucination_Evaluation_for_RAG_Under_an_Abstention_Policy_A_Controlled_30-Query_Study_with_RAGAS_DeepEval_and_LLM-as-Judge}
}
```

### IEEE
```text
L. Siagian, “Benchmarking Hallucination Evaluation for RAG Under an Abstention Policy: A Controlled 30-Query Study with RAGAS, DeepEval, and LLM-as-Judge,” preprint, Jan. 2026, doi: 10.13140/RG.2.2.23948.78726.
```

---

## License
MIT
