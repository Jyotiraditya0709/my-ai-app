# The eval harness (framework/engine) is the reusable core that:
# 1. Takes a list of test cases (input + expected output)
# 2. Runs your prompt function on each input
# 3. Uses a judge function to check if the output is correct
# 4. Returns accuracy %, latency, cost, and which cases failed
#
# You write this ONCE and reuse it for every eval you ever run.

import asyncio
import time
from dataclasses import dataclass   
# dataclass = decorator that auto-generates __init__, __repr__ for a class
# like a lightweight Pydantic model for simple data containers

from app.core.llm import llm


@dataclass
class EvalCase:
    """One test case: the input text and the expected correct output."""
    input: str      # the text to classify/process
    expected: str   # what the correct answer should be


async def run_eval(
    name: str,                  # label for this eval run (e.g. "v1 (casual)")
    prompt_fn,                  # function that takes input → returns prompt string
    judge_fn,                   # function that takes (actual, expected) → True/False
    cases: list[EvalCase],      # list of test cases to evaluate
    model: str = "gpt-4o-mini",
) -> dict:
    """Run a prompt function against all test cases and return metrics (measurements)."""
    
    results = []    # will hold one dict per test case
    
    for case in cases:
        start = time.time()
        
        try:
            # Build the prompt using the prompt function
            prompt = prompt_fn(case.input)
            
            # Call the LLM and get a response
            res = await llm.complete(prompt, model=model)
            
            # Use the judge function to check if the answer is correct
            passed = judge_fn(res.content, case.expected)
            
            results.append({
                "input": case.input,
                "expected": case.expected,
                "actual": res.content,          # what the model actually said
                "passed": passed,               # True or False
                "latency_ms": (time.time() - start) * 1000,
                "cost_usd": res.cost_usd,
            })
            
        except Exception as e:
            # If any call fails, record it as a failure rather than crashing
            results.append({
                "input": case.input,
                "expected": case.expected,
                "actual": str(e),    # the error message
                "passed": False,
                "latency_ms": (time.time() - start) * 1000,
                "cost_usd": 0.0,
            })
    
    # Calculate summary metrics (measurements)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])   
    # sum() with a generator = count how many are True
    
    failures = [r for r in results if not r["passed"]]
    
    return {
        "name": name,
        "accuracy": passed / total,           # e.g. 0.85 = 85% correct
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / total,
        "total_cost_usd": sum(r["cost_usd"] for r in results),
        "passed": passed,
        "total": total,
        "failures": failures,    # the specific cases that got wrong answers
    }