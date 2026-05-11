# This file compares THREE different ways to prompt for sentiment classification.
# The point is NOT to write the best prompt — it's to MEASURE which is better.
# Real numbers > opinions.

import asyncio
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert = adds the project root to Python's module search path
# so "from evals.harness import..." works when running this file directly

from evals.harness import run_eval, EvalCase

# 20 test cases — input text + correct sentiment label
CASES = [
    EvalCase("This product saved my weekend!", "positive"),
    EvalCase("Terrible customer service, never again.", "negative"),
    EvalCase("The package arrived on time.", "neutral"),
    EvalCase("I have mixed feelings about this.", "neutral"),
    EvalCase("Life-changing, truly amazing.", "positive"),
    EvalCase("Worst purchase I've ever made.", "negative"),
    EvalCase("It works as described.", "neutral"),
    EvalCase("Absolutely love it, 10/10.", "positive"),
    EvalCase("Garbage. Broke on day 2.", "negative"),
    EvalCase("Does what it says, nothing more.", "neutral"),
    EvalCase("This is hands down the best.", "positive"),
    EvalCase("Don't waste your money.", "negative"),
    EvalCase("Average product, fair price.", "neutral"),
    EvalCase("Exceeded all my expectations!", "positive"),
    EvalCase("Poorly made and overpriced.", "negative"),
    EvalCase("Nothing special either way.", "neutral"),
    EvalCase("Outstanding quality throughout.", "positive"),
    EvalCase("Slightly disappointing experience.", "negative"),
    EvalCase("It's fine, I guess.", "neutral"),
    EvalCase("This changed how I work.", "positive"),
]


# ─── JUDGE FUNCTION ──────────────────────────────────────────────────────────
# Judge = a function that checks if the model's answer is correct.
# Returns True (correct) or False (wrong).

def judge(actual: str, expected: str) -> bool:
    return expected in actual.lower()
    # actual.lower() = convert response to lowercase before checking
    # so "Positive", "POSITIVE", "positive" all match "positive"


# ─── THREE PROMPT STRATEGIES ─────────────────────────────────────────────────
# Each is a function that takes the text and returns a different prompt string.
# Same question, different framing — we measure which framing the model handles best.

def prompt_v1(text: str) -> str:
    # v1 = casual, minimal instructions
    # Problem: model might respond with a full sentence instead of one word
    return f'Classify sentiment: "{text}". Answer:'


def prompt_v2(text: str) -> str:
    # v2 = structured, explicit format constraints (rules for output format)
    # Better because it tells the model EXACTLY what format to use
    return f"""Classify the sentiment of the text below.
Answer with only one word: positive, negative, or neutral.

Text: "{text}"
Sentiment:"""


def prompt_v3_cot(text: str) -> str:
    # v3 = Chain-of-Thought (CoT) — asks the model to reason before answering
    # CoT = showing reasoning steps improves accuracy on ambiguous cases
    # "mixed feelings" is hard — CoT helps the model think it through
    return f"""Text: "{text}"

First, note any strong emotional signals or qualifiers.
Then classify as exactly one of: positive, negative, neutral.

Format:
Signals: ...
Answer: """


# ─── MAIN: run all three and compare ─────────────────────────────────────────

async def main():
    # Run all three eval variants
    v1 = await run_eval("v1 (casual)", prompt_v1, judge, CASES)
    v2 = await run_eval("v2 (structured)", prompt_v2, judge, CASES)
    
    # v3 needs a custom judge — extract only the part after "Answer:"
    # because the response includes "Signals: ..." before the answer
    v3 = await run_eval(
        "v3 (CoT)",
        lambda t: prompt_v3_cot(t),   # lambda = inline function
        lambda a, e: e in a.lower().split("answer:")[-1],
        # split("answer:")[-1] = take everything AFTER the last "Answer:" label
        # so "Signals: great | Answer: positive" → checks "positive" only
        CASES
    )
    
    # Print results for all three
    for result in [v1, v2, v3]:
        print(f"\n{result['name']}")
        print(f"  Accuracy:     {result['accuracy']:.0%}")   # .0% = format as percentage
        print(f"  Avg latency:  {result['avg_latency_ms']:.0f}ms")
        print(f"  Total cost:   ${result['total_cost_usd']:.4f}")
        
        if result["failures"]:
            print(f"  Failures ({len(result['failures'])}):")
            for f in result["failures"][:3]:   # [:3] = show max 3 failures
                print(f"    '{f['input']}' expected={f['expected']} got={f['actual'][:40]}")
                # [:40] = truncate (shorten) actual response to first 40 characters

asyncio.run(main())