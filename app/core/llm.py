from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from typing import AsyncIterator, Type, TypeVar
import os, time, asyncio
from dotenv import load_dotenv
from collections import Counter

import asyncio

load_dotenv() 

T = TypeVar("T", bound=BaseModel)

#----- RESPONSE MODEL ---------------------------
class LLMResponse(BaseModel):
    content:str
    model: str
    input_tokens : int
    output_tokens: int
    latency_ms: float
    cost_usd: float

#------COST TABLE ------------------
COSTS_PER_1M = {
    "gpt-4o" :                   (2.50, 10.00),
    "gpt-4o-mini":               (0.15,   0.60),
    "gpt-4":                     (30.00,  60.00),
    "claude-3-5-sonnet-20241022":(3.00,  15.00),
    "claude-3-5-haiku-20241022": (0.80,   4.00),
    "claude-3-opus-20240229":    (15.00,  75.00),
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for one API call."""
    if model not in COSTS_PER_1M:
        return 0.0
    input_price, output_price = COSTS_PER_1M[model]
    return (input_tokens * input_price + output_price * output_tokens) / 1_000_000

# THE LLM CLIENT CLASS
class LLMClient:
    def __init__(self):
        
        self.openai = AsyncOpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            max_retries=3,
            timeout=30.0,
        )
        self.anthropic = AsyncAnthropic(
            api_key = os.getenv("ANTHROPIC_API_KEY"),
            max_retries=3,
            timeout=30.0,
        )
    
    async def complete(
        self,
        prompt:str,
        system: str = "You are a helpful AI assistant.",
        model: str = "gpt-4o-mini",
        temperature:float=0.7,
        max_tokens:int = 1000,
    ) -> LLMResponse:
        """Send a prompt and get a complete response back at once."""
        start = time.time()

        if model.startswith("gpt") or model.startswith("o"):
            res = await self.openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content":system},
                    {"role": "user", "content": prompt},
                ],
                temperature= temperature,
                max_tokens = max_tokens,
            )
            input_tokens    = res.usage.prompt_tokens
            output_tokens   = res.usage.completion_tokens
            content         = res.choices[0].message.content
            actual_model = res.model

        elif model.startswith("claude"):
            res = await self.anthropic.messages.create(
                model = model,
                system = system,
                messages= [{"role":"user", "content":prompt}],
                temperature= temperature,
                max_tokens=max_tokens,
            )   
            input_tokens  = res.usage.input_tokens
            output_tokens = res.usage.output_tokens
            content       = res.content[0].text
            actual_model  = res.model

        else:
            raise ValueError(f"Unknown model: {model}")


        return LLMResponse(
            content=content,
            model=actual_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round((time.time() - start) * 1000),
            cost_usd = calculate_cost(actual_model, input_tokens, output_tokens),
        )
    
    async def stream(
        self,
        prompt: str,
        system:str = "You are a helpful AI assistant.",
        model:str = "gpt-4o-mini",

    )-> AsyncIterator[str]:
        """Stream tokens one by one as they're generated.

        AsyncIterator = async generator - use with: async for token in llm.stream(...)
        Each iteration yields (produces) one small string fragement(token).
        """

        if model.startswith("gpt") or model.startswith("o"):
            stream = await self.openai.chat.completions.create(
                model=model,
                messages=[
                   {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}, 
                ],
                stream = True,
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content

                if token:
                    yield token
        elif model.startswith("claude"):
            async with self.anthropic.messages.stream(
                model=model,
                system=system,
                message=[{"role":"user", "content": prompt}],
                max_tokens = 1000,
            ) as stream: 
                async for text in stream.text_stream:
                    yield text

    async def complete_structured(
        self, 
        prompt: str,
        schema: Type[T],
        model: str = "gpt-4o-mini",
        system: str = "You are a helpful AI assistant. Always responds with valid JSON.",
    ) -> T:
        """Get a response parsed into a typed Pydantic object instead of raw text.
        
        Example: pass schema=Sentiment and get back a Sentiment object
        with .sentiment and .confidence attributes - not a raw string.
        """
        
        import json 
        
        schema_str = json.dumps(schema.model_json_schema(), indent =2)

        structured_prompt = f"{prompt}\n\nRespond with JSON matching this schema:\n{schema_str}"

        res = await self.complete(
            prompt=structured_prompt,
            system=system,
            model=model,
            temperature=0.1
        ) 

        raw = res.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return schema.model_validate(json.loads(raw))

# ================== ADVANCED METHODS ==================
# ── METHOD 3: complete_with_consistency ──────────────────
    async def complete_with_consistency(
        self,
        prompt: str,
        *,
        schema: Type[T],
        system: str = "You are a helpful AI assistant.",
        model: str = "gpt-4o-mini",
        n_samples: int = 5,
        sample_temperature: float = 0.7,
        answer_field: str = "final_answer",
    ) -> tuple[T, float]:
        tasks = [
            self.complete_structured(
                prompt, schema=schema, system=system,
                model=model, temperature=sample_temperature,
            )
            for _ in range(n_samples)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid = [r for r in results if isinstance(r, schema)]
        if not valid:
            raise RuntimeError("All samples failed")
        answers = [str(getattr(r, answer_field)) for r in valid]
        counter = Counter(answers)
        winner_str, winner_count = counter.most_common(1)[0]
        consistency = winner_count / len(valid)
        winner = next(r for r in valid if str(getattr(r, answer_field)) == winner_str)
        return winner, consistency

    # ── METHOD 4: complete_with_reflection ───────────────────
    async def complete_with_reflection(
        self,
        prompt: str,
        *,
        schema: Type[T],
        critic_schema: Type[BaseModel],
        system: str = "You are a helpful AI assistant.",
        model: str = "gpt-4o-mini",
        max_iterations: int = 2,
    ) -> T:
        draft = await self.complete_structured(
            prompt, schema=schema, system=system, model=model
        )
        for _ in range(max_iterations):
            critique_prompt = f"""Review this draft answer strictly.

Original prompt: {prompt}

Draft answer: {draft.model_dump_json()}
Find concrete issues. If none, return severity='none'."""

            critique = await self.complete_structured(
                critique_prompt, schema=critic_schema, model=model,
            )
            if getattr(critique, "severity", None) == "none":
                return draft
            revise_prompt = f"""Revise based on the critique.

Original prompt: {prompt}
Previous answer: {draft.model_dump_json()}
Issues: {getattr(critique, 'issues_found', [])}
Suggested fix: {getattr(critique, 'suggested_fix', '')}

Provide a revised answer."""

            draft = await self.complete_structured(
                revise_prompt, schema=schema, model=model,
            )
        return draft


# singleton — one instance for the whole app
llm = LLMClient()


"""

# load_dotenv()

# openai_client = AsyncOpenAI(
#     api_key = os.getenv("OPENAI_API_KEY"),
#     max_retries = 3, 
#     timeout=30.0,
# )

# anthropic_client = AsyncAnthropic(
#     api_key = os.getenv("ANTHROPIC_API_KEY"),
#     max_retries=3,
#     timeout=30.0,
# )

# async def call_openai(prompt:str, system:str="You are helpful.") -> str:
#     res = await openai_client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role":"system", "content":system},
#             {"role":"user", "content": prompt},
#         ],
#         max_tokens=500,
#     )
#     return res.choices[0].message.content

# async def call_anthropic(prompt: str, system: str = "You are helpful.") -> str:
#     res= await anthropic_client.messages.create(
#         model="claude-3-haiku-20240307",
#         system= system,
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=500,
#     )
#     return res.content[0].text
"""

