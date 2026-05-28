from app.core.llm import LLMResponse
from textwrap import dedent
from app.db.vector_store import SearchResult

def summarize_for_engineer(article: str) -> str:
    """Returns a prompt that asks a concise engineering-focused summary."""
    return f"""You are a senior engineer summarizing technical content for your team.
    
    Summarize the following in 2-3 sentences. Focus on:
    - What problem it solves
    - Key technical tradeoffs
    - When you would or wouldn't use it

    Article:
    {article}
    """

def rag_answer(question: str, context_chunks: list[str]) -> str:
    """Returns a RAG(Retrieval Augmented Generation) prompt with hallucination guard.

    Hallucination guard - instruction that tells the model to say 'I don't know'
    instead of making up an answer when the context doesn't cover the question.
    """

    context = "\n---\n".join(context_chunks)

    return f"""Answer the question using ONLY the provided context.
    If the answer is not in the context, say exactly: "I don't know based on the provided context."
    Do not use you training knowledge. Do not make up information.

    Context:
    {context}

    Question: {question}
    Answer: """

def few_shot_classifier(text: str, examples: list[dict])->str:
    """Returns a few-shot classification prompt.

    Few-shot = showing the model examples of input-> output paires before asking it 
    to classify new input. More reliable than zero-shot(no examples).
    """
    examples_str = "\n".join([
        f"Text:{ex['text']}\nLabel:{ex['label']}"
        for ex in examples
    ])
    return f"""Classify the following text. Use only the labels shown in the examples.

    Examples:
    {examples_str}

    Text to classify: {text}
    Label:"""

def chain_of_thought(problem:str)->str:
        """Returns a chain-of-thought prompt.

        Chain-of-thought = asking the model to show its reasoning step by step
        before giving the final answer. Significantly improves accuracy on 
        complex reasoning tasks.
        """

        return f"""Solve this step by step. Show your reasoning before giving the final answer.

        Problem: {problem}

        Let me think through this step by step:"""

def extract_json(text:str, schema_description: str) -> str:
    """Returns a prompt that extracts structured data from unstructured text.

    Unstructured text = plain text with no fixed format (emails, articals, etc.)
    Structured data = data in a predictable format(JSON with defined fields)
    """

    return f"""Extract imformation from the text below and return it as JSON.

Schema (the fields to extract):
{schema_description}

Text:
{text}
Return only valid JSON. No explanation, no markdown formatting."""
   


def classify_with_cot(text: str, categories: list[str]) -> str:
    """Chain-of-thought classification with explicit reasoning steps.

    CoT (Chain-of-Thought) = asking the model to show reasoning before answering.
    Structured CoT = giving it specific steps to follow (better than "think step by step").
    """
    cats = ", ".join(categories)
    return dedent(f"""
        Classify the text into one of: {cats})

        Think about it:
        1. Identify key signals in the text
        2. Match against each category
        3. Pick the best fit

        Format your response as:
        Signals:
        Answer: 

        Text: {text}
        """).strip()

def extract_with_schema(text:str, schema_description: str)-> str:
    """Structured extraction prompt with strict rules to prevent hallucination.
    
    Extraction = pulling specific fields out of unstructured (free-form) text.
    The rules prevent the model from inventing values that aren't in the text.
    """
    
    return dedent(f"""
        Extract structured data from the text below.

        Schema:
        {schema_description}

        Rules:
        - Only extract what's explicitly stated
        - Use null for missing fields, never invent values
        - Dates must be ISO 8601 format (e.g. 2026-03-14)

        Text:
        <
        {text}
        >>>
    """).strip()

    def revise_based_on_critique(original: str, critique: str) -> str:
        """Prompt for the revision step in a generate→critique→revise loop.
    
    Used in self-reflection workflows where an editor found issues.
    "Do not defend the original" prevents the model from arguing back.
    """
    return dedent(f"""
        You wrote this draft:
        {original}

        A reviewer found these issues:
        {critique}
 
        Rewrite the draft addressing every issue.
        Do not defend the original — just fix it.
    """).strip()

def rag_with_citations(question:str, chunks: list[SearchResult]) -> str:
    """Build a citiation-aware prompt. Each chunk becomes a numberd source."""
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source_lable = chunk.source
        if hasattr(chunk, "chunk_index"):
            source_lable = f"{chunk.source}, chunk {chunk.chunk_index}"
        formatted.append(f"[{i}] ({source_lable}) \\n {chunk.text}")

        context_block = "\n\n".join(formatted)

    return dedent(f"""
            You are answering question based on provided context.
            
            Rules:
            - Use only the context below.
            - Cite sources in square brackets like [1], [2] after each claim.
            - If context doesn't contain the answer,
            say "I don't know based on the provided context."
            - Be conciese: 2-4 sentences.

            Context: 
            {context_block}

            Answer:

            """).strip()