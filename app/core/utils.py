import tiktoken

def truncate(text:str, max_len:int)-> str:
    """Truncate text to max_len characters, adding '...' if truncated"""
    if text is None:
        raise ValueError("Text can't be None")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."

def count_tokens(text: str, model: str= "gpt-4")-> int:
    """Count tokens in text using tiktoken -same tokenizer the model uses."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def build_prompt(context: str, question: str) -> str:
    """Build a RAG prompt from retrieved context and user question."""
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    tokens = count_tokens(prompt)
    if tokens > 3000:
        raise ValueError(f"Prompt too long: {tokens} tokens")
    return prompt
