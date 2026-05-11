import numpy as np

from openai import AsyncOpenAI
import os 
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

#--------EMBEDDER CLASS ---------------------
class Embedder:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
        max_retries = 3,
        timeout = 30.0,
    )

    async def embed_one(self, text:str) -> list[float]:
        """Embed a single piece of text. Returns a vector (list of floats).
        float = a decimal number (e.g. 0.0234, -0.891)
        vector = a list of floats that represents the meaning of the text
        """

        text = text.replace("\n", " ").strip()

        res = await self.client.embeddings.create(
            model =EMBEDDING_MODEL,
            input = text,
        )
        return res.data[0].embedding

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in ONE API call - much cheaper than calling embed_one N times.
        Why batch? OpenAI charges per token. One API call has less overhead then N calls.
        You get back one vector per input text, in the same order as you inputs.
        """

        cleaned = [t.replace("\n", " ").strip() for t in texts]

        res = await self.client.embeddings.create(
            model = EMBEDDING_MODEL,
            input = cleaned,
        )
        sorted_data = sorted(res.data, key= lambda x : x.index)

        return [item.embedding for item in sorted_data]
#--------Math Functions---------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate how similar two vectors are. Returns a score from -1.0 to 1.0.

    cosine similarity = dot product of two vectors / (magnitude of a x magnitude of b)
    dot product (np.dot) = multiply each pair of numbers then sum them all up
    magnitude (np.linalg.norm) = then "length" of vector in space

    Why cosine instead of plain distance? 
    Cosine measures Direction(meaning) not length(how long the text is).
    "cat" and "the cat sat on the mat" should be similar even though one is longer/
    """
    #for fast maths
    vec_a = np.array(a)
    vec_b = np.array(b)

    #dot(a,b)/ (||a|| x ||b||)
    dot_product = np.dot(vec_a, vec_b)
    
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return float(dot_product / (magnitude_a * magnitude_b))
    
def find_most_similar(
    query_vector:list[float],
    candidate_vectors: list[list[float]],
    top_k : int = 5,

)-> list[tuple[int, float]]:
    """Find the top-K most similar vectors to a query vector.
    query_vector = the embedding of the user's question
    candidate_vectors = embeddings of all your documents/chunks
    top_k = how wmany results to return

    Returns a list of (index, score) tupples, sorted best-first.
    tuple = a pair of values that can't be changed like(3, 0.87)
    index = which candidate matched (so you can look up the original text)
    score = how similar it was(0.0 to 1.0)
    """

    scores = [
        (i, cosine_similarity(query_vector, candidate))
        for i, candidate in enumerate(candidate_vectors)
        #enumerate = gives you (index, value) pairs as you loop
        #so i=0, candidate = vectors[0] then i = 1 candidate= vec[1]

    ]

    scores.sort(key=lambda x:x[1],reverse=True)
    return scores[:top_k]

   # ─── SINGLETON ───────────────────────────────────────────────────────────────
# Create ONE instance that the whole app imports and reuses.
# "from app.core.embeddings import embedder" gives you this instance.

embedder = Embedder() 

