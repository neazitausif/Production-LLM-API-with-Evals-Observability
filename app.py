
import os
import json
import time
import threading
from datetime import datetime, timezone
from typing import Generator

import requests
import chromadb

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:1.5b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434"
)

LOG_FILE = os.getenv(
    "LOG_FILE",
    "request_logs.jsonl"
)

# Illustrative local cost estimate.
# This is NOT an actual Ollama API charge.
COST_PER_1K_TOKENS = float(
    os.getenv("COST_PER_1K_TOKENS", "0.001")
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Tax RAG API",
    version="1.0.0"
)




# ============================================================
# OBSERVABILITY MIDDLEWARE
# ============================================================

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Record latency, token estimates, model and cache status."""

    start_time = time.perf_counter()

    # Default values for the request
    request.state.query = ""
    request.state.cache_hit = False
    request.state.input_tokens = 0
    request.state.output_tokens = 0

    try:
        response = await call_next(request)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Log only /chat requests
        if request.url.path == "/chat":
            write_log(
                query=request.state.query,
                latency_ms=latency_ms,
                input_tokens=request.state.input_tokens,
                output_tokens=request.state.output_tokens,
                cache_hit=request.state.cache_hit
            )

        return response

    except Exception:
        # Log failed /chat requests as well
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        if request.url.path == "/chat":
            write_log(
                query=request.state.query,
                latency_ms=latency_ms,
                input_tokens=request.state.input_tokens,
                output_tokens=request.state.output_tokens,
                cache_hit=request.state.cache_hit
            )

        raise

# ============================================================
# CACHE
# ============================================================

CACHE = {}
CACHE_LOCK = threading.Lock()


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="User tax-related question"
    )


# ============================================================
# TAX KNOWLEDGE BASE
# ============================================================

tax_documents = [
    {
        "id": "tax_01",
        "title": "Section 80C",
        "text": """
Section 80C provides eligible taxpayers with deductions for specified
investments and payments subject to applicable rules.
"""
    },
    {
        "id": "tax_02",
        "title": "Form 16",
        "text": """
Form 16 is a certificate issued by an employer showing salary income
and tax deducted at source (TDS) from an employee's salary.
"""
    },
    {
        "id": "tax_03",
        "title": "HRA",
        "text": """
House Rent Allowance, commonly called HRA, is a salary component that
may qualify for tax exemption subject to applicable conditions and rules.
"""
    },
    {
        "id": "tax_04",
        "title": "TDS",
        "text": """
Tax Deducted at Source, or TDS, is tax collected by the payer at the
time of making certain specified payments and deposited with the government.
"""
    },
    {
        "id": "tax_05",
        "title": "Income Tax Return",
        "text": """
An Income Tax Return, or ITR, is a form used by a taxpayer to report
income, deductions, taxes paid and other required information to the
Income Tax Department.
"""
    },
    {
        "id": "tax_06",
        "title": "Capital Gains",
        "text": """
Capital gains are profits or gains arising from the transfer of capital
assets. They may be classified as short-term or long-term according to
applicable tax rules.
"""
    },
    {
        "id": "tax_07",
        "title": "Advance Tax",
        "text": """
Advance tax is income tax paid in installments during the financial year
when the taxpayer's estimated tax liability meets the applicable threshold.
"""
    },
    {
        "id": "tax_08",
        "title": "New Tax Regime",
        "text": """
The new tax regime is one of India's income tax regimes. Under the new
tax regime, applicable tax rates, deductions and exemptions differ from
those available under the old tax regime. The exact rules and tax rates
may change between financial years.
"""
    },
    {
        "id": "tax_09",
        "title": "Old Tax Regime",
        "text": """
The old tax regime is an income tax regime that allows taxpayers to
claim various deductions and exemptions subject to applicable conditions.
"""
    },
    {
        "id": "tax_10",
        "title": "GST",
        "text": """
Goods and Services Tax, or GST, is an indirect tax imposed on the supply
of goods and services in India.
"""
    }
]


# ============================================================
# CHROMADB CONNECTION
# ============================================================

chroma_host = os.getenv("CHROMA_HOST")

if chroma_host:
    chroma_client = chromadb.HttpClient(
        host=chroma_host,
        port=8000
    )
else:
    chroma_client = chromadb.PersistentClient(
        path="./chroma_data"
    )

collection = chroma_client.get_or_create_collection(
    name="indian_tax_knowledge"
)


# ============================================================
# SEED KNOWLEDGE BASE
# ============================================================

collection.upsert(
    ids=[document["id"] for document in tax_documents],
    documents=[document["text"] for document in tax_documents],
    metadatas=[
        {"title": document["title"]}
        for document in tax_documents
    ]
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Simple token estimate used for observability.
    This is an approximation, not a tokenizer calculation.
    """

    if not text:
        return 0

    return max(1, len(text.split()))


def estimate_cost(
    input_tokens: int,
    output_tokens: int
) -> float:

    total_tokens = input_tokens + output_tokens

    return round(
        (total_tokens / 1000) * COST_PER_1K_TOKENS,
        6
    )


def retrieve_context(
    query: str,
    number_of_results: int = 3
):
    """
    Hybrid-style retrieval using Chroma semantic search
    with keyword boosting for important tax topics.
    """

    # Retrieve more candidates first, then rerank them
    candidate_count = min(10, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=candidate_count
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    important_terms = [
        "section 80c",
        "form 16",
        "hra",
        "tds",
        "income tax return",
        "capital gains",
        "advance tax",
        "new tax regime",
        "old tax regime",
        "gst"
    ]

    query_lower = query.lower()
    candidates = []

    for document, distance in zip(documents, distances):

        document_lower = document.lower()

        # Convert Chroma distance into a simple similarity score
        semantic_score = 1 / (1 + distance)

        # Give an additional score when the important topic
        # appears in both the question and the document
        keyword_boost = 0.0

        for term in important_terms:
            if term in query_lower and term in document_lower:
                keyword_boost += 1.0

        final_score = semantic_score + keyword_boost

        candidates.append({
            "document": document,
            "score": final_score
        })

    # Highest combined score first
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return [
        item["document"]
        for item in candidates[:number_of_results]
    ]

def call_llm(
    query: str,
    context: list[str]
) -> str:

    context_text = "\n\n".join(context)

    prompt = f"""
You are an Indian tax information assistant.

Answer the user's question using ONLY the
provided knowledge context.

If the answer is not available in the context,
say that the information is not available.

Do not invent tax rates, thresholds or deadlines.

Question:
{query}

Knowledge context:
{context_text}

Answer:
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get(
        "response",
        ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "LLM returned an empty response."
        )

    return answer


def write_log(
    query: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool
):

    estimated_cost = estimate_cost(
        input_tokens,
        output_tokens
    )

    log_entry = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "query": query,

        "latency_ms": round(
            latency_ms,
            2
        ),

        "input_tokens_estimated": input_tokens,

        "output_tokens_estimated": output_tokens,

        "total_tokens_estimated":
            input_tokens + output_tokens,

        "estimated_cost_usd":
            estimated_cost,

        "model": MODEL_NAME,

        "cache_hit": cache_hit
    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(log_entry)
            + "\n"
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "cache_entries": len(CACHE),
        "documents": collection.count()
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
def chat(request: Request, chat_request: ChatRequest):

    query = chat_request.query.strip()

    # Information consumed by observability middleware
    request.state.query = query
    request.state.input_tokens = estimate_tokens(query)

    if not query:

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    start_time = time.perf_counter()

    normalized_query = query.lower()

    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    with CACHE_LOCK:

        if normalized_query in CACHE:

            answer = CACHE[
                normalized_query
            ]

            cache_hit = True

        else:

            cache_hit = False

            # ------------------------------------------------
            # RETRIEVE CONTEXT
            # ------------------------------------------------

            context = retrieve_context(
                query
            )

            if not context:

                raise HTTPException(
                    status_code=404,
                    detail="No relevant information found."
                )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            answer = call_llm(
                query,
                context
            )

            CACHE[
                normalized_query
            ] = answer

    # Store information for observability middleware
    request.state.cache_hit = cache_hit
    request.state.output_tokens = estimate_tokens(answer)

    # --------------------------------------------------------
    # OBSERVABILITY
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    input_tokens = estimate_tokens(
        query
    )

    output_tokens = estimate_tokens(
        answer
    )


    # --------------------------------------------------------
    # SSE STREAMING
    # --------------------------------------------------------

    def generate_stream() -> Generator[str, None, None]:

        words = answer.split()

        for word in words:

            yield (
                "data: "
                + word
                + "\n\n"
            )

            time.sleep(0.015)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ============================================================
# STARTUP MESSAGE
# ============================================================

print(
    f"Indian Tax RAG API initialized "
    f"with {collection.count()} documents."
)
