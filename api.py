import os
import time
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uvicorn
from multihop_rag import app as rag_app

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Middleware and responses  libraries
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Streaming
from sse_starlette.sse import EventSourceResponse
import asyncio
import json as json_module

from uuid import uuid4
from fastapi.exceptions import RequestValidationError


load_dotenv()

# Langfuse setup

langfuse = Langfuse(public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),secret_key=os.getenv("LANGFUSE_SECRET_KEY"),host=os.getenv("LANGFUSE_BASE_URL"))

# CACHE SETUP

query_cache = {}
CACHE_MAX_SIZE = 100

def get_cached_response(question: str):
    return query_cache.get(question.lower().strip())

def cache_response(question: str, response: dict):
    if len(query_cache) >= CACHE_MAX_SIZE:
        oldest_key = next(iter(query_cache))
        del query_cache[oldest_key]
    query_cache[question.lower().strip()] = response

# Logging setup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
 
# RATE LIMITER

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


# REQUEST SIZE LIMIT MIDDLEWARE SETUP


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_bytes: int = 1_048_576):  # 1MB default
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            if int(content_length) > self.max_bytes:
                logger.warning(f"Request rejected: body too large ({content_length} bytes) "
                    f"from {request.client.host}")

                return JSONResponse(status_code=413,content={"detail": "Request body too large. Maximum size is 1MB."})

        body = await request.body()
        if len(body) > self.max_bytes:
            logger.warning(f"Request rejected: body too large ({len(body)} bytes) "
                f"from {request.client.host}")
            return JSONResponse(status_code=413,content={"detail": "Request body too large. Maximum size is 1MB."})

        return await call_next(request)

# FASTAPI 

app = FastAPI(title="Multi-Hop Adaptive RAG API",description="Production grade agentic RAG with self correction",version="1.0.0")

# Attaching rate limiter to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["Content-Type","Accept"])
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=1_048_576)

@app.on_event("startup")
async def startup_event():
    logger.info("Multi-Hop RAG API starting up")
    logger.info("Hybrid retriever: BM25 + Dense vectors")
    logger.info("Reranking: Cohere rerank-v3.5")
    logger.info("LLM: Llama 3.1 8B via Groq")
    logger.info("Observability: Langfuse enabled")
    logger.info("API ready to serve requests")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Multi-Hop RAG API shutting down")



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,content={"detail": "Invalid request. Please check your input and try again."})    

# GLOBAL EXCEPTION HANDLER

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):

    # Full error logged internally.
    logger.error(f"Unhandled error on {request.method} {request.url.path} "
        f"from {request.client.host}: {type(exc).__name__}: {str(exc)}")

    # sending clean message without exposing internals
    return JSONResponse(status_code=500,content={"detail": "An internal error occurred. Please try again later."})

# REQUEST AND RESPONSE MODELS

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    thread_id: Optional[str] = Field(default="default", max_length=50)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be blank or only whitespace")
        return stripped

class DocumentSource(BaseModel):
    content: str
    source: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[DocumentSource]
    attempt_count: int
    is_grounded: bool
    addresses_question: bool
    latency_seconds: float

class IngestRequest(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=500)

    @field_validator("file_path")
    @classmethod
    def block_path_traversal(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("Invalid file path ,directory traversal not allowed")
        if not v.endswith(".pdf"):
            raise ValueError("Only PDF files are supported")
        return v

# Routings
@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {"name": "Multi-Hop Adaptive RAG API","version": "1.0.0",
    "status": "running","docs": "/docs","health": "/health"}    

@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):           
    logger.info("Health check called")
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, body: QueryRequest):   
    logger.info(f"Query received: {body.question}")
    start_time = time.time()

    cached = get_cached_response(body.question)
    if cached:
        logger.info(f"Cache hit for: {body.question}")
        return QueryResponse(**cached)

    try:
        initial_state = {"question": body.question,
            "current_question": body.question,
            "documents": [],
            "answer": "",
            "attempt_count": 0,
            "is_grounded": False,
            "addresses_question": False}

        thread_id = body.thread_id or str(uuid4())
        langfuse_handler = CallbackHandler()
        config = {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

        final_state = rag_app.invoke(initial_state, config=config)

        docs = final_state["documents"]
        sources = [DocumentSource(content=doc.page_content, source=doc.metadata.get("source", "unknown"))
            for doc in docs]

        latency = round(time.time() - start_time, 2)
        logger.info(f"Query completed in {latency}s, attempts: {final_state['attempt_count']}")

        response_dict = {"question": body.question,
            "answer": final_state["answer"],
            "sources": sources,
            "attempt_count": final_state["attempt_count"],
            "is_grounded": final_state["is_grounded"],
            "addresses_question": final_state["addresses_question"],
            "latency_seconds": latency}
        cache_response(body.question, response_dict)

        return QueryResponse(**response_dict)

    except HTTPException:
        raise                                                           
    except ValueError as e:
        logger.error(f"Query validation error: {str(e)}")
        raise HTTPException(status_code=422, detail="Invalid query format. Please check your request.")

    except Exception as e:
        logger.error(f"Query failed: {type(e).__name__}: {str(e)}")   # full error in your logs
        raise HTTPException(status_code=500, detail="Query processing failed. Please try again.")


@app.post("/ingest")
@limiter.limit("5/minute")
async def ingest_document(request: Request, body: IngestRequest):
    logger.info(f"Ingest request received for: {body.file_path}")

    try:
        if not os.path.exists(body.file_path):
            raise HTTPException(status_code=404, detail="The specified file could not be found on the server.")

        
        from ingestion import ingest_document as run_ingestion, setup_qdrant, load_hashes, save_hashes
        #from ingestion import ingest_document, setup_qdrant, load_hashes, save_hashes
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",huggingfacehub_api_token=hf_token)

        client, vector_store = setup_qdrant(embeddings)
        saved_hashes = load_hashes()

        saved_hashes = run_ingestion(body.file_path, client, vector_store, saved_hashes)

        #saved_hashes = ingest_document(body.file_path, client, vector_store, saved_hashes)
        save_hashes(saved_hashes)

        return {"status": "success", "message": f"Document ingested: {body.file_path}"}

    except HTTPException:
        raise                                                           

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise HTTPException(status_code=404, detail="The specified file could not be found on the server.")

    except Exception as e:
        logger.error(f"Ingest failed: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail="Document ingestion failed. Please try again.")


@app.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream(request: Request, body: QueryRequest):          
    logger.info(f"Streaming query received: {body.question}")

    cached = get_cached_response(body.question)
    if cached:
        logger.info(f"Cache hit for: {body.question}")
        return QueryResponse(**cached)

    async def event_generator():
        try:
            start_time = time.time()

            yield {"event": "status", "data": json_module.dumps({"message": "Starting RAG pipeline..."})}

            initial_state = {"question": body.question,
                "current_question": body.question,
                "documents": [],
                "answer": "",
                "attempt_count": 0,
                "is_grounded": False,
                "addresses_question": False}

            thread_id = body.thread_id or str(uuid4())                 
            langfuse_handler = CallbackHandler()                        
            config = {"configurable": {"thread_id": thread_id},"callbacks": [langfuse_handler]}

            yield {"event": "status", "data": json_module.dumps({"message": "Retrieving documents..."})}

            final_state = rag_app.invoke(initial_state, config=config)

            answer = final_state["answer"]
            words = answer.split()

            yield {"event": "status", "data": json_module.dumps({"message": "Streaming answer..."})}

            for word in words:
                yield {"event": "token", "data": json_module.dumps({"token": word + " "})}
                await asyncio.sleep(0.05)

            docs = final_state["documents"]
            sources = [{"content": doc.page_content, "source": doc.metadata.get("source", "unknown")}
                for doc in docs]
            latency = round(time.time() - start_time, 2)

            # fix: cache the streamed response too
            response_dict = {"question": body.question,
                "answer": answer,
                "sources": sources,
                "attempt_count": final_state["attempt_count"],
                "is_grounded": final_state["is_grounded"],
                "addresses_question": final_state["addresses_question"],
                "latency_seconds": latency}
            cache_response(body.question, response_dict)

            yield {"event": "done", "data": json_module.dumps(response_dict)}

        except Exception as e:
            logger.error(f"Streaming failed: {type(e).__name__}: {str(e)})") 
            yield {"event": "error",
                "data": json_module.dumps({"error": "Streaming failed. Please try again."})  }

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
