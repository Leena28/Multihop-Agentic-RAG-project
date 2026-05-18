# MultiHop Agentic RAG with Self-RAG and Corrective RAG

An advanced Agentic Retrieval-Augmented Generation (RAG) system built using LangChain, LangGraph, FastAPI, Qdrant, Hybrid Search(BM25+Semantic Search), HuggingFace Embeddings, Groq llama-3.1-8b, Cohere Reranking, and Streamlit.

This project implements a production-style Self-RAG and Corrective RAG pipeline capable of:

- multi-hop reasoning across multiple PDFs,
- hybrid retrieval (BM25 + semantic search),
- retrieval grading,
- hallucination detection,
- answer grounding verification,
- query reformulation,
- answer quality evaluation,
- and self-correcting retrieval loops.

The LLM is used to:

- evaluate retrieved documents,
- judge retrieval relevance,
- detect hallucinations,
- verify groundedness,
- grade generated answers,
- reformulate failed queries,
- and decide whether another retrieval cycle is required.

The CI/CD pipeline integrates automated RAGAS evaluation before deployment, ensuring retrieval quality, faithfulness, and answer relevance are validated before production deployment to AWS EC2.
<img width="1311" height="533" alt="image" src="https://github.com/user-attachments/assets/e638fba5-1d23-486c-9e84-40de72b3a439" />

<img width="1345" height="596" alt="image" src="https://github.com/user-attachments/assets/a78d5cdf-95d1-4ab6-b33e-e084d9c0b35a" />


Live Demo:

* [Live Application](https://multihop-agentic-rag-project-resume.streamlit.app/)

GitHub Repository:

* [GitHub Repository](https://github.com/Leena28/Multihop-Agentic-RAG-project)

<img width="1365" height="588" alt="image" src="https://github.com/user-attachments/assets/84be262d-c927-4616-afb4-fd522f533182" />


---

# Project Overview

Traditional RAG pipelines often fail when:

* answers require reasoning across multiple documents,
* retrieval returns noisy chunks,
* hallucinations occur,
* or retrieved context is incomplete.

This project solves these problems by implementing an Agentic Self-Correcting RAG architecture using LangGraph.

The system can:

* retrieve information from multiple PDFs
* hop across 3 different documents
* evaluate retrieval quality
* reformulate failed queries
* rerank retrieved chunks
* detect hallucinations
* verify groundedness
* and retry generation until a grounded answer is produced.

The entire correction pipeline is orchestrated using LangGraph state graphs.

---

# Core Features

## Hybrid Retrieval Pipeline

* Semantic vector search
* BM25 sparse retrieval
* Hybrid Search (BM25 + Semantic Search)
* Ensemble Retriever for combining retrieval strategies
* Parent-child retrieval architecture
* Multi-hop document retrieval

## Vector Database

* Qdrant Vector Database
* Metadata-aware retrieval
* Persistent vector storage
* Structured filtering support

## Reranking

* Cohere Rerank API
* Retrieves broad candidate chunks
* Reranks top documents before generation

## Self-RAG / Corrective RAG Pipeline

* Retrieval grading
* Answer grading
* Hallucination checking
* Groundedness verification
* Query reformulation
* Retry-based corrective retrieval
* LLM-as-a-Judge architecture

The system uses the LLM powered by Groq llama-3.1-8b-instant to:

* evaluate retrieved vectors
* judge retrieval quality
* verify grounding
* detect hallucinations
* evaluate generated answers
* reformulate weak queries
* and decide whether another retrieval cycle is needed.

---

# LangGraph Agentic Workflow

Every self-correction stage is implemented using LangGraph.

The workflow includes:

* Query Decomposition
* Retrieval Node
* Reranking Node
* Retrieval Grading Node
* Hallucination Detection Node
* Groundedness Check Node
* Answer Grading Node
* Query Reformulation Node
* Retry Logic
* Final Synthesis Node

Conditional routing allows the system to dynamically decide the next action based on evaluation results.

---

# Multi-Hop Reasoning

The model can:

* retrieve from multiple PDFs,
* connect information across documents,
* perform multi-hop retrieval,
* synthesize answers from 3 different PDFs,
* and generate grounded final responses.

This enables complex question answering instead of single-document retrieval.

---

# Observability and Monitoring

## Langfuse Integration

Langfuse is used for:

* End-to-End Tracing
* Observability
* Request Monitoring
* Token Usage Tracking
* Retrieval Inspection
* Latency Tracking
* Pipeline Debugging

This helps analyze:

* what chunks were retrieved
* why an answer failed
* retrieval latency
* generation latency
* and total pipeline execution time.

---

# Backend Optimizations

## Caching

Caching is implemented in the backend to:

* store similar queries
* reduce repeated API calls
* improve latency
* and optimize response time.

## Rate Limiting

Rate limiting is implemented to:

* protect APIs
* prevent abuse
* and avoid excessive request spikes.

## Retry Control

A `max_retries` mechanism is implemented to:

* avoid infinite loops
* reduce API overuse
* and handle API rate limits safely.

---

# Tech Stack

| Category         | Technologies                            |
| ---------------- | --------------------------------------- |
| LLM              | Groq llama-3.1-8b-instant               |
| Framework        | LangChain, LangGraph                    |
| Backend          | FastAPI                                 |
| Frontend         | Streamlit                               |
| Vector DB        | Qdrant                                  |
| Embeddings       | HuggingFace Sentence Transformers       |
| Retrieval        | BM25, Hybrid Search, Ensemble Retrieval |
| Reranking        | Cohere Rerank API                       |
| Evaluation       | RAGAS                                   |
| Observability    | Langfuse                                |
| Deployment       | AWS EC2                                 |
| CI/CD            | GitHub Actions                          |
| Containerization | Docker                                  |

---

# Project Architecture

```text id="5uk0de"
User Query
    ↓
Query Decomposition
    ↓
Hybrid Retrieval (BM25 + Semantic Search)
    ↓
Ensemble Retrieval
    ↓
Cohere Reranking
    ↓
Retrieval Grading
    ↓
Groundedness Verification
    ↓
Hallucination Detection
    ↓
Answer Generation
    ↓
Answer Grading
    ↓
Query Reformulation (if needed)
    ↓
Retry Loop using LangGraph
    ↓
Final Grounded Response
```

---

# Evaluation

The system uses RAGAS evaluation metrics to measure:

* Context Precision
* Context Recall
* Faithfulness
* Answer Relevance

This allows objective evaluation of every retrieval and generation improvement. 

---

# Deployment

The project is deployed on AWS EC2 using:

* Docker
* GitHub Actions CI/CD
* Automated deployment pipeline via `deploy.yml`

GitHub Actions automatically:

* builds the application
* runs deployment steps
* and deploys updates to EC2 on push.

---

# Installation

## Clone Repository

```bash id="ydgr6m"
git clone https://github.com/Leena28/Multihop-Agentic-RAG-project.git
cd Multihop-Agentic-RAG-project
```

---

# Create Virtual Environment

```bash id="wt2q5j"
python -m venv venv
```

### Windows

```bash id="q4ksfj"
venv\Scripts\activate
```

### Linux / Mac

```bash id="v0h22h"
source venv/bin/activate
```

---

# Install Dependencies

```bash id="akdrf3"
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file:

```env id="db8s0g"
GROQ_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=your_langfuse_host
```

---

# Running the Application

## Start Backend

```bash id="dj4zzn"
uvicorn app.main:app --reload
```

## Start Streamlit Frontend

```bash id="j5lt1m"
streamlit run streamlit_app.py
```

---

# API Endpoints

| Method | Endpoint  | Description   |
| ------ | --------- | ------------- |
| POST   | `/query`  | Ask questions |
| POST   | `/ingest` | Upload PDFs   |
| GET    | `/health` | Health check  |

---

# Key Learning Outcomes

This project demonstrates:

* Self-RAG and Corrective RAG architectures
* LangGraph agentic workflows
* Multi-hop retrieval systems
* Hybrid search pipelines
* LLM-as-a-Judge patterns
* Retrieval grading systems
* Hallucination detection
* Grounded answer verification
* Production-grade observability
* API optimization and caching
* CI/CD deployment workflows
* AWS deployment practices

---

# Acknowledgements

Built using:

* [LangChain](https://www.langchain.com/)
* [LangGraph](https://www.langchain.com/langgraph)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Streamlit](https://share.streamlit.io/user/leena28)
* [Langfuse](https://langfuse.com/)
* [Cohere](https://cohere.com/rerank)
* [RAGAS](https://www.ragas.io/)

---

# Author

Developed by Leena Harpal as an advanced production-style Agentic AI and Self-Corrective RAG project.
www.linkedin.com/in/leena-harpal-b4327a141

Email-Leenaharpal96@gmail.com
